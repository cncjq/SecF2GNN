# utils.py (YOUR helper utilities, safe to live at repo root)
import random
from typing import List, Tuple, Dict, Optional, List
from pathlib import Path
import os
from collections import OrderedDict
import numpy as np
import torch
import subprocess
import re
import json
import copy


MPSPDZ_PATH = os.environ.get(
    "MP_SPDZ_PATH",
    "/home/jiaqiang/SecF2GNN/external/MP-SPDZ"
)

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if args.gpu != 'cpu':
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)

# def fair_metric(output, idx, labels, sens):
#     val_y = labels[idx].cpu().numpy()
#     idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()] == 0
#     idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()] == 1

#     idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
#     idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)

#     pred_y = (output[idx].squeeze() > 0).type_as(labels).cpu().numpy()
#     parity = abs(sum(pred_y[idx_s0]) / sum(idx_s0) - sum(pred_y[idx_s1]) / sum(idx_s1))
#     equality = abs(sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1))

#     return parity, equality

# def local_fairness(model, clients, device):
#     local_parity, local_equality = [], []
#     model = model.to(device)
#     model.eval()
#     for client in clients:
#         model.eval()
#         client.sub_G = client.sub_G.to(device)
#         client.sub_features = client.sub_features.to(device)
#         local_ouput = model(client.sub_G, client.sub_features)
#         local_parity_element, local_equality_element = fair_metric(local_ouput, client.idx_train_with_mask,
#                                                                    client.sub_labels, client.sub_sens)
#         local_parity.append(local_parity_element)
#         local_equality.append(local_equality_element)

#     mean_local_parity = sum(local_parity) / len(local_parity)
#     mean_local_equality = sum(local_equality) / len(local_equality)
#     print("mean_local_parity: ", mean_local_parity.item())
#     print("mean_local_equality: ", mean_local_equality.item())
#     return mean_local_parity.item(), mean_local_equality.item()

def save_local_params(out_dir, case_name, lambda1, tau, data_bias_weights, model_fairness_weights, local_weights):
    """
    Save key scalar and list parameters to human-readable .txt files locally.

    Produces:
      saved_params/
        ├── scalars_<case>.txt
        ├── data_bias_weights_<case>.txt
        ├── model_fairness_weights_<case>.txt
        └── local_weights_<case>.txt
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    with open(out_dir / f"scalars_{case_name}.txt", "w") as f:
        f.write(f"lambda1: {lambda1}\n")
        f.write(f"tau: {tau}\n")

    # --- Lists ---
    # def write_list(vals, name):
    #     path = out_dir / f"{name}_{case_name}.txt"
    #     with open(path, "w") as f:
    #         for v in (vals or []):
    #             f.write(f"{float(v)}\n")

    def write_list(vals, name):
        path = out_dir / f"{name}_{case_name}.txt"
        if vals is None:
            iterable = []
        elif isinstance(vals, torch.Tensor):
            iterable = vals.detach().cpu().view(-1).tolist()
        else:
            iterable = list(vals)

        with open(path, "w") as f:
            for v in iterable:
                f.write(f"{v}\n")


    write_list(data_bias_weights, "data_bias_weights")
    write_list(model_fairness_weights, "model_fairness_weights")

    def save_ordereddict_list_with_savetxt(local_weights, out_dir):
        """
        Save each OrderedDict (model.state_dict) in local_weights as layer-wise .txt files using np.savetxt.
        Works when local_weights = [OrderedDict(...), OrderedDict(...), ...].
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

        for i, state_dict in enumerate(local_weights):
            if not isinstance(state_dict, (dict, OrderedDict)):
                print(f"⚠️ Skipping index {i}: not a dict.")
                continue

            model_dir = out_dir / f"model_{i}"
            model_dir.mkdir(exist_ok=True)

            for name, tensor in state_dict.items():
                if not torch.is_tensor(tensor):
                    print(f"⚠️ Skipping {name}: not a tensor")
                    continue

                arr = tensor.detach().cpu().numpy()

                # np.savetxt only handles 1D/2D arrays — flatten higher dims
                if arr.ndim == 0:
                    arr = arr.reshape(1, 1)
                elif arr.ndim > 2:
                    arr = arr.reshape(arr.shape[0], -1)

                safe_name = name.replace('.', '_').replace('/', '_')
                path = model_dir / f"{safe_name}.txt"

                np.savetxt(path, arr, fmt="%.8f")
            print(f"✅ Saved model #{i} layers to {model_dir}")

# Example usage
# save_ordereddict_list_with_savetxt(local_weights, "weights_txt")
    save_ordereddict_list_with_savetxt(local_weights, out_dir)

    # write_list(local_weights, "local_weights")

    print(f"✅ Parameters saved under {out_dir}/ for case '{case_name}'")



import os
import re
import json
import subprocess
from pathlib import Path
import numpy as np
import torch

def f2gnn_agg(
    program="f2gnn_agg",
    mp_spdz_path="/home/jiaqiang/SecF2GNN/external/MP-SPDZ",
    parties=2,
    fracbits=16,             # only used if reveal() prints fixed-point integers (usually not needed)
    port_base=15000,
    verbose=True,
    only_party0=True,        # parse P0 only (recommended)
    default_fracbits=None,   # set to e.g. 16 if reveal() prints scaled integers
    transpose_for_linear=None, # e.g. {"classifier.weight"}
    save_txt=True,
    save_npz=True,
    out_dir="/home/jiaqiang/SecF2GNN/results/mpc_outputs",
):
    """
    Compile and execute Programs/Source/<program>.mpc using semi2k-party.x,
    parse tensors printed as:

        TENSOR <name> SHAPE r c
        %s
        TENSOR <name> SHAPE n
        %s
        TENSOR <name> SHAPE 1
        %s   (scalar, no brackets also OK)

    Returns:
        dict[str, torch.Tensor] mapping tensor names to tensors (float32, on CPU).
    Also saves .txt (np.savetxt) and .npz (np.savez_compressed) if requested.
    """
    if transpose_for_linear is None:
        transpose_for_linear = set()

    mp = Path(mp_spdz_path)
    binary = mp / "semi2k-party.x"

    # --- Check binaries ---
    if not binary.exists():
        raise FileNotFoundError(f"{binary} not found. Build it via 'make semi2k-party.x'")

    # --- Compile the MPC program ---
    if verbose:
        print(f"🔧 Compiling {program}.mpc ...")
    subprocess.run(["./compile.py", "-R", "128", program], cwd=mp, check=True)

    # --- Run all parties ---
    if verbose:
        print(f"🚀 Running {program} with {parties} parties ...")
    procs, outs, errs = [], [], []

    for p in range(parties):
        cmd = [
            "./semi2k-party.x",
            "-N", str(parties),
            "-p", str(p),
            "-pn", str(port_base),
            program,
            "-v",
        ]
        procs.append(subprocess.Popen(
            cmd, cwd=mp, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ))

    for proc in procs:
        out, err = proc.communicate()
        outs.append(out or "")
        errs.append(err or "")

    # --- Print results (optional) ---
    if verbose:
        print("\n=== STDOUT ===")
        for i, o in enumerate(outs):
            print(f"[P{i}]\n{o}")
        print("=== STDERR ===")
        for i, e in enumerate(errs):
            if e.strip():
                print(f"[P{i}] {e}")

    # --- Choose which output to parse ---
    to_parse = outs[0] if (only_party0 and outs) else "\n".join(outs)

    # --- Parsing helpers ---
    header_re = re.compile(r"^TENSOR\s+(\S+)\s+SHAPE\s+(\d+)(?:\s+(\d+))?\s*$", re.M)

    def harvest_nums(s: str):
        # supports floats & scientific notation
        return [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", s)]

    def extract_first_bracket_block(s: str):
        i = s.find('[')
        if i == -1:
            return None
        depth, j = 0, i
        while j < len(s):
            ch = s[j]
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    return s[i:j+1]
            j += 1
        return None

    # --- Parse tensors from stdout ---
    headers = list(header_re.finditer(to_parse))
    if not headers:
        raise ValueError("No 'TENSOR <name> SHAPE ...' headers found in stdout. "
                         "Ensure the MPC program prints headers before each reveal().")

    tensors = {}
    for idx, m in enumerate(headers):
        name = m.group(1)
        d0 = int(m.group(2))
        d1 = m.group(3)
        shape = (d0, int(d1)) if d1 else (d0,)

        # region to search: from after this header to before next header (or EOF)
        start = m.end()
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(to_parse)
        body = to_parse[start:end].strip()
        if not body:
            raise ValueError(f"No body found after header for tensor '{name}'.")

        # 1) Try bracketed JSON-like block first
        arr = None
        block = extract_first_bracket_block(body)
        if block is not None:
            try:
                data = json.loads(block)
                arr = np.array(data, dtype=float)
            except json.JSONDecodeError:
                nums = harvest_nums(block)
                need = int(np.prod(shape))
                if len(nums) < need:
                    raise ValueError(f"{name}: bracket block has {len(nums)} numbers, need {need}")
                arr = np.array(nums[:need], dtype=float).reshape(shape)
        else:
            # 2) No brackets — just collect numbers
            nums = harvest_nums(body)
            need = int(np.prod(shape))
            if len(nums) < need:
                if need == 1 and len(nums) >= 1:
                    nums = nums[:1]
                else:
                    # help debug by showing a small slice of body
                    snippet = body.splitlines()[:5]
                    raise ValueError(f"{name}: found {len(nums)} numbers, need {need} for shape {shape}. "
                                     f"Body head:\n" + "\n".join(snippet))
            arr = np.array(nums[:need], dtype=float).reshape(shape)

        # Optional: fixed-point rescaling (only if your reveal() is scaled integers)
        if default_fracbits is not None:
            arr = arr / (2 ** default_fracbits)

        t = torch.from_numpy(arr.astype(np.float32))
        if name in transpose_for_linear and t.ndim == 2:
            t = t.T.contiguous()
        tensors[name] = t  # keep on CPU

    if verbose:
        shapes = {k: tuple(v.shape) for k, v in tensors.items()}
        print(f"\n✅ MPC complete. Parsed tensors: {shapes}")

    # --- Save artifacts ---
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if save_txt:
        for name, t in tensors.items():
            arr = t.detach().cpu().numpy()
            txt_path = out_path / f"{name.replace('.', '_')}.txt"
            np.savetxt(txt_path, arr, fmt="%.8f", header=f"{name} shape={arr.shape}", comments="")
            if verbose:
                print(f"💾 Saved {name} -> {txt_path}")

    if save_npz:
        npz_path = out_path / "mpc_tensors.npz"
        np.savez_compressed(npz_path, **{k: v.detach().cpu().numpy() for k, v in tensors.items()})
        if verbose:
            print(f"💾 Saved bundle -> {npz_path}")

    # return tensors
    sd = build_state_dict(tensors)

    def fix_shapes_for_fairgnn(sd):
    # Fix classifier.weight: (64,) -> (1,64); (64,1) -> (1,64)
        if "classifier.weight" in sd:
            w = sd["classifier.weight"]
            if w.dim() == 1:                 # (64,)
                sd["classifier.weight"] = w.unsqueeze(0)      # -> (1,64)
            elif w.dim() == 2 and w.shape[1] == 1 and w.shape[0] == 64:  # (64,1)
                sd["classifier.weight"] = w.t().contiguous()  # -> (1,64)

        # Fix classifier.bias: scalar or (1,1) -> (1,)
        if "classifier.bias" in sd:
            b = sd["classifier.bias"]
            if b.dim() == 0:                  # ()
                sd["classifier.bias"] = b.view(1)             # -> (1,)
            elif b.dim() == 2 and b.shape == (1, 1):
                sd["classifier.bias"] = b.view(1)             # -> (1,)

        return sd

    sd = fix_shapes_for_fairgnn(sd)
    return sd



def build_state_dict(mpc_tensors):
    """
    Construct a state_dict directly from MPC outputs.
    Adds the prefix (like 'GNN.') to all GNN layers, if needed.

    Args:
        mpc_tensors: dict of torch.Tensor objects from f2gnn_agg()
            e.g. {'gc1.weight': T, 'gc1.bias': T, 'gc2.weight': T, ...}
        prefix: optional prefix for GNN layers.

    Returns:
        OrderedDict usable with model.load_state_dict()
    """
    state_dict = OrderedDict()

    for name, tensor in mpc_tensors.items():
        state_dict[name] = tensor

    return state_dict

def aggregate_weights(w, norm_w):
    """
    Returns the weighted weights.
    """
    w_new = copy.deepcopy(w[0])
    for key in w_new.keys():
        temp = 0
        for i in range(0, len(w)):
            temp += torch.mul(w[i][key], norm_w[i])
        w_new[key] = temp
    return w_new

def update_weights(w_local, w_server, beta):
    """
    Returns the new local weights.
    """
    w_new = copy.deepcopy(w_local)
    for key in w_new.keys():
        w_new[key] = torch.add(torch.mul(w_new[key], (1 - beta)), torch.mul(w_server[key], beta))
    return w_new

def aggregate_updates(updates, norm_w):
    """
    Aggregate local updates:
        Delta = sum_i norm_w[i] * updates[i]
    where updates[i] is a state_dict-like dict:
        updates[i][k] = local_weights[i][k] - round_global_weights[k]
    """
    upd_new = copy.deepcopy(updates[0])

    for key in upd_new.keys():
        temp = 0
        for i in range(len(updates)):
            temp += torch.mul(updates[i][key], norm_w[i])
        upd_new[key] = temp

    return upd_new


def apply_updates(w_server, agg_update):
    """
    Apply aggregated update to the server model:
        w_new = w_server + agg_update
    """
    w_new = copy.deepcopy(w_server)

    for key in w_new.keys():
        w_new[key] = torch.add(w_server[key], agg_update[key])

    return w_new
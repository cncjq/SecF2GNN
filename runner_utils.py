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


MPSPDZ_PATH = os.environ.get(
    "MP_SPDZ_PATH",
    "/home/jiaqiang/SecF2GNN/external/MP-SPDZ"
)

def set_seed(seed: int, gpu_flag) -> None:
    """
    Set all RNG seeds. Pass args.gpu (str or int) or 'cpu' in gpu_flag.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu_flag != 'cpu' and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def fair_metric(output: torch.Tensor,
                idx: torch.Tensor,
                labels: torch.Tensor,
                sens: torch.Tensor) -> Tuple[float, float]:
    """
    Returns (parity, equality) on the subset given by idx.
    """
    val_y = labels[idx].detach().cpu().numpy()
    idx_cpu = idx.detach().cpu().numpy()
    sens_np = sens.detach().cpu().numpy()

    idx_s0 = sens_np[idx_cpu] == 0
    idx_s1 = sens_np[idx_cpu] == 1

    idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)

    pred_y = (output[idx].squeeze() > 0).type_as(labels).detach().cpu().numpy()

    # small eps avoidance if a group has zero samples
    def safe_ratio(num, den): 
        den = max(int(den), 1)
        return float(num) / float(den)

    parity = abs(safe_ratio(sum(pred_y[idx_s0]), sum(idx_s0)) -
                 safe_ratio(sum(pred_y[idx_s1]), sum(idx_s1)))

    equality = abs(safe_ratio(sum(pred_y[idx_s0_y1]), sum(idx_s0_y1)) -
                   safe_ratio(sum(pred_y[idx_s1_y1]), sum(idx_s1_y1)))

    return float(parity), float(equality)


def local_fairness(model,
                   clients: List,
                   device: torch.device) -> Tuple[float, float]:
    """
    Evaluate mean local parity/equality over clients (each client has subgraph tensors).
    """
    local_parity, local_equality = [], []
    model = model.to(device)
    model.eval()

    for client in clients:
        client.sub_G = client.sub_G.to(device)
        client.sub_features = client.sub_features.to(device)

        with torch.no_grad():
            local_output = model(client.sub_G, client.sub_features)

        lp, le = fair_metric(local_output,
                             client.idx_train_with_mask,
                             client.sub_labels,
                             client.sub_sens)
        local_parity.append(lp)
        local_equality.append(le)

    mean_local_parity = float(sum(local_parity) / max(len(local_parity), 1))
    mean_local_equality = float(sum(local_equality) / max(len(local_equality), 1))
    print("mean_local_parity: ", mean_local_parity)
    print("mean_local_equality: ", mean_local_equality)
    return mean_local_parity, mean_local_equality

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
    def write_list(vals, name):
        path = out_dir / f"{name}_{case_name}.txt"
        with open(path, "w") as f:
            for v in (vals or []):
                f.write(f"{float(v)}\n")

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



# def f2gnn_agg(
#     mp_spdz_path="/home/jiaqiang/SecF2GNN/external/MP-SPDZ",
#     program="f2gnn_agg",
#     parties=2,
#     fracbits=16,
#     port_base=15000,
#     verbose=True,
#     expect_shape=None
# ):
#     """
#     Compile and execute Programs/Source/f2gnn_agg.mpc using semi2k-party.x.

#     Assumes Player-Data/Input-P0-0 is already written.
#     """

#     mp = Path(mp_spdz_path)
#     binary = mp / "semi2k-party.x"

#     # --- Check binaries ---
#     if not binary.exists():
#         raise FileNotFoundError(f"{binary} not found. Build it via 'make semi2k-party.x'")

#     # --- Compile the MPC program ---
#     print(f"🔧 Compiling {program}.mpc ...")
#     subprocess.run(["./compile.py", "-R", "128", program],
#                    cwd=mp, check=True)

#     # --- Run all parties ---
#     print(f"🚀 Running {program} with {parties} parties ...")
#     procs, outs, errs = [], [], []

#     for p in range(parties):
#         cmd = [
#             f"./semi2k-party.x",
#             "-N", str(parties),
#             "-p", str(p),
#             "-pn", str(port_base),
#             # "-F", str(fracbits),
#             program,
#         ]
#         procs.append(subprocess.Popen(
#             cmd, cwd=mp, text=True,
#             stdout=subprocess.PIPE, stderr=subprocess.PIPE
#         ))

#     for proc in procs:
#         out, err = proc.communicate()
#         outs.append(out or "")
#         errs.append(err or "")

#     # --- Print results ---
#     if verbose:
#         print("\n=== STDOUT ===")
#         for i, o in enumerate(outs):
#             print(f"[P{i}]\n{o}")
#         print("=== STDERR ===")
#         for i, e in enumerate(errs):
#             if e.strip():
#                 print(f"[P{i}] {e}")

#     # --- Extract first revealed number (if any) ---

#     def _extract_next_bracket_block(text, start_pos):
#         # Find the first '[' after start_pos, then return the balanced block as a string and end index.
#         i = text.find('[', start_pos)
#         if i == -1:
#             return None, start_pos
#         depth, j = 0, i
#         while j < len(text):
#             ch = text[j]
#             if ch == '[':
#                 depth += 1
#             elif ch == ']':
#                 depth -= 1
#                 if depth == 0:
#                     return text[i:j+1], j+1
#             j += 1
#         return None, start_pos  # unbalanced

#     def parse_tensors_from_stdout(to_parse, default_fracbits=None, device="cpu", transpose_for_linear=None):
#         """
#         Parses outputs like:

#         TENSOR gc1.weight SHAPE 276 64
#         [ [ ... ], [ ... ], ... ]
#         TENSOR gc1.bias SHAPE 64
#         0.1 0.2 0.3 ...  (or [0.1, 0.2, ...])
#         TENSOR classifier.bias SHAPE 1
#         -0.0123456        (scalar, no brackets)

#         Returns {name: torch.Tensor}.
#         """
#         if transpose_for_linear is None:
#             transpose_for_linear = set()

#         tensors = {}

#         # Find all headers and their spans
#         header_re = re.compile(r"^TENSOR\s+(\S+)\s+SHAPE\s+(\d+)(?:\s+(\d+))?\s*$", re.M)
#         headers = list(header_re.finditer(to_parse))

#         if not headers:
#             raise ValueError("No 'TENSOR <name> SHAPE ...' headers found in stdout.")

#         # Helper: harvest all numbers from a text chunk
#         def harvest_nums(s):
#             return [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", s)]

#         # Helper: first balanced bracketed block between [ ... ]
#         def extract_first_bracket_block(s):
#             i = s.find('[')
#             if i == -1:
#                 return None
#             depth, j = 0, i
#             while j < len(s):
#                 ch = s[j]
#                 if ch == '[':
#                     depth += 1
#                 elif ch == ']':
#                     depth -= 1
#                     if depth == 0:
#                         return s[i:j+1]
#                 j += 1
#             return None

#         for idx, m in enumerate(headers):
#             name = m.group(1)
#             d0 = int(m.group(2))
#             d1 = m.group(3)
#             shape = (d0, int(d1)) if d1 else (d0,)

#             # Slice the body for this tensor: from end of this header to start of next header (or EOF)
#             start = m.end()
#             end = headers[idx + 1].start() if idx + 1 < len(headers) else len(to_parse)
#             body = to_parse[start:end].strip()

#             if not body:
#                 raise ValueError(f"No body found after header for tensor '{name}'.")

#             # 1) Try bracketed JSON-like block first
#             block = extract_first_bracket_block(body)
#             arr = None
#             if block is not None:
#                 # Try JSON parse; if commas are missing, fall back to number harvesting
#                 try:
#                     data = json.loads(block)
#                     arr = np.array(data, dtype=float)
#                 except json.JSONDecodeError:
#                     nums = harvest_nums(block)
#                     size = int(np.prod(shape))
#                     if len(nums) < size:
#                         raise ValueError(f"{name}: bracket block has {len(nums)} nums, need {size}")
#                     arr = np.array(nums[:size], dtype=float).reshape(shape)
#             else:
#                 # 2) No brackets — collect all numbers from the body chunk
#                 nums = harvest_nums(body)
#                 size = int(np.prod(shape))
#                 if len(nums) < size:
#                     # Maybe the value is on the next non-empty line only; try harder by
#                     # grabbing until a blank line or just accept scalar for shape (1,)
#                     if size == 1 and len(nums) >= 1:
#                         nums = nums[:1]
#                     else:
#                         raise ValueError(f"{name}: found {len(nums)} numbers, need {size} for shape {shape}")
#                 arr = np.array(nums[:size], dtype=float).reshape(shape)

#             # If your reveal() printed fixed-point integers, rescale here:
#             # if default_fracbits is not None:
#             #     arr = arr / (2 ** default_fracbits)

#             t = torch.from_numpy(arr.astype(np.float32))
#             if name in transpose_for_linear and t.ndim == 2:
#                 t = t.T.contiguous()

#             tensors[name] = t.to(device)

#         return tensors

#     def save_tensors_text(tensors, out_dir="/home/jiaqiang/SecF2GNN/results/mpc_outputs/", fmt="%.8f"):
#         out = Path(out_dir)
#         out.mkdir(exist_ok=True)
#         for name, t in tensors.items():
#             arr = t.detach().cpu().numpy()
#             path = out / f"{name.replace('.', '_')}.txt"
#             # Add a helpful header with shape
#             np.savetxt(path, arr, fmt=fmt, header=f"{name} shape={arr.shape}", comments="")
#             print(f"💾 Saved {name} -> {path}")
    
    
    # After your MP-SPDZ run:
    # to_parse = outs[0]  # P0 output
    # tensors = parse_tensors_from_stdout(
    #     to_parse,
    #     # default_fracbits=16,        # uncomment only if reveal() yields fixed-point integers
    #     device="cpu",
    #     transpose_for_linear={"classifier.weight"}  # nn.Linear expects (out, in)
    # )

    # save_dir = Path("mpc_outputs")
    # save_dir.mkdir(exist_ok=True)

    # for name, tensor in tensors.items():
    #     arr = tensor.cpu().numpy()  # ensure numpy array
    #     filename = save_dir / f"{name.replace('.', '_')}.txt"
        
    #     # Save as plain text
    #     # fmt="%.8f" gives up to 8 decimals; adjust as needed.
    #     np.savetxt(filename, arr, fmt="%.8f")
        
    #     print(f"💾 Saved {name} → {filename}")
# Load into model
    # model.gc1.weight.data.copy_(tensors["gc1.weight"])
    # model.gc1.bias.data.copy_(tensors["gc1.bias"].view(-1))
    # model.gc2.weight.data.copy_(tensors["gc2.weight"])
    # model.gc2.bias.data.copy_(tensors["gc2.bias"].view(-1))
    # model.classifier.weight.data.copy_(tensors["classifier.weight"])
    # model.classifier.bias.data.copy_(tensors["classifier.bias"].view(-1))


    # After your MP-SPDZ run:
    # to_parse = outs[0]  # P0 output
    # tensors = parse_tensors_from_stdout(
    #     to_parse,
    #     # default_fracbits=16,        # uncomment only if reveal() yields fixed-point integers
    #     device="cpu",
    #     transpose_for_linear={"classifier.weight"}  # nn.Linear expects (out, in)
    # )

    # Load into model
    # model.gc1.weight.data.copy_(tensors["gc1.weight"])
    # model.gc1.bias.data.copy_(tensors["gc1.bias"].view(-1))
    # model.gc2.weight.data.copy_(tensors["gc2.weight"])
    # model.gc2.bias.data.copy_(tensors["gc2.bias"].view(-1))
    # model.classifier.weight.data.copy_(tensors["classifier.weight"])
    # model.classifier.bias.data.copy_(tensors["classifier.bias"].view(-1))


    # to_parse = outs[0]
    # nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", to_parse)]
    # if expect_shape:
    #     r, c = expect_shape
    #     if len(nums) < r * c:
    #         raise ValueError(f"Found {len(nums)} numbers, need {r*c} for shape {expect_shape}")
    #     mat = np.array(nums[: r * c], dtype=float).reshape(r, c)
    #     print(f"\n✅ MPC execution complete. Revealed matrix shape: {mat.shape}")
    #     return mat
    
    # rows = []
    # for line in to_parse.splitlines():
    #     line = line.strip()
    #     if line.startswith("[") and line.endswith("]"):
    #         try:
    #             rows.append(json.loads(line))
    #         except Exception:
    #             pass
    # if rows:
    #     mat = np.array(rows, dtype=float)
    #     print(f"\n✅ MPC execution complete. Revealed matrix shape: {mat.shape}")
    #     np.savetxt("/home/jiaqiang/SecF2GNN/results/mpc_outputs/GNN_gc1_weight.txt", mat.squeeze(0), fmt="%.6f")
    #     return mat
    
    # revealed = nums[0] if nums else None
    # print(f"\n⚠️ Could not parse a 2D array; returning first float: {revealed}")
    # return revealed

    # revealed = None
    # for o in outs:
    #     for line in o.splitlines():
    #         try:
    #             revealed = float(line.strip())
    #             break
    #         except ValueError:
    #             continue
    #     if revealed is not None:
    #         break

    # print(f"\n✅ MPC execution complete. Revealed value: {revealed}")
    # return revealed

import os
import re
import json
import subprocess
from pathlib import Path
import numpy as np
import torch

def f2gnn_agg(
    mp_spdz_path="/home/jiaqiang/SecF2GNN/external/MP-SPDZ",
    program="f2gnn_agg",
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

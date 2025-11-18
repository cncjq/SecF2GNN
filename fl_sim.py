# fl_sim.py
import os
import sys
import io
import copy
import time
import warnings
warnings.filterwarnings("ignore", message="Recommend creating graphs by `dgl.graph")

from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.multiprocessing as mp
from scipy.special import softmax
from scipy.spatial import distance
from torch_geometric.utils import subgraph, k_hop_subgraph
from sklearn.metrics import roc_auc_score

# --- use your renamed helpers to avoid collision with external utils ---
from runner_utils import set_seed, fair_metric

# ----------------- External imports helper -----------------
import importlib.util
def import_external(ext_src: Path):
    sys.path.insert(0, str(ext_src))
    # ensure external utils is registered as "utils"
    spec = importlib.util.spec_from_file_location("utils", str(ext_src / "utils.py"))
    ext_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext_utils)
    sys.modules["utils"] = ext_utils

    from options import args_parser
    from models import FairGNN
    from terminal import Client

    return {
        "args_parser": args_parser,
        "FairGNN": FairGNN,
        "Client": Client,
        "average_weights": ext_utils.average_weights,
        "aggregate_weights": ext_utils.aggregate_weights,
        "exp_details": ext_utils.exp_details,
        "load_pokec": ext_utils.load_pokec,
        "accuracy": ext_utils.accuracy,
        "density": ext_utils.density,
        "update_weights": ext_utils.update_weights,
        "model_fair_metric": ext_utils.model_fair_metric,
    }

# ----------------- (De)serialization helpers -----------------
def state_dict_to_bytes(sd: Dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k,v in sd.items()}, buf)
    return buf.getvalue()

def bytes_to_state_dict(b: bytes) -> Dict[str, torch.Tensor]:
    buf = io.BytesIO(b)
    return torch.load(buf, map_location="cpu")

# ----------------- Client worker process -----------------
def client_worker(cid: int,
                  device_str: str,
                #   args_dict: Dict[str, Any],
                  args,
                  ext_src: str,
                  data_blob: Dict[str, Any],
                  in_q: mp.Queue,
                  out_q: mp.Queue):
    """
    One process per client.
    Waits for global weights -> trains locally -> returns updated weights and metrics.
    """
    # local imports / setup
    ext = import_external(Path(ext_src))
    # args = args_dict  # plain dict sent by parent
    torch.set_num_threads(1)  # avoid oversubscription
    device = torch.device(device_str)

    # reconstruct tensors for this client
    # data_blob contains tensors moved to CPU before sending
    features = data_blob["features"]
    labels   = data_blob["labels"]
    sens     = data_blob["sens"]
    G        = data_blob["G"]
    idx_train = data_blob["idx_train"]
    idx_val   = data_blob["idx_val"]
    idx_test  = data_blob["idx_test"]
    edge_index = data_blob["edge_index"]

    # build client’s ego-subgraph from the seed provided in message or from precomputed in data_blob
    # here we precomputed sub_nodes/sub_edges per cid in parent to keep it simple:
    sub_nodes = data_blob["sub_nodes_list"][cid]
    sub_edges = data_blob["sub_edges_list"][cid]

    # Create the "Client" object from the external repo
    cl = ext["Client"](features, labels, sens, sub_nodes, sub_edges, idx_train, idx_val, idx_test, args)

    # Create local model
    local_model = ext["FairGNN"](nfeat=features.shape[1], args=args)
    local_model = local_model.to(device)

    # move some tensors to device once (subgraph tensors can be updated per round if needed)
    cl.sub_G        = cl.sub_G.to(device)
    cl.sub_features = cl.sub_features.to(device)
    cl.sub_labels   = cl.sub_labels.to(device)
    cl.sub_sens     = cl.sub_sens.to(device)

    while True:
        msg = in_q.get()
        if msg is None:
            # shutdown
            break

        round_idx = msg["round"]
        global_sd = bytes_to_state_dict(msg["weights"])
        update_method = msg["update_method"]
        tau_combine   = msg["tau_combine"]

        # load global weights (optionally "combine" with JS distance)
        if update_method == "combine":
            # need a global model to compute y_global
            global_model = ext["FairGNN"](nfeat=features.shape[1], args=args)
            global_model.load_state_dict(global_sd)
            global_model = global_model.to(device)
            with torch.no_grad():
                y_global = global_model(cl.sub_G, cl.sub_features)[cl.idx_train_with_mask]

            with torch.no_grad():
                y_local  = local_model(cl.sub_G, cl.sub_features)[cl.idx_train_with_mask]

            if len(y_global) >= 1 and len(y_local) >= 1:
                preds_global = (y_global > 0).type_as(cl.sub_labels)
                preds_local  = (y_local  > 0).type_as(cl.sub_labels)
                y_g = softmax([float(v) / tau_combine for v in preds_global])
                y_l = softmax([float(v) / tau_combine for v in preds_local])
                Beta = distance.jensenshannon(y_g, y_l)
                if not torch.is_tensor(Beta):
                    Beta = torch.tensor(Beta, dtype=torch.float32, device=device)
                new_local_sd = ext["update_weights"](local_model.state_dict(), global_sd, Beta)
                local_model.load_state_dict(new_local_sd)
            else:
                local_model.load_state_dict(copy.deepcopy(global_sd))
        else:
            local_model.load_state_dict(copy.deepcopy(global_sd))

        # local training
        for _ in range(args["local_ep"]):
            local_model.train()
            local_model.optimize(cl.sub_G, cl.sub_features, cl.sub_labels, cl.sub_sens, cl.idx_train_with_mask)

        # metrics to help server weighting (parity/equality, acc)
        with torch.no_grad():
            local_model.eval()
            pred = local_model(cl.sub_G, cl.sub_features)
            acc = ext["accuracy"](pred[cl.idx_val_with_mask], cl.sub_labels[cl.idx_val_with_mask])
            # parity/equality on client’s test partition
            parity, equality = ext["model_fair_metric"](
                (pred[cl.idx_test_with_mask].squeeze() > 0).type_as(cl.sub_labels),
                cl.sub_labels[cl.idx_test_with_mask],
                cl.sub_sens[cl.idx_test_with_mask]
            )

        # send back
        out_q.put({
            "round": round_idx,
            "cid": cid,
            "weights": state_dict_to_bytes(local_model.state_dict()),
            "acc": float(acc),
            "parity": float(parity),
            "equality": float(equality),
            "intra_ratio": float(cl.intra_ratio),
            "inter_ratio": float(cl.inter_ratio),
            "num_nodes": int(sub_nodes.shape[0]),
        })

# ----------------- Server/orchestrator -----------------
def run_fl(rounds: int,
           num_clients: int,
           args,
           ext,
           device: torch.device,
           G, features, labels, sens,
           idx_train, idx_val, idx_test,
           edge_index,
           aggregate_method: str):
    """
    Spawns client processes, runs FL rounds, aggregates weights, returns final model and logs.
    """
    # global model on server
    global_model = ext["FairGNN"](nfeat=features.shape[1], args=args)
    global_sd = global_model.state_dict()

    # Pre-build client ego-subgraphs on server (keep it deterministic)
    rng = np.random.default_rng(args.seed)
    sub_nodes_list, sub_edges_list = [], []
    for _ in range(num_clients):
        sub_nodes = torch.LongTensor([1,1])
        while sub_nodes.shape[0] <= 2:
            # pick an ego from train set
            ego_np = int(rng.choice(idx_train.cpu().numpy()))
            ego_idx = torch.tensor([ego_np], dtype=torch.long, device=edge_index.device)
            sub_nodes, sub_edges, _, _ = k_hop_subgraph(ego_idx, num_hops=args.num_hops, edge_index=edge_index)
        sub_nodes_list.append(sub_nodes.cpu())
        sub_edges_list.append(sub_edges.cpu())

    # prepare shared queues
    in_queues = [mp.Queue(maxsize=1) for _ in range(num_clients)]
    out_queue = mp.Queue()

    # package common data once (CPU tensors)
    data_blob = dict(
        features = torch.as_tensor(np.array(features)).cpu(),
        labels   = torch.as_tensor(labels).cpu(),
        sens     = torch.as_tensor(sens).cpu(),
        G        = G.cpu(),
        idx_train = idx_train.cpu(),
        idx_val   = idx_val.cpu(),
        idx_test  = idx_test.cpu(),
        edge_index = edge_index.cpu(),
        sub_nodes_list = sub_nodes_list,
        sub_edges_list = sub_edges_list,
    )

    # spin up client processes
    # args_dict = vars(args).copy()
    ext_src = str(Path(__file__).resolve().parent / "external" / "F2GNN" / "src")
    procs = []
    for cid in range(num_clients):
        # You can pin all clients to CPU to avoid GPU contention; or keep cuda:0 sequentially
        p = mp.Process(
            target=client_worker,
            args=(
                cid,
                "cuda:0" if (args.gpu != "cpu" and torch.cuda.is_available()) else "cpu",
                # args_dict,
                args,
                ext_src,
                data_blob,
                in_queues[cid],
                out_queue,
            ),
            daemon=True,
        )
        p.start()
        procs.append(p)

    # run rounds
    history = []
    for r in range(rounds):
        # Broadcast current global weights
        msg = {
            "round": r,
            "weights": state_dict_to_bytes(global_sd),
            "update_method": args.update_method,
            "tau_combine": args.tau_combine,
        }
        for q in in_queues:
            q.put(msg)

        # Collect updates
        updates = []
        for _ in range(num_clients):
            updates.append(out_queue.get())

        # Rebuild list in client id order, also gather metrics for weighting
        updates.sort(key=lambda x: x["cid"])
        local_sds = [bytes_to_state_dict(u["weights"]) for u in updates]
        local_par = [u["parity"] for u in updates]
        local_eq  = [u["equality"] for u in updates]
        inter = [u["inter_ratio"] for u in updates]
        intra = [u["intra_ratio"] for u in updates]

        # Simple average or your custom weighting
        if aggregate_method == "avg":
            global_sd = ext["average_weights"](local_sds)
        else:
            if aggregate_method == "1-|inter-intra|":
                diff = [abs(inter[i] - intra[i]) for i in range(num_clients)]
                w1 = softmax([(1 - d) for d in diff])  # numpy list
                w2 = softmax([1.0 / (local_par[i] + local_eq[i] + 1e-5) for i in range(num_clients)])
                norm = torch.tensor([(args.lambda1 * a + b) / args.tau for a, b in zip(w1, w2)], dtype=torch.float32)
                norm = torch.softmax(norm, dim=-1).tolist()
                # use external aggregator that expects weights and a vector
                wvec = torch.tensor(norm, dtype=torch.float32)
                global_sd = ext["aggregate_weights"](local_sds, wvec.to(torch.device("cpu")))
            else:
                raise ValueError("Unknown aggregate_method")

        # Log round summary (server-side eval optional)
        history.append({
            "round": r,
            "mean_parity": float(np.mean(local_par)),
            "mean_equality": float(np.mean(local_eq)),
        })
        print(f"[Server] Round {r}: mean parity={history[-1]['mean_parity']:.4f}, "
              f"mean equality={history[-1]['mean_equality']:.4f}")

    # shutdown clients
    for q in in_queues:
        q.put(None)
    for p in procs:
        p.join(timeout=5)

    # load final weights into a model for downstream eval
    final_model = ext["FairGNN"](nfeat=features.shape[1], args=args)
    final_model.load_state_dict(global_sd)
    return final_model, history

# ----------------- Entry point: data prep + run -----------------
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    mp.set_start_method("spawn", force=True)  # required for CUDA safety across processes

    repo_root = Path(__file__).resolve().parent
    ext_src = repo_root / "external" / "F2GNN" / "src"
    ext = import_external(ext_src)

    args = ext["args_parser"]()
    ext["exp_details"](args)
    set_seed(args.seed, args.gpu)

    device = torch.device(f"cuda:{args.gpu}" if (args.gpu != 'cpu' and torch.cuda.is_available()) else "cpu")
    print("device:", device)

    # ----- load dataset using external loader, but with absolute path to external data -----
    if args.dataset == 'pokec-z':
        dataset = 'region_job'; sens_attr = "region"; predict_attr = "I_am_working_in_field"
        data_path = str((ext_src.parent / "data" / "pokec").resolve())
    elif args.dataset == 'pokec-n':
        dataset = 'region_job_2'; sens_attr = "region"; predict_attr = "I_am_working_in_field"
        data_path = str((ext_src.parent / "data" / "pokec").resolve())
    elif args.dataset == 'NBA':
        dataset = 'nba'; sens_attr = "country"; predict_attr = "SALARY"
        data_path = str((ext_src.parent / "data" / "NBA").resolve())
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    graph, G, adj, features, labels, sens, num_nodes, edge_index = ext["load_pokec"](
        dataset, sens_attr, predict_attr, args, path=data_path
    )

    # train/val/test split (same as before)
    label_idx = np.where(labels >= 0)[0]
    sens_idx = set(np.where(sens >= 0)[0])
    idx_used = np.asarray(list(sens_idx & set(label_idx)))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx_used)

    idx_train = torch.LongTensor(idx_used[:int(0.5 * len(idx_used))])
    idx_val   = torch.LongTensor(idx_used[int(0.5 * len(idx_used)):int(0.75 * len(idx_used))])
    idx_test  = torch.LongTensor(idx_used[int(0.75 * len(idx_used)):])
    edge_index = torch.LongTensor(graph.edge_index)

    # Run the FL simulation
    rounds = args.epochs  # you can decouple rounds from epochs if you want
    final_model, history = run_fl(
        rounds=rounds,
        num_clients=args.ego_number,
        args=args,
        ext=ext,
        device=device,
        G=G,
        features=features,
        labels=labels,
        sens=sens,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        edge_index=edge_index,
        aggregate_method=args.aggregate_method
    )

    print("FL done. Final rounds:", len(history))

if __name__ == "__main__":
    main()

# client.py
import io, os, sys, time, base64
import numpy as np
import requests
import torch
from pathlib import Path
from scipy.special import softmax
from torch_geometric.utils import k_hop_subgraph
from sklearn.metrics import roc_auc_score

# ---- external imports ----
import importlib.util
ROOT = Path(__file__).resolve().parent
EXT_SRC = ROOT / "external" / "F2GNN" / "src"
sys.path.insert(0, str(EXT_SRC))
spec = importlib.util.spec_from_file_location("utils", str(EXT_SRC / "utils.py"))
ext_utils = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext_utils)
from models import FairGNN
from terminal import Client as FClient
from utils import load_pokec, accuracy

SERVER = os.getenv("SERVER_URL", "http://server:5000")
CID = int(os.getenv("CID", "0"))
DATASET = os.getenv("DATASET", "pokec-z")
EGO_HOPS = int(os.getenv("NUM_HOPS", "3"))
LOCAL_EP = int(os.getenv("LOCAL_EP", "1"))
SEED = int(os.getenv("SEED", "31"))

def _serialize_sd(sd):
    buf = io.BytesIO(); torch.save(sd, buf)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def main():
    # 1) Fetch/init global model
    r = requests.get(f"{SERVER}/ready").json()
    # minimal args; adjust to match your real parser if needed
    from types import SimpleNamespace
    args = SimpleNamespace(num_hidden=64, model="GCN", dropout=0.5, penalty=0.001,
                           gpu='cpu', local_ep=LOCAL_EP, tau_combine=0.01,
                           update_method="uniform", lambda1=0.5, tau=4, patience=20)

    # 2) Load dataset from external folder
    # map dataset names
    if DATASET == 'pokec-z':
        dataset='region_job'; sens_attr="region"; predict_attr="I_am_working_in_field"
        data_path = str((EXT_SRC.parent / "data" / "pokec").resolve())
    elif DATASET == 'pokec-n':
        dataset='region_job_2'; sens_attr="region"; predict_attr="I_am_working_in_field"
        data_path = str((EXT_SRC.parent / "data" / "pokec").resolve())
    else:
        raise ValueError("only pokec-z/pokec-n in this minimal client")

    graph, G, adj, features, labels, sens, num_nodes, edge_index = load_pokec(
        dataset, sens_attr, predict_attr, args, path=data_path
    )

    # 3) Train/val/test split (deterministic per CID)
    rng = np.random.default_rng(SEED)
    label_idx = np.where(labels >= 0)[0]
    sens_idx = set(np.where(sens >= 0)[0])
    idx_used = np.asarray(list(sens_idx & set(label_idx)))
    rng.shuffle(idx_used)
    idx_train = torch.LongTensor(idx_used[:int(0.5 * len(idx_used))])
    idx_val   = torch.LongTensor(idx_used[int(0.5 * len(idx_used)):int(0.75 * len(idx_used))])
    idx_test  = torch.LongTensor(idx_used[int(0.75 * len(idx_used)):])
    edge_index = torch.LongTensor(graph.edge_index)

    # 4) Ego subgraph: pick ego based on CID to diversify clients
    rng2 = np.random.default_rng(SEED + CID)
    sub_nodes = torch.LongTensor([1,1])
    while sub_nodes.shape[0] <= 2:
        ego_np = int(rng2.choice(idx_train.numpy()))
        ego_idx = torch.tensor([ego_np], dtype=torch.long, device=edge_index.device)
        sub_nodes, sub_edges, _, _ = k_hop_subgraph(ego_idx, num_hops=EGO_HOPS, edge_index=edge_index)

    client = FClient(features, labels, sens, sub_nodes, sub_edges, idx_train, idx_val, idx_test, args)

    # 5) Ensure server has a global model initialized
    requests.post(f"{SERVER}/init", params={"nfeat": int(features.shape[1])})

    # 6) FL loop: fetch global weights -> local train -> post updated weights
    for rnd in range(int(os.getenv("ROUNDS", "5"))):
        g = requests.get(f"{SERVER}/global").json()
        round_from_server = g["round"]
        sd_global = torch.load(io.BytesIO(base64.b64decode(g["state_dict_b64"].encode("utf-8"))), map_location="cpu")

        local_model = FairGNN(nfeat=features.shape[1], args=args)
        local_model.load_state_dict(sd_global)

        # put tensors on CPU for simplicity in this container; switch to cuda if you build GPU images
        client.sub_G        = client.sub_G
        client.sub_features = client.sub_features
        client.sub_labels   = client.sub_labels
        client.sub_sens     = client.sub_sens

        # local train
        for _ in range(LOCAL_EP):
            local_model.train()
            local_model.optimize(client.sub_G, client.sub_features, client.sub_labels,
                                 client.sub_sens, client.idx_train_with_mask)

        # (optional) quick val metric before sending
        local_model.eval()
        with torch.no_grad():
            out = local_model(client.sub_G, client.sub_features)
            acc = accuracy(out[client.idx_val_with_mask], client.sub_labels[client.idx_val_with_mask])
        print(f"[client {CID}] round {rnd} local val acc={float(acc):.4f}")

        # send full state_dict (simple average on server)
        payload = {
            "cid": CID,
            "round": round_from_server,
            "state_dict_b64": _serialize_sd(local_model.state_dict())
        }
        r = requests.post(f"{SERVER}/update", json=payload).json()
        print(f"[client {CID}] server reply:", r)

        time.sleep(float(os.getenv("SLEEP_BETWEEN_ROUNDS", "0.1")))

if __name__ == "__main__":
    main()

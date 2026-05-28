# main.py (YOUR launcher that uses your utils + the external repo)
import os
import sys
import copy
import warnings
from pathlib import Path
from time import perf_counter as t

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.special import softmax
from scipy.spatial import distance
from sklearn.metrics import roc_auc_score

# --- YOUR helpers (import these FIRST to avoid name clash) ---
import random
from runner_utils import f2gnn_agg, save_local_params, aggregate_weights, update_weights, aggregate_updates, apply_updates
# from opacus.accountants import RDPAccountant

# Torch Geometric utils

from torch_geometric.utils import subgraph, k_hop_subgraph

import math

# from Helper_Functions import Vname_to_FeedPname, Vname_to_Pname, check_validaity_of_FLAGS, create_save_dir, \
#     global_step_creator, load_from_directory_or_initialize, bring_Accountant_up_to_date, save_progress, \
#     WeightsAccountant, print_loss_and_accuracy, print_new_comm_round, PrivAgent, Flag

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu != 'cpu':
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

def fair_metric(output, idx, labels, sens):
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()] == 0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()] == 1

    idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)

    pred_y = (output[idx].squeeze() > 0).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0]) / sum(idx_s0) - sum(pred_y[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1))

    return parity, equality

def local_fairness(model, clients, device):
    local_parity, local_equality = [], []
    model = model.to(device)
    model.eval()
    for client in clients:
        model.eval()
        client.sub_G = client.sub_G.to(device)
        client.sub_features = client.sub_features.to(device)
        local_ouput = model(client.sub_G, client.sub_features)
        local_parity_element, local_equality_element = fair_metric(local_ouput, client.idx_train_with_mask,
                                                                   client.sub_labels, client.sub_sens)
        local_parity.append(local_parity_element)
        local_equality.append(local_equality_element)

    mean_local_parity = sum(local_parity) / len(local_parity)
    mean_local_equality = sum(local_equality) / len(local_equality)
    print("mean_local_parity: ", mean_local_parity.item())
    print("mean_local_equality: ", mean_local_equality.item())
    return mean_local_parity.item(), mean_local_equality.item()

def sd_to_update(local_sd, global_sd):
    upd = {}
    for k, v in local_sd.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            upd[k] = (v.detach() - global_sd[k].detach()).clone()
    return upd

def get_l2_norm(upd, eps=1e-12):
     # Accept torch.nn.Module as input
    if isinstance(upd, torch.nn.Module):
        upd = upd.state_dict()

    sq = None
    for v in upd.values():
        # only include floating tensors (skip integer buffers, counters, etc.)
        if torch.is_tensor(v) and torch.is_floating_point(v):
            vv = v.reshape(-1).float()
            s = vv.pow(2).sum()
            sq = s if sq is None else sq + s

    if sq is None:
        sq = torch.tensor(0.0)

    return torch.sqrt(sq + eps)

def clip_update(upd, clip_norm):
    n = get_l2_norm(upd)
    scale = min(1.0, (clip_norm / n).item())
    # scale = 1
    if scale < 1.0:
        for k in upd:
        # scale = min(1.0, (clip_norm / n[k]))
        # if scale < 1.0:
            upd[k] = upd[k] * scale
    # return upd, float(n.detach().cpu()), scale
    return upd

def _to_update_dict(x):
    """Accept a model or dict, return dict[str, Tensor]."""
    if isinstance(x, torch.nn.Module):
        return x.state_dict()
    return x

def get_clip_norm(local_items, min_clip=1e-12):
#     """
#     Return median L2 norm over a list of local items.

#     local_items can be:
#       - list of client models (torch.nn.Module)
#       - list of state_dicts / update dicts (dict[str, Tensor])

#     Note: This computes norms of whatever dict you pass in.
#           If you pass full weights (state_dict), this is 'weight norm'.
#           If you pass updates Δ_i, this is 'update norm' (what clipping usually wants).
#     """
    norms = []

    for item in local_items:
        upd = _to_update_dict(item)

        # compute norm, but safely skip non-floating tensors
        n = get_l2_norm(upd)
        if torch.is_tensor(n):
            n = n.detach().cpu().item()
        norms.append(n)

    clip_norm = max(float(np.median(norms)) if norms else 0.0, float(min_clip)) #median

    ### find the largest norm - smallest norm

    # if len(norms) == 0:
    #     clip_norm = float(min_clip)
    # else:
    #     max_norm = max(norms)
    #     min_norm_val = min(norms)
    #     clip_norm = max(max_norm - min_norm_val, float(min_clip))

    return clip_norm, norms

def add_noise_to_update(upd, noise_multiplier, clip_norm):
    """Add N(0, (sigma*C)^2) noise elementwise to each tensor."""
    if noise_multiplier <= 0:
        return upd
    std = noise_multiplier * clip_norm
    for k, v in upd.items():
        # print(clip_norm)
        # print(std)
        # print(upd[k])
        upd[k] = v + torch.normal(mean=0.0, std=std, size=v.shape, device=v.device, dtype=v.dtype)
    # upd = upd + torch.normal(mean=0.0, std=std, size=upd.shape, device=upd.device, dtype=upd.dtype)
    return upd

# def aggregate_updates(updates, weights, noise_multiplier, clip_norm):
#     """Aggregate dict updates. If weights provided, they must be nonnegative; we normalize them."""
#     keys = list(updates[0].keys())
#     device = next(iter(updates[0].values())).device
#     out = {k: torch.zeros_like(updates[0][k]) for k in keys}

#     if weights is None:
        # for upd in updates:
        #     for k in keys:
        #         out[k] += upd[k]
        # m = max(len(updates), 1)
        # for k in keys:
        #     out[k] /= m
    #     print("error for weights")
    #     exit(99)
    # else:
    #     w = torch.as_tensor(weights, device=device, dtype=torch.float32)
    #     w = w / (w.sum() + 1e-12)
    #     for upd, wi in zip(updates, w):
    #         for k in keys:
    #             # out[k] += add_noise_to_update(upd[k], noise_multiplier, clip_norm) * wi            
    #             out[k] += wi * upd[k]
    # return out

# def apply_update(global_sd, upd):
#     """Return new global state dict after adding update to float params."""
#     new_sd = {}
#     for k, v in global_sd.items():
#         if torch.is_floating_point(v) and k in upd:
#             new_sd[k] = (v + upd[k]).detach()
#         else:
#             new_sd[k] = v
#     return new_sd

# def dp_server_aggregate(local_state_dicts, 
#                         norms,
#                         # global_state_dict, 
#                         clip_norm, noise_multiplier, weights=None):
#     """
#     DP server aggregation for client-level privacy:
#     XXXXX  - build client updates Δ_i = w_i - w_global
#       - clip each Δ_i to ||Δ_i||2 <= C
#       - aggregate (mean or weighted)
#       - add Gaussian noise to aggregate
#       - apply to global
#     """
#     # updates = [sd_to_update(sd, global_state_dict) for sd in local_state_dicts]

#     clipped_updates = []
#     # norms = []
#     # scales = []
#     for upd in local_state_dicts:
#         cu = clip_update(upd, clip_norm)
#         clipped_updates.append(cu)

#     agg = aggregate_updates(clipped_updates, weights, noise_multiplier, clip_norm)
#     new_global = add_noise_to_update(agg, noise_multiplier, clip_norm)
#     # new_global = apply_update(global_state_dict, agg)
#     return new_global
#     # return agg

############## adding for update dp ##############

# def sd_to_update(local_sd, global_sd):
#     upd = {}
#     for k, v in local_sd.items():
#         gv = global_sd[k]
#         if torch.is_tensor(v) and torch.is_floating_point(v):
#             upd[k] = (v - gv).detach()
#     return upd

# def apply_update(global_sd, upd):
#     new_sd = {}
#     for k, v in global_sd.items():
#         if torch.is_tensor(v) and torch.is_floating_point(v) and k in upd:
#             new_sd[k] = (v + upd[k]).detach()
#         else:
#             new_sd[k] = v
#     return new_sd

def dp_server_aggregate(local_updates,
                        global_state_dict,
                        clip_norm, 
                        noise_multiplier,
                        weights=None):
    # 1) build updates Δ_i = w_i - w_global
    # updates = [sd_to_update(sd, global_state_dict) for sd in local_state_dicts]

    # 2) clip each update
    clipped_updates = []
    for upd in local_updates:
        cu = clip_update(upd, clip_norm)   # your clip_update uses get_l2_norm(dict)
        clipped_updates.append(cu)

    # 3) aggregate updates (weighted or uniform)
    agg = aggregate_updates(clipped_updates, weights)
    # agg = aggregate_updates(local_updates, weights)

    # print(clip_norm)
    # 4) add noise to aggregated update
    noisy_agg = add_noise_to_update(agg, noise_multiplier, 1/10 * clip_norm)

    # 5) apply update to global weights
    new_global = apply_updates(global_state_dict, noisy_agg)
    # new_global = apply_updates(global_state_dict, agg)
    return new_global

# def aggregate_updates(updates, weights, noise_multiplier, clip_norm):
#     keys = list(updates[0].keys())
#     out = {k: torch.zeros_like(updates[0][k]) for k in keys}

#     if weights is None:
#         m = max(len(updates), 1)
#         for upd in updates:
#             for k in keys:
#                 out[k] += upd[k]
#         for k in keys:
#             out[k] /= m
#     else:
#         device = next(iter(updates[0].values())).device
#         w = torch.as_tensor(weights, device=device, dtype=torch.float32)
#         w = w / (w.sum() + 1e-12)
#         for upd, wi in zip(updates, w):
#             for k in keys:
#                 out[k] += wi * upd[k]
#     return out

def rdp_epsilon_approx(q: float, sigma: float, steps: int, delta: float) -> float:
    """
    Approximate RDP->(eps,delta) for subsampled Gaussian mechanism.
    Works best when q is small (<= 0.05). Conservative when q is larger.
    """
    if sigma <= 0:
        return float("inf")
    if not (0 < q <= 1):
        raise ValueError("q must be in (0,1].")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1).")

    # Candidate Renyi orders; extend if you want.
    orders = [2, 3, 4, 5, 8, 16, 32, 64, 128, 256]
    log_inv_delta = math.log(1.0 / delta)

    best_eps = float("inf")
    for alpha in orders:
        # Small-q approximation:
        # RDP ≈ (q^2 * alpha * steps) / (2*sigma^2)
        rdp = (q * q * alpha * steps) / (2.0 * sigma * sigma)
        eps = rdp + log_inv_delta / (alpha - 1.0)
        best_eps = min(best_eps, eps)

    return best_eps



def import_external(ext_src: Path):
    """
    Prepend external src path and import its modules.
    Returns a dict of needed external symbols.
    """
    sys.path.insert(0, str(ext_src))  # ensure external modules can be imported as top-level

    # external modules
    from options import args_parser
    from models import FairGNN
    from utils import (  # external utils (name-clash is fine because of sys.path order)
        average_weights, 
        aggregate_weights, 
        exp_details, load_pokec,
        accuracy, density, 
        # update_weights, 
        model_fair_metric
    )
    from terminal import Client  # Subgraph not used directly here

    return {
        "args_parser": args_parser,
        "FairGNN": FairGNN,
        "average_weights": average_weights,
        "aggregate_weights": aggregate_weights,
        "exp_details": exp_details,
        "load_pokec": load_pokec,
        "accuracy": accuracy,
        "density": density,
        # "update_weights": update_weights,
        "model_fair_metric": model_fair_metric,
        "Client": Client,
    }


def main():

    # ---- DP server defaults (override via args if you add flags later) ----
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.set_printoptions(profile="full")
    torch.autograd.set_detect_anomaly(True)

    repo_root = Path(__file__).resolve().parent
    ext_src = repo_root / "external" / "F2GNN" / "src"          # <- adjust if your submodule path differs
    ext_data = ext_src.parent / "data"                           # external/F2GNN/data

    ext = import_external(ext_src)

    args = ext["args_parser"]()
    ext["exp_details"](args)

    # sys.path.insert(0, "/home/jiaqiang/SecF2GNN/external/opacus")


    logger = SummaryWriter(log_dir=getattr(args, "log_dir", "runs"))
    if not hasattr(args, "mpc"): args.mpc = False          # enable/disable
    if not hasattr(args, "dp_server"): args.dp_server = False          # enable/disable
    if not hasattr(args, "dp_clip"): args.dp_clip = 1.0              # C
    if not hasattr(args, "dp_noise"): args.dp_noise = 0.5            # sigma
    # if not hasattr(args, "dp_noise"): args.dp_noise = 1.0            # sigma

    # delta = getattr(args, "dp_delta", None)
    # if not hasattr(args, "dp_delta") or args.dp_delta is None:
    #     args.dp_delta = 1.0 / (args.ego_number ** 2)
    # if delta is None:
    #     delta = 1.0 / (args.ego_number ** 2)
    if not hasattr(args, "dp_delta"): args.dp_delta = 1.0 / (args.ego_number ** 2)
    if not hasattr(args, "dp_report_every"): args.dp_report_every = 10


    # device
    if args.gpu != 'cpu':
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        torch.use_deterministic_algorithms(True)
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print("device: ", device)

    # set_seed(args.seed, args.gpu)
    set_seed(args)

    # --- dataset selection & paths (use external data folder explicitly) ---
    if args.dataset == 'pokec-z':
        dataset = 'region_job'
        sens_attr = "region"
        predict_attr = "I_am_working_in_field"
        path = str(ext_data / "pokec")
        test_idx = False
    elif args.dataset == 'pokec-n':
        dataset = 'region_job_2'
        sens_attr = "region"
        predict_attr = "I_am_working_in_field"
        path = str(ext_data / "pokec")
        test_idx = False
    elif args.dataset == 'NBA':
        dataset = 'nba'
        sens_attr = 'country'
        predict_attr = 'SALARY'
        path = str(ext_data / "NBA")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # --- load data via external loader ---
    graph, G, adj, features, labels, sens, num_nodes, edge_index = ext["load_pokec"](
        dataset, sens_attr, predict_attr, args, path=path
    )

    # --- split train/val/test the same way as original ---
    label_idx = np.where(labels >= 0)[0]
    sens_idx = set(np.where(sens >= 0)[0])
    idx_used = np.asarray(list(sens_idx & set(label_idx)))
    # np.random.seed(args.seed)
    random.seed(args.seed)
    random.shuffle(idx_used)

    idx_train = torch.LongTensor(idx_used[:int(0.5 * len(idx_used))])
    idx_val   = torch.LongTensor(idx_used[int(0.5 * len(idx_used)):int(0.75 * len(idx_used))])
    idx_test  = torch.LongTensor(idx_used[int(0.75 * len(idx_used)):])

    edge_index = torch.LongTensor(graph.edge_index)
    edges_train = subgraph(idx_train, edge_index)[0]
    edges_val   = subgraph(idx_val, edge_index)[0]
    edges_test  = subgraph(idx_test, edge_index)[0]

    idx_used_t = torch.LongTensor(idx_used)

    print('num_nodes_applied_graph: ', len(idx_used_t))
    print('num_nodes_test_graph: ', features[idx_test].shape[0])
    print('num_nodes_train_graph: ', features[idx_train].shape[0])
    print('num_edges_test_graph: ', edges_test.shape[1])
    print('sparsity_test_graph: ', ext["density"](features[idx_test].shape[0], m=edges_test.shape[1]))

    # --- build ego networks (clients) ---
    clients = []
    temp = []
    num_nodes_per_client = []
    intra_ratio_per_client = []
    inter_ratio_per_client = []

    print("Begin to generate ego networks.")
    random.seed(args.seed)
    for _ in range(args.ego_number):
        sub_nodes = torch.LongTensor([1, 1])
        while sub_nodes.shape[0] <= 2:
            # ego = int(np.random.choice(idx_train.cpu().numpy().tolist()))
            ego = random.choice(idx_train.tolist())
            sub_nodes, sub_edges, _, _ = k_hop_subgraph(node_idx=ego, num_hops=args.num_hops, edge_index=edge_index)

        clients.append(ext["Client"](features, labels, sens, sub_nodes, sub_edges, idx_train, idx_val, idx_test, args))
        num_nodes_per_client.append(float(sub_nodes.shape[0]))
        intra_ratio_per_client.append(clients[-1].intra_ratio)
        inter_ratio_per_client.append(clients[-1].inter_ratio)
        temp.extend(sub_nodes.cpu().numpy().tolist())

    from collections import Counter
    duplicates = {item: count for item, count in Counter(temp).items() if count > 1}
    print('num_nodes_overlap: ', len(duplicates))
    print('overlap partation: ', len(duplicates) / len(idx_used_t))

    # --- build models ---
    if args.model in ["GCN", "SGC"]:
        local_models = [ext["FairGNN"](nfeat=features.shape[1], args=args) for _ in range(args.ego_number)]
        global_model = ext["FairGNN"](nfeat=features.shape[1], args=args)
    else:
        raise SystemExit('Error: unrecognized model')

    # move tensors/models
    if args.gpu:
        global_model = global_model.to(device)
        features = torch.LongTensor(np.array(features)).to(device)
        labels   = torch.LongTensor(labels).to(device)
        idx_train = idx_train.to(device)
        idx_val   = idx_val.to(device)
        idx_test  = idx_test.to(device)
        sens      = torch.LongTensor(sens).to(device)
        G         = G.to(device)

    global_weights = global_model.state_dict()
    patience = args.patience

    for i in range(len(clients)):
        local_models[i] = local_models[i].to(device)

    train_loss = []
    acc_old = 0.0
    best_fair = 999.0
    best_result, cls_best_result = {}, {}
    counter = 0
    start = t()

    np.random.seed(args.seed)
    # accountant 
    # = RDPAccountant()
    # if not privacy_agent:
    #     privacy_agent = PrivAgent(len(egos), 'default_agent')
    for epoch in tqdm(range(args.epochs)):
        
        # eps = accountant.get_epsilon(args.dp_delta)
        # if eps > args.dp_eps:
        #     break

        # local_weights, local_losses, local_cls_losses, local_acc = [], [], [], []
        # local_parity, local_equality = [], []
        local_weights, local_updates = [], []
        local_losses, local_cls_losses, local_acc = [], [], []
        local_parity, local_equality = [], []

        round_global_weights = copy.deepcopy(global_weights)

        print(f'\n | Training Round : {epoch + 1} |\n')

        if args.ego_number <= 200:
            egos = np.random.choice(range(len(clients)), int(args.ego_number / 10), replace=False)
        elif args.ego_number == 1000:
            egos = np.random.choice(range(len(clients)), int(args.ego_number / 50), replace=False)
        else:
            egos = np.random.choice(range(len(clients)), int(args.ego_number / 20), replace=False)
        # from opacus.accountants.utils import get_noise_multiplier

        # sigma = get_noise_multiplier(
        #     target_epsilon=8.0,
        #     target_delta=1/(args.ego_number**2),
        #     sample_rate=egos / args.ego_number,
        #     steps=epoch,
        #     accountant="rdp",
        # )
        # args.dp_delta = sigma

        for i in egos:
            if args.gpu:
                clients[i].sub_features = clients[i].sub_features.to(device)
                clients[i].sub_labels   = clients[i].sub_labels.to(device)
                clients[i].sub_sens     = clients[i].sub_sens.to(device)
                clients[i].sub_G        = clients[i].sub_G.to(device)
                local_models[i]         = local_models[i].to(device)

            # Optional 'combine' update method using JS distance
            if args.update_method == 'combine':
                # y_global = global_model(clients[i].sub_G, clients[i].sub_features)[clients[i].idx_train_with_mask]
                # y_local  = local_models[i](clients[i].sub_G, clients[i].sub_features)[clients[i].idx_train_with_mask]

                ### copy and paste from source code ###
                y_global = global_model(clients[i].sub_G, clients[i].sub_features)
                y_global_ = y_global.squeeze()
                y_global_ = y_global[clients[i].idx_train_with_mask]
                y_local = local_models[i](clients[i].sub_G, clients[i].sub_features)
                y_local_ = y_local.squeeze()
                y_local_ = y_local[clients[i].idx_train_with_mask]
                # if len(y_global) >= 1 and len(y_local) >= 1:
                #     preds_global = (y_global > 0).type_as(labels)
                #     preds_local  = (y_local  > 0).type_as(labels)
                #     y_g = softmax([float(v) / args.tau_combine for v in preds_global])
                #     y_l = softmax([float(v) / args.tau_combine for v in preds_local])
                #     Beta = distance.jensenshannon(y_g, y_l)
                #     if not torch.is_tensor(Beta):
                #         Beta = torch.tensor(Beta, dtype=torch.float32, device=device)

                #     w_local = local_models[i].state_dict()
                #     w_server = {}
                #     for k, v in global_weights.items():
                #         if k in w_local:
                #             w_server[k] = v.to(device=w_local[k].device, dtype=w_local[k].dtype)
                #     new_local_weights = ext["update_weights"](w_local, w_server, Beta)

                #     # new_local_weights = ext["update_weights"](local_models[i].state_dict(), global_weights, Beta)
                #     local_models[i].load_state_dict(new_local_weights)
                # else:
                #     local_models[i].load_state_dict(copy.deepcopy(global_weights))
                if len(y_global_) >= 2 and len(y_local_) >= 2:

                    preds_y_global = (y_global_ > 0).type_as(labels)
                    temp_y_global = [float(i) / args.tau_combine for i in preds_y_global]
                    y_global = softmax(temp_y_global)

                    preds_y_local = (y_local_ > 0).type_as(labels)
                    temp_y_local = [float(i) / args.tau_combine for i in preds_y_local]
                    y_local = softmax(temp_y_local)

                    Beta = distance.jensenshannon(y_global, y_local)
                elif len(y_global_) == 1 and len(y_local_) == 1:
                    y_global_ = y_global_.detach().cpu().numpy()
                    y_global_[y_global_ > 0] = 1
                    temp_y_global = y_global_ / args.tau_combine
                    y_global = softmax(temp_y_global)

                    y_local_ = y_local_.detach().cpu().numpy()
                    y_local_[y_local_ > 0] = 1
                    temp_y_local = y_local_ / args.tau_combine
                    y_local = softmax(temp_y_local)

                    Beta = distance.jensenshannon(y_global, y_local)
                    Beta = torch.FloatTensor(Beta)
                    Beta = Beta.to(device)
                else:
                    pass
            if args.update_method == 'combine':
                if len(y_global_) >= 1 and len(y_local_) >= 1:
                    new_local_weights = update_weights(local_models[i].state_dict(), global_weights, Beta)
                    local_models[i].load_state_dict(new_local_weights)
                else:
                    local_models[i].load_state_dict(copy.deepcopy(global_weights))
            elif args.update_method == 'uniform':
                local_models[i].load_state_dict(copy.deepcopy(global_weights))

            # local training
            for _ in range(args.local_ep):
                local_models[i].train()
                local_models[i].optimize(clients[i].sub_G, clients[i].sub_features, clients[i].sub_labels,
                                         clients[i].sub_sens, clients[i].idx_train_with_mask)

            # metrics from local model
            # local_weights.append(copy.deepcopy(local_models[i].state_dict()))
            final_local_sd = copy.deepcopy(local_models[i].state_dict())
            local_weights.append(final_local_sd)  # optional, for debugging/logging
            local_updates.append(sd_to_update(final_local_sd, round_global_weights))
            ###################

            local_losses.append(local_models[i].G_loss.item())
            local_cls_losses.append(local_models[i].cls_loss.item())
            local_parity.append(local_models[i].parity)
            local_equality.append(local_models[i].equality)
            local_acc.append(local_models[i].acc)

        # reductions
        reduction_functions = {'median': torch.median, 'mean': torch.mean, 'min': torch.min}
        if args.local_fairness_measure in reduction_functions:
            reduction_func = reduction_functions[args.local_fairness_measure]
            input_local_parity = torch.cat([
                (torch.tensor(t, dtype=torch.float32) if isinstance(t, (int, float)) else t).unsqueeze(0).to(device)
                for t in local_parity
            ])
            input_local_equality = torch.cat([
                (torch.tensor(t, dtype=torch.float32) if isinstance(t, (int, float)) else t).unsqueeze(0).to(device)
                for t in local_equality
            ])
            measure_local_parity = reduction_func(input_local_parity)
            measure_local_equality = reduction_func(input_local_equality)

        # log local stats
        logger.add_scalar("train_loss", local_losses[-1], epoch)
        logger.add_scalar("local_cls_loss", local_cls_losses[-1], epoch)
        logger.add_scalar("local_train_parity_median", measure_local_parity.item(), epoch)
        logger.add_scalar("local_train_equality_median", measure_local_equality.item(), epoch)

        # aggregate
        if args.aggregate_method == 'avg':
            # print("++++++++++++++++++")
            # global_weights = ext["average_weights"](local_weights)
            # if args.dp_server:
            #     global_weights, dp_norms, dp_scales = dp_server_aggregate(
            #         local_state_dicts=local_weights,
            #         # global_state_dict=global_weights,
            #         clip_norm=args.dp_clip,
            #         noise_multiplier=args.dp_noise,
            #         weights=None
            #     )
            # else:
                global_weights = ext["average_weights"](local_weights)

        else:
            if args.mpc in ("mpc", "exp"):
                if args.aggregate_method == '1-|inter-intra|':
                    difference_weight_vector  = [abs(inter_ratio_per_client[i] - intra_ratio_per_client[i]) for i in egos]
                    difference_weight_vector2 = [(1 - i) for i in difference_weight_vector]
                    temp_weights_vector = softmax(difference_weight_vector2)
                    temp_weights_vector2 = [(1 / (local_models[i].equality + local_models[i].parity + 1E-5)) for i in egos]

                    if args.mpc == "mpc":
                        temp_weights_vector2 = torch.tensor(temp_weights_vector2, device=device)
                        save_local_params(
                            "results/mpc_inputs",
                            "client0_round1",
                            args.lambda1,
                            args.tau,
                            difference_weight_vector2,
                            temp_weights_vector2,
                            local_weights
                        )
                        global_weights = f2gnn_agg()
                        global_weights = {
                            k: v.to(device) if torch.is_tensor(v) else v
                            for k, v in global_weights.items()
                        }
                    else:
                        # difference_weight_vector3 = torch.exp(difference_weight_vector2)
                        # temp_weights_vector3 = torch.exp(temp_weights_vector2)
                        diff2_t = torch.as_tensor(difference_weight_vector2, device=device, dtype=torch.float32)
                        temp2_t = torch.as_tensor(temp_weights_vector2, device=device, dtype=torch.float32)

                        diff3_t = torch.exp(diff2_t)
                        temp3_t = torch.exp(temp2_t)

                        if args.mpc == "exp":
                            save_local_params(
                                "results/mpc_inputs",
                                "client0_round1",
                                args.lambda1,
                                args.tau,
                                diff3_t,
                                temp3_t,
                                local_weights
                            )
                            global_weights = f2gnn_agg("exp_f2gnn_agg")
                        else:
                            raise ValueError(f"mpc must be one of ['mpc','exp','none'], got: {args.mpc}")
                    # temp_weights_vector2 = torch.softmax(temp_weights_vector2, dim=-1)
                    # temp_weights_vector2 = [torch.exp(i) for i in temp_weights_vector2]

                    # normalized_weights_vector = [(args.lambda1 * w1 + w2) / args.tau
                    #                             for w1, w2 in zip(temp_weights_vector, temp_weights_vector2)]
                    # normalized_weights_vector = torch.tensor(normalized_weights_vector, device=device)
                    # normalized_weights_vector = torch.softmax(normalized_weights_vector, dim=-1)
                else:
                    raise ValueError("args.aggregate_method is wrong!")
                # global_weights = ext["aggregate_weights"](local_weights, normalized_weights_vector)
            else:    
                if args.aggregate_method == '1-|inter-intra|':
                    difference_weight_vector  = [abs(inter_ratio_per_client[i] - intra_ratio_per_client[i]) for i in egos]
                    difference_weight_vector2 = [(1 - i) for i in difference_weight_vector]
                    temp_weights_vector = softmax(difference_weight_vector2)

                    temp_weights_vector2 = [(1 / (local_models[i].equality + local_models[i].parity + 1E-5)) for i in egos]
                    temp_weights_vector2 = torch.tensor(temp_weights_vector2, device=device)
                    temp_weights_vector2 = torch.softmax(temp_weights_vector2, dim=-1)
                    temp_weights_vector2 = [torch.exp(i) for i in temp_weights_vector2]

                    normalized_weights_vector = [(args.lambda1 * w1 + w2) / args.tau
                                                for w1, w2 in zip(temp_weights_vector, temp_weights_vector2)]
                    normalized_weights_vector = torch.tensor(normalized_weights_vector, device=device)
                    normalized_weights_vector = torch.softmax(normalized_weights_vector, dim=-1)
                else:
                    raise ValueError("args.aggregate_method is wrong!")
                # global_weights = ext["aggregate_weights"](local_weights, normalized_weights_vector)
                if args.dp_server:
                    args.dp_clip, norms = get_clip_norm(local_updates)
                    global_weights = dp_server_aggregate(
                        local_updates,
                        global_weights,
                        args.dp_clip,
                        # norms,
                        args.dp_noise,
                        normalized_weights_vector
                    )
                    # q = 1 / len(egos)
                    # accountant.step(args.dp_noise, q)
                else:
                # global_weights = aggregate_weights(local_weights, normalized_weights_vector)
                    global_updates = aggregate_updates(local_updates, normalized_weights_vector)
                    global_weights = apply_updates(round_global_weights, global_updates)

        # if args.dp_server:
        #     logger.add_scalar("dp/clip_norm", args.dp_clip, epoch)
        #     logger.add_scalar("dp/noise_multiplier", args.dp_noise, epoch)
        #     logger.add_scalar("dp/mean_client_update_norm", float(np.mean(dp_norms)), epoch)
        #     logger.add_scalar("dp/mean_clip_scale", float(np.mean(dp_scales)), epoch)
        if args.dp_server and args.dp_noise > 0 and ((epoch + 1) % args.dp_report_every == 0):
            q = len(egos) / float(args.ego_number)
            delta = getattr(args, "dp_delta", 1.0 / (args.ego_number ** 2))
            delta = args.dp_delta
            eps = rdp_epsilon_approx(q=q, sigma=args.dp_noise, steps=epoch + 1, delta=delta)
            print(f"[DP approx] round={epoch+1} q={q:.4f} sigma={args.dp_noise} delta={delta:g} => eps≈{eps:.3f}")


        global_model.load_state_dict(global_weights)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        global_model = global_model.to(device)


        # train/val metrics
        loss_avg = sum(local_losses) / max(len(local_losses), 1)
        train_loss.append(loss_avg)

        train_acc_avg = sum(local_acc) / max(len(local_acc), 1)

        global_model.eval()
        with torch.no_grad():
            output = global_model(G, features)

        acc_val = ext["accuracy"](output[idx_val], labels[idx_val])
        roc_val = roc_auc_score(np.float64(labels[idx_val].detach().cpu().numpy()),
                                np.float64(output[idx_val].detach().cpu().numpy()))
        parity_val, equality_val = fair_metric(output, idx_val, labels, sens)

        acc_test = ext["accuracy"](output[idx_test], labels[idx_test])
        roc_test = roc_auc_score(np.float64(labels[idx_test].detach().cpu().numpy()),
                                 np.float64(output[idx_test].detach().cpu().numpy()))
        parity, equality = fair_metric(output, idx_test, labels, sens)

        # local test metrics over clients
        list_test_parity, list_test_equality, list_test_acc, list_test_roc = [], [], [], []
        for i in range(args.ego_number):
            client = clients[i]
            with torch.no_grad():
                output = global_model(G, features)
                acc_llocal = ext["accuracy"](output[client.idx_test_with_mask], labels[client.idx_test_with_mask])
                list_test_acc.append(acc_llocal)
                try:
                    roc_llocal = roc_auc_score(
                        np.float64(labels[client.idx_test_with_mask].detach().cpu().numpy()),
                        np.float64(output[client.idx_test_with_mask].detach().cpu().numpy()))
                    list_test_roc.append(roc_llocal)
                except (ValueError, TypeError):
                    pass

                pred_local = (output[client.idx_test_with_mask].squeeze() > 0).type_as(labels)
                parity_llocal, equality_llocal = ext["model_fair_metric"](
                    pred_local, labels[client.idx_test_with_mask], sens[client.idx_test_with_mask]
                )
                list_test_parity.append(parity_llocal)
                list_test_equality.append(equality_llocal)

        reduction_func = reduction_functions[args.local_fairness_measure]
        measure_local_test_equality = reduction_func(
            torch.cat([torch.tensor(t, dtype=torch.float32, device=device).unsqueeze(0) for t in list_test_equality]))
        measure_local_test_parity = reduction_func(
            torch.cat([torch.tensor(t, dtype=torch.float32, device=device).unsqueeze(0) for t in list_test_parity]))
        measure_local_test_acc = reduction_func(
            torch.cat([torch.tensor(t, dtype=torch.float32, device=device).unsqueeze(0) for t in list_test_acc]))
        if len(list_test_roc) > 0:
            measure_local_test_roc = reduction_func(
                torch.cat([torch.tensor(t, dtype=torch.float32, device=device).unsqueeze(0) for t in list_test_roc]))
        else:
            measure_local_test_roc = torch.tensor(float('nan'), device=device)

        # track bests (same logic as original)
        if float(acc_old) < float(acc_val):
            cls_best_result = {
                'acc': acc_val.item(), 'roc': roc_val, 'parity': parity_val, 'equality': equality_val,
                'acc_test': acc_test.item(), 'roc_test': roc_test, 'parity_test': parity, 'equality_test': equality,
                'measure_local_test_parity': measure_local_test_parity.item(),
                'measure_local_test_equality': measure_local_test_equality.item(),
                'measure_local_test_acc': measure_local_test_acc.item(),
                'measure_local_test_roc': float(measure_local_test_roc.item()) if torch.isfinite(measure_local_test_roc) else float('nan'),
                'epoch': epoch + 1
            }
            best_result = dict(cls_best_result)
            acc_old = acc_val

        if acc_val > args.acc:
            if best_fair > parity_val + equality_val:
                best_result = {
                    'acc': acc_val.item(), 'roc': roc_val, 'parity': parity_val, 'equality': equality_val,
                    'acc_test': acc_test.item(), 'roc_test': roc_test, 'parity_test': parity, 'equality_test': equality,
                    'measure_local_test_parity': measure_local_test_parity.item(),
                    'measure_local_test_equality': measure_local_test_equality.item(),
                    'measure_local_test_acc': measure_local_test_acc.item(),
                    'measure_local_test_roc': float(measure_local_test_roc.item()) if torch.isfinite(measure_local_test_roc) else float('nan'),
                    'epoch': epoch + 1
                }
                best_fair = parity_val + equality_val

        # simple early stopping mimic
        counter = counter + 1 if abs(float(acc_val) - float(acc_old)) <= 1e-5 else 0
        if counter > patience and epoch >= 250:
            break

        # logs
        logger.add_scalar("val_acc", acc_val.item(), epoch + 1)
        logger.add_scalar("val_roc", roc_val, epoch + 1)
        logger.add_scalar("val_parity", parity_val, epoch + 1)
        logger.add_scalar("val_equality", equality_val, epoch + 1)
        logger.add_scalar("test_acc", acc_test.item(), epoch + 1)
        logger.add_scalar("test_roc", roc_test, epoch + 1)
        logger.add_scalar("test_parity", parity, epoch + 1)
        logger.add_scalar("test_equality", equality, epoch + 1)
        logger.add_scalar("measure_local_test_parity", measure_local_test_parity, epoch + 1)
        logger.add_scalar("measure_local_test_equality", measure_local_test_equality, epoch + 1)

        print("\n=================================")
        print("counter: ", counter)
        print('Epoch: {:04d}\n'.format(epoch + 1),
              '*acc_train*: {:.4f}\n'.format(train_acc_avg),
              '*acc_val*: {:.4f}'.format(acc_val.item()),
              "auc_val: {:.4f}".format(roc_val),
              "parity_val: {:.4f}".format(parity_val),
              "equality: {:.4f}".format(equality_val),
              "measure_local_test_parity: {:.4f}".format(measure_local_test_parity),
              "measure_local_test_equality: {:.4f}".format(measure_local_test_equality))
        print("Test:\n",
              "*acc_test*: {:.4f}".format(acc_test.item()),
              "auc: {:.4f}".format(roc_test),
              "parity: {:.4f}".format(parity),
              "equality: {:.4f}".format(equality),
              "measure_local_test_parity: {:.4f}".format(measure_local_test_parity),
              "measure_local_test_equality: {:.4f}".format(measure_local_test_equality))

    now = t()
    logger.close()

    # write exp log like the original
    exp_log = repo_root / "exp_log.txt"
    with open(exp_log, mode='a+') as file:
        file.write("\n=================================================\n")
        file.write("Parameters: \n" + str(args))
        if best_result:
            file.write(
                "\nBest Results:\n"
                f"Epoch: {best_result['epoch']}\n"
                f"Validation accuracy: {best_result['acc']} Validation auc: {best_result['roc']} "
                f"Validation parity: {best_result['parity']} Validation equality: {best_result['equality']}\n"
                f"Test accuracy: {best_result['acc_test']} Test auc: {best_result['roc_test']} "
                f"Test parity: {best_result['parity_test']} Test equality: {best_result['equality_test']} "
                f"Test measure_local_parity: {best_result['measure_local_test_parity']} "
                f"Test measure_local_equality: {best_result['measure_local_test_equality']}\n"
                f"Time: {(now - start) / 60:.4f} mins\n"
            )
        elif 'acc' in locals().get('cls_best_result', {}):
            c = cls_best_result
            file.write(
                "\nThe Best Classifying Results:\n"
                f"Epoch: {c['epoch']}\n"
                f"Validation accuracy: {c['acc']} Validation auc: {c['roc']} "
                f"Validation parity: {c['parity']} Validation equality: {c['equality']}\n"
                f"Test accuracy: {c['acc_test']} Test auc: {c['roc_test']} "
                f"Test parity: {c['parity_test']} Test equality: {c['equality_test']} "
                f"Test measure_local_parity: {c['measure_local_test_parity']} "
                f"Test measure_local_equality: {c['measure_local_test_equality']}\n"
            )


if __name__ == "__main__":
    main()

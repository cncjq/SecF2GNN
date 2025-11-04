# utils.py (YOUR helper utilities, safe to live at repo root)
import random
from typing import List, Tuple

import numpy as np
import torch


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

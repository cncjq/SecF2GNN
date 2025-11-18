# server.py
import io
import os
import base64
from typing import Dict, List

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

# ----- import external repo modules -----
import sys, importlib.util
ROOT = Path(__file__).resolve().parent
EXT_SRC = ROOT / "external" / "F2GNN" / "src"
sys.path.insert(0, str(EXT_SRC))
spec = importlib.util.spec_from_file_location("utils", str(EXT_SRC / "utils.py"))
ext_utils = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext_utils)
from models import FairGNN

app = FastAPI()

# Global state on the server
global_model = None
model_layout_keys: List[str] = []   # preserve parameter key order for consistent merging
round_idx = 0
updates_buffer: List[Dict] = []
expected_clients = int(os.getenv("EXPECTED_CLIENTS", "1"))

class Update(BaseModel):
    cid: int
    round: int
    state_dict_b64: str  # base64 of torch.save(state_dict)

def _init_model(nfeat: int, args_ns):
    global global_model, model_layout_keys
    global_model = FairGNN(nfeat=nfeat, args=args_ns)
    model_layout_keys = list(global_model.state_dict().keys())

def _deserialize_sd(b64: str) -> Dict[str, torch.Tensor]:
    raw = base64.b64decode(b64.encode("utf-8"))
    buf = io.BytesIO(raw)
    return torch.load(buf, map_location="cpu")

@app.get("/ready")
def ready():
    return {"ok": True, "round": round_idx, "expected_clients": expected_clients}

@app.post("/init")
def init(nfeat: int):
    # minimal args shim for FairGNN; client sends a subset via headers/env if required
    from types import SimpleNamespace
    args = SimpleNamespace(num_hidden=64, model="GCN", dropout=0.5, penalty=0.001)
    _init_model(nfeat, args)
    return {"ok": True}

@app.get("/global")
def get_global():
    # send current global weights
    sd = global_model.state_dict()
    buf = io.BytesIO(); torch.save(sd, buf)
    return {"round": round_idx, "state_dict_b64": base64.b64encode(buf.getvalue()).decode("utf-8")}

@app.post("/update")
def post_update(upd: Update):
    global updates_buffer, global_model, round_idx
    # collect this update
    updates_buffer.append({"cid": upd.cid, "round": upd.round, "sd": _deserialize_sd(upd.state_dict_b64)})
    if len(updates_buffer) < expected_clients:
        return {"ok": True, "awaiting": expected_clients - len(updates_buffer)}

    # aggregate when all expected arrived
    updates_buffer.sort(key=lambda x: x["cid"])
    new_sd = {}
    for k in model_layout_keys:
        tensors = [u["sd"][k].detach().cpu().float() for u in updates_buffer]
        new_sd[k] = torch.stack(tensors, dim=0).mean(0).to(global_model.state_dict()[k].dtype)

    global_model.load_state_dict(new_sd)
    round_idx += 1
    updates_buffer.clear()
    return {"ok": True, "aggregated_round": round_idx}

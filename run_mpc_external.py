import re
import subprocess
from pathlib import Path
from typing import Optional, List


# --- Config ---
import os

MPSPDZ_PATH = os.environ.get(
    "MP_SPDZ_PATH",
    "/home/jiaqiang/SecF2GNN/external/MP-SPDZ"
)
# PROGRAM_NAME = "test"           # Programs/Source/test.mpc
PROGRAM_NAME = 'f2gnn_agg'
PORT_BASE = "15000"             # arbitrary, avoid clashes
NUM_PARTIES = 2                 # semi2k = 2 parties
BINARY = "semi2k-party.x"
FLOAT_PATTERN = re.compile(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?")

def _ensure_binary():
    exe = Path(MPSPDZ_PATH) / BINARY
    if not exe.exists():
        raise FileNotFoundError(
            f"{BINARY} not found in {MPSPDZ_PATH}. Build it:\n"
            f"  cd {MPSPDZ_PATH} && make -j8 {BINARY}"
        )

def _compile_ring64():
    """Compile program for 64-bit ring (semi2k requirement)."""
    cmd = ["./compile.py", "-R", "64", PROGRAM_NAME]
    subprocess.run(cmd, cwd=MPSPDZ_PATH, check=True)

def _write_inputs_txt(
    lambda1, tau,
    data_bias_weights, model_fairness_weights, local_weights,
    case_name="case1", f=16
):
    """
    Save MPC inputs in a plain-text format compatible with sfix.input_tensor_from().
    Each value appears on its own line. Floating-point values are scaled by 2^f.

    Args:
        lambda1: float scalar
        tau: int scalar
        data_bias_weights, model_fairness_weights, local_weights: lists of floats
        case_name: identifier for naming the file (e.g., "case1", "client_0")
        f: number of fractional bits for sfix scaling (match -F in MP-SPDZ)
    """
    pd = Path(MPSPDZ_PATH) / "Player-Data"
    pd.mkdir(exist_ok=True)

    # e.g. Player-Data/inputs_case1.txt
    path_txt = pd / f"inputs_{case_name}.txt"

    scale = 1 << f
    lines = []

    # --- Scalars ---
    lines.append(str(int(round(lambda1 * scale))))   # scaled sfix
    lines.append(str(int(tau)))                      # integer tau

    # --- Vectors ---
    for v in data_bias_weights:
        lines.append(str(int(round(v * scale))))
    for v in model_fairness_weights:
        lines.append(str(int(round(v * scale))))
    for v in local_weights:
        lines.append(str(int(round(v * scale))))

    # --- Write ---
    path_txt.write_text("\n".join(lines) + "\n")
    print(f"✅ Saved {len(lines)} values to {path_txt} (precision F={f})")

    return path_txt



def _run_parties_and_collect():
    procs, outs, errs = [], [], []
    for p in range(NUM_PARTIES):
        cmd = [
            f"./{BINARY}",
            "-N", str(NUM_PARTIES),
            "-p", str(p),
            "-pn", PORT_BASE,
            "-OF", ".",             # stdout on all parties
            PROGRAM_NAME,
        ]
        procs.append(subprocess.Popen(
            cmd, cwd=MPSPDZ_PATH, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ))
    for i, proc in enumerate(procs):
        out, err = proc.communicate()
        outs.append(out or "")
        errs.append(err or "")

    return outs, errs

def run_mpspdz(
        lambda1: float, 
        tau: int, data_bias_weights: list, 
        model_fairness_weights: list, 
        local_weights: list,
        *,
        fractional_bits: int = 16) -> Optional[List[float]]:
    _ensure_binary()
    _compile_ring64()
    _write_inputs_txt(lambda1, tau, data_bias_weights, model_fairness_weights, local_weights)
    outs, errs = _run_parties_and_collect()
    for i, err in enumerate(errs):
        if err.strip():
            print(f"[P{i}] {err}")
    numbers: List[float] = []
    for out in outs:
        for token in FLOAT_PATTERN.findall(out):
            try:
                numbers.append(float(token))
            except ValueError:
                continue
    return numbers

def run_secure_aggregation(expected_outputs: Optional[int] = None,
                           fractional_bits: int = 16) -> List[float]:
    """
    Execute the compiled MPC program and return all numeric outputs as floats.

    Args:
        expected_outputs: optional safety check for the number of floats required.
    fractional_bits: compile precision for sfix inputs.
    """
    _ensure_binary()
    _compile_ring64()
    outs, errs = _run_parties_and_collect()

    for i, err in enumerate(errs):
        if err.strip():
            print(f"[P{i}] {err}")

    numbers: List[float] = []
    for out in outs:
        for token in FLOAT_PATTERN.findall(out):
            try:
                numbers.append(float(token))
            except ValueError:
                continue
    if expected_outputs is not None and len(numbers) < expected_outputs:
        raise ValueError(f"Expected at least {expected_outputs} MPC outputs, got {len(numbers)}")
    return numbers if expected_outputs is None else numbers[:expected_outputs]

if __name__ == "__main__":
    lambda1 = 0.5
    tau = 4
    data_bias_weights = [0.10256410256410264, 0.08091024020227566, 0.08091024020227566]
    model_fairness_weights = [0.5816631317138672, 0.20614328980445862, 0.20614328980445862]
    local_weights = [1.0, 1.5, 2.0]

    _write_inputs_txt(
        lambda1, tau,
        data_bias_weights, model_fairness_weights, local_weights,
        case_name="client0_round1"
    )

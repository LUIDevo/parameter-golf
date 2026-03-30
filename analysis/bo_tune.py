"""
Bayesian Optimisation for hyperparameter tuning with an extremely small compute budget.

Uses Optuna's TPE (Tree-structured Parzen Estimator) sampler to search the
hyperparameter space efficiently.  Each trial runs a short proxy training
(few iterations) and reports the final val_bpb as the objective.

Setup (run once on the H100 node before starting):
    pip install optuna

Usage (from the repo root, i.e. parent of analysis/):
    python analysis/bo_tune.py                              # 15 trials x 200 iters
    python analysis/bo_tune.py --n-trials 10 --iters 100    # ultra-cheap
    python analysis/bo_tune.py --n-trials 25 --iters 300    # slightly larger budget
    python analysis/bo_tune.py --db sqlite:///bo.db          # persist to resume later

All output goes to stdout so you can copy the winning config directly from
your terminal.  The final block is a ready-to-paste torchrun command.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys

import optuna


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_trial_training(hparams: dict, iters: int, trial_id: int) -> float | None:
    """Launch a single short training run and return the final val_bpb."""
    env = os.environ.copy()
    env.update({k: str(v) for k, v in hparams.items()})
    env["ITERATIONS"] = str(iters)
    env["VAL_LOSS_EVERY"] = str(iters)       # only validate at the end
    env["TRAIN_LOG_EVERY"] = str(max(iters // 5, 1))
    env["RUN_ID"] = f"bo_trial_{trial_id:03d}"

    cmd = ["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"]
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        print(f"  [trial {trial_id}] TIMEOUT")
        return None

    output = result.stdout + "\n" + result.stderr

    # Parse the final int8 roundtrip val_bpb (most representative of submission quality)
    match = re.search(r"final_int8_zlib_roundtrip_exact\s+val_loss:\S+\s+val_bpb:(\S+)", output)
    if match:
        return float(match.group(1))

    # Fallback: parse last val_bpb line
    matches = re.findall(r"val_bpb:(\S+)", output)
    if matches:
        return float(matches[-1])

    print(f"  [trial {trial_id}] FAILED -- could not parse val_bpb")
    if result.returncode != 0:
        for line in output.strip().splitlines()[-10:]:
            print(f"    {line}")
    return None


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

# Divisibility-safe architecture choices.
# Each entry is (model_dim, num_heads, num_kv_heads) -- all pre-validated.
ARCH_CONFIGS = [
    # dim, heads, kv_heads
    (384, 6, 3),
    (384, 8, 4),
    (448, 8, 4),
    (512, 8, 4),
    (512, 8, 8),
    (512, 16, 4),
    (576, 8, 4),
    (576, 12, 4),
    (640, 8, 4),
    (640, 10, 5),
    (768, 8, 4),
    (768, 12, 4),
    (768, 16, 8),
]


def define_search_space(trial: optuna.Trial) -> dict:
    """Define the hyperparameter search space for one trial."""

    # Architecture (categorical to guarantee divisibility)
    arch_idx = trial.suggest_categorical("arch_idx", list(range(len(ARCH_CONFIGS))))
    model_dim, num_heads, num_kv_heads = ARCH_CONFIGS[arch_idx]

    num_layers = trial.suggest_int("num_layers", 6, 16)
    mlp_mult = trial.suggest_categorical("mlp_mult", [2, 3, 4])

    # Learning rates (log-uniform)
    matrix_lr = trial.suggest_float("matrix_lr", 0.01, 0.12, log=True)
    scalar_lr = trial.suggest_float("scalar_lr", 0.01, 0.12, log=True)
    tied_embed_lr = trial.suggest_float("tied_embed_lr", 0.01, 0.15, log=True)

    # Muon optimizer
    muon_momentum = trial.suggest_float("muon_momentum", 0.90, 0.98)

    # Other
    qk_gain_init = trial.suggest_float("qk_gain_init", 0.5, 3.0)
    warmdown_iters = trial.suggest_int("warmdown_iters", 200, 2000, step=200)

    return {
        "MODEL_DIM": model_dim,
        "NUM_HEADS": num_heads,
        "NUM_KV_HEADS": num_kv_heads,
        "NUM_LAYERS": num_layers,
        "MLP_MULT": mlp_mult,
        "MATRIX_LR": matrix_lr,
        "SCALAR_LR": scalar_lr,
        "TIED_EMBED_LR": tied_embed_lr,
        "MUON_MOMENTUM": muon_momentum,
        "QK_GAIN_INIT": qk_gain_init,
        "WARMDOWN_ITERS": warmdown_iters,
    }


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial, iters: int) -> float:
    hparams = define_search_space(trial)

    dim = hparams["MODEL_DIM"]
    layers = hparams["NUM_LAYERS"]
    heads = hparams["NUM_HEADS"]
    mlp = hparams["MLP_MULT"]
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: dim={dim} layers={layers} heads={heads} mlp_mult={mlp}")
    print(f"  matrix_lr={hparams['MATRIX_LR']:.4f} scalar_lr={hparams['SCALAR_LR']:.4f} "
          f"tied_embed_lr={hparams['TIED_EMBED_LR']:.4f}")
    print(f"  muon_momentum={hparams['MUON_MOMENTUM']:.3f} qk_gain={hparams['QK_GAIN_INIT']:.2f}")
    print(f"{'='*60}", flush=True)

    val_bpb = run_trial_training(hparams, iters, trial.number)

    if val_bpb is None or not math.isfinite(val_bpb):
        print(f"  -> PRUNED (no valid val_bpb)", flush=True)
        raise optuna.TrialPruned()

    print(f"  -> val_bpb = {val_bpb:.6f}", flush=True)
    return val_bpb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter search for train_gpt.py",
        epilog="Prerequisite: pip install optuna",
    )
    parser.add_argument("--n-trials", type=int, default=15,
                        help="Number of BO trials (default: 15)")
    parser.add_argument("--iters", type=int, default=200,
                        help="Training iterations per trial (default: 200)")
    parser.add_argument("--study-name", type=str, default="parameter-golf-bo",
                        help="Optuna study name")
    parser.add_argument("--db", type=str, default=None,
                        help="Optuna storage URL (e.g. sqlite:///bo.db). "
                             "Default: in-memory (no persistence)")
    cli = parser.parse_args()

    # Silence Optuna's internal logs -- we print our own summaries
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = cli.db
    study = optuna.create_study(
        study_name=cli.study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    print(f"Starting BO search: {cli.n_trials} trials x {cli.iters} iters each")
    print(f"Search space: {len(ARCH_CONFIGS)} arch configs, layers 6-16, mlp_mult [2,3,4]")
    print(f"Working directory: {REPO_ROOT}", flush=True)

    study.optimize(
        lambda trial: objective(trial, cli.iters),
        n_trials=cli.n_trials,
    )

    # -----------------------------------------------------------------------
    # Results summary -- all to stdout for easy copy-paste
    # -----------------------------------------------------------------------

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\n{'='*60}")
    print(f"SEARCH COMPLETE: {len(completed)}/{cli.n_trials} trials succeeded")
    print(f"{'='*60}")

    if not completed:
        print("No trials completed successfully. Check errors above.")
        return

    # Leaderboard
    ranked = sorted(completed, key=lambda t: t.value)
    print(f"\nTop 5 trials:")
    for i, t in enumerate(ranked[:5]):
        dim, heads, kv = ARCH_CONFIGS[t.params["arch_idx"]]
        print(f"  #{i+1}  val_bpb={t.value:.6f}  "
              f"dim={dim} layers={t.params['num_layers']} heads={heads} "
              f"mlp={t.params['mlp_mult']} matrix_lr={t.params['matrix_lr']:.4f}")

    # Best config as copy-paste command
    best = study.best_trial
    p = best.params
    dim, heads, kv = ARCH_CONFIGS[p["arch_idx"]]

    print(f"\n{'='*60}")
    print(f"BEST: trial {best.number}, val_bpb={best.value:.6f}")
    print(f"{'='*60}")
    print()
    print("# --- Copy-paste this for your full training run ---")
    parts = [
        f"RUN_ID=bo_best",
        f"MODEL_DIM={dim}",
        f"NUM_HEADS={heads}",
        f"NUM_KV_HEADS={kv}",
    ]
    for k, v in p.items():
        if k == "arch_idx":
            continue
        parts.append(f"{k.upper()}={v}")
    parts.append("torchrun --standalone --nproc_per_node=1 train_gpt.py")
    print(" \\\n  ".join(parts))
    print()


if __name__ == "__main__":
    main()

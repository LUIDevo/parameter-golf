#!/usr/bin/env python3
"""CLI tool for analyzing parameter-golf training runs.

Usage:
    python analysis/analyze.py summary                     # Table of all runs
    python analysis/analyze.py overfit                     # Overfitting check (all runs)
    python analysis/analyze.py overfit run_id              # Overfitting check (single run)
    python analysis/analyze.py compare run1 run2           # Diff two runs
    python analysis/analyze.py plot loss                   # Loss curves
    python analysis/analyze.py plot bpb                    # BPB bar chart
    python analysis/analyze.py plot param matrix_lr        # Hyperparam vs BPB
    python analysis/analyze.py plot memory                 # VRAM + submission size
    python analysis/analyze.py detail run_id               # Full detail for one run
    python analysis/analyze.py report                      # HTML report (opens in browser)
    python analysis/analyze.py pull user@host:/path/logs/  # Pull logs from RunPod
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

# Allow running from repo root or analysis/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parse_logs import RunResult, parse_all_logs

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
OUTPUT_DIR = Path(__file__).resolve().parent / "plots"


def fmt_bpb(val: float | None) -> str:
    return f"{val:.4f}" if val is not None else "—"


def fmt_int(val: int | None) -> str:
    return f"{val:,}" if val is not None else "—"


def fmt_mb(val: int | None) -> str:
    return f"{val / 1024 / 1024:.1f}MB" if val is not None else "—"


def fmt_time(ms: float | None) -> str:
    if ms is None:
        return "—"
    if ms < 60_000:
        return f"{ms / 1000:.1f}s"
    return f"{ms / 60_000:.1f}min"


def _overfit_diagnosis(run: RunResult) -> dict | None:
    """Compute overfitting metrics for a single run.

    Returns a dict with:
      - train_losses / val_losses: aligned lists at validation steps
      - gap_start / gap_end: train-val gap at first and last validation point
      - gap_trend: positive means the gap is widening (overfitting signal)
      - val_turning_step: step where val loss first increases (None if always decreasing)
      - verdict: "ok", "mild", or "overfitting"
    """
    # Build aligned (step -> train_loss, val_loss) pairs.
    # train_loss is logged more often, so interpolate: use the most recent
    # train_loss logged at or before each validation step.
    train_by_step = {}
    for s in run.steps:
        if s.train_loss is not None:
            train_by_step[s.step] = s.train_loss
    val_points = [(s.step, s.val_loss) for s in run.steps if s.val_loss is not None]

    if len(val_points) < 2 or not train_by_step:
        return None

    sorted_train_steps = sorted(train_by_step.keys())

    aligned = []  # (step, train_loss, val_loss)
    for v_step, v_loss in val_points:
        # Find most recent train_loss at or before this val step
        tl = None
        for ts in reversed(sorted_train_steps):
            if ts <= v_step:
                tl = train_by_step[ts]
                break
        if tl is None:
            # Use the earliest train_loss if val is logged before any train
            tl = train_by_step[sorted_train_steps[0]]
        aligned.append((v_step, tl, v_loss))

    steps = [a[0] for a in aligned]
    train_losses = [a[1] for a in aligned]
    val_losses = [a[2] for a in aligned]
    gaps = [v - t for t, v in zip(train_losses, val_losses)]

    gap_start = gaps[0]
    gap_end = gaps[-1]
    gap_trend = gap_end - gap_start

    # Detect where val loss first turns upward
    val_turning_step = None
    min_val = val_losses[0]
    for i in range(1, len(val_losses)):
        if val_losses[i] < min_val:
            min_val = val_losses[i]
        elif val_losses[i] > min_val and val_turning_step is None:
            val_turning_step = steps[i]

    # Verdict
    if val_turning_step is not None and gap_trend > 0.05:
        verdict = "overfitting"
    elif gap_trend > 0.02 or val_turning_step is not None:
        verdict = "mild"
    else:
        verdict = "ok"

    return {
        "steps": steps,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "gaps": gaps,
        "gap_start": gap_start,
        "gap_end": gap_end,
        "gap_trend": gap_trend,
        "val_turning_step": val_turning_step,
        "verdict": verdict,
    }


def cmd_overfit(runs: list[RunResult], run_id: str | None) -> None:
    if run_id:
        targets = [r for r in runs if r.run_id == run_id]
        if not targets:
            print(f"Run '{run_id}' not found")
            return
    else:
        targets = runs

    results = []
    for run in targets:
        diag = _overfit_diagnosis(run)
        if diag is None:
            print(f"{run.run_id}: not enough data (need >= 2 val checkpoints)")
            continue
        results.append((run, diag))

    if not results:
        return

    # Print table
    header = f"{'Run ID':<25} {'Verdict':>11} {'Gap Start':>10} {'Gap End':>10} {'Gap Trend':>10} {'Val Turn':>10}"
    print(header)
    print("-" * len(header))
    for run, d in results:
        turn = str(d["val_turning_step"]) if d["val_turning_step"] else "--"
        tag = {"ok": "OK", "mild": "MILD", "overfitting": "OVERFIT"}[d["verdict"]]
        print(
            f"{run.run_id:<25} "
            f"{tag:>11} "
            f"{d['gap_start']:>10.4f} "
            f"{d['gap_end']:>10.4f} "
            f"{d['gap_trend']:>+10.4f} "
            f"{turn:>10}"
        )

    # Detailed per-step breakdown for single-run mode
    if len(results) == 1:
        run, d = results[0]
        print(f"\n-- Per-checkpoint detail for {run.run_id} --")
        print(f"{'Step':>8} {'Train Loss':>12} {'Val Loss':>12} {'Gap':>10}")
        for step, tl, vl, g in zip(d["steps"], d["train_losses"], d["val_losses"], d["gaps"]):
            print(f"{step:>8} {tl:>12.4f} {vl:>12.4f} {g:>+10.4f}")

        print(f"\nDiagnosis: ", end="")
        if d["verdict"] == "ok":
            print("No overfitting detected. Train-val gap is stable.")
        elif d["verdict"] == "mild":
            print("Mild overfitting signal. Gap is widening slightly or val loss bumped up once.")
            if d["val_turning_step"]:
                print(f"  Val loss first increased at step {d['val_turning_step']}.")
        else:
            print("Overfitting detected. Val loss is rising while train loss drops.")
            print(f"  Val loss first increased at step {d['val_turning_step']}.")
            print(f"  Train-val gap grew by {d['gap_trend']:+.4f} over training.")
            print("  Consider: fewer iterations, more regularization, or smaller model.")




    if not runs:
        print("No log files found in logs/")
        return

    runs.sort(key=lambda r: r.int8_val_bpb or r.final_val_bpb or 999)

    header = f"{'Run ID':<25} {'BPB':>8} {'BPB(q)':>8} {'Loss':>8} {'Params':>12} {'Sub Size':>10} {'VRAM':>8} {'Time':>8} {'Steps':>6}"
    print(header)
    print("─" * len(header))

    for r in runs:
        last_step = r.steps[-1].step if r.steps else 0
        print(
            f"{r.run_id:<25} "
            f"{fmt_bpb(r.final_val_bpb):>8} "
            f"{fmt_bpb(r.int8_val_bpb):>8} "
            f"{fmt_bpb(r.final_val_loss):>8} "
            f"{fmt_int(r.model_params):>12} "
            f"{fmt_mb(r.submission_bytes):>10} "
            f"{fmt_int(r.peak_memory_mib) + 'Mi' if r.peak_memory_mib else '—':>8} "
            f"{fmt_time(r.final_train_time_ms):>8} "
            f"{last_step:>6}"
        )


def cmd_compare(runs: list[RunResult], id1: str, id2: str) -> None:
    r1 = next((r for r in runs if r.run_id == id1), None)
    r2 = next((r for r in runs if r.run_id == id2), None)
    if not r1:
        print(f"Run '{id1}' not found"); return
    if not r2:
        print(f"Run '{id2}' not found"); return

    fields = [
        ("Final BPB", "final_val_bpb", fmt_bpb),
        ("INT8 BPB", "int8_val_bpb", fmt_bpb),
        ("Final Val Loss", "final_val_loss", fmt_bpb),
        ("Model Params", "model_params", fmt_int),
        ("Submission Size", "submission_bytes", fmt_mb),
        ("Peak VRAM (MiB)", "peak_memory_mib", fmt_int),
        ("Train Time", "final_train_time_ms", fmt_time),
        ("Batch Tokens", "train_batch_tokens", fmt_int),
        ("Seq Len", "train_seq_len", fmt_int),
        ("Iterations", "iterations", fmt_int),
        ("Warmup Steps", "warmup_steps", fmt_int),
        ("Wallclock Cap", "max_wallclock_seconds", lambda v: f"{v:.0f}s" if v else "—"),
        ("Num Heads", "num_heads", fmt_int),
        ("Num KV Heads", "num_kv_heads", fmt_int),
        ("Embed LR", "embed_lr", fmt_bpb),
        ("Head LR", "head_lr", fmt_bpb),
        ("Matrix LR", "matrix_lr", fmt_bpb),
        ("Scalar LR", "scalar_lr", fmt_bpb),
        ("Tie Embeddings", "tie_embeddings", lambda v: str(v) if v is not None else "—"),
        ("Train Shards", "train_shards", fmt_int),
        ("World Size", "world_size", fmt_int),
        ("Seed", "seed", fmt_int),
    ]

    w = 22
    print(f"{'':>{w}} {id1:>20} {id2:>20} {'Delta':>12}")
    print("─" * (w + 55))
    for label, attr, fmt in fields:
        v1 = getattr(r1, attr, None)
        v2 = getattr(r2, attr, None)
        s1 = fmt(v1)
        s2 = fmt(v2)
        delta = ""
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            d = v2 - v1
            if isinstance(d, float):
                delta = f"{d:+.4f}"
            else:
                delta = f"{d:+,}"
        diff_marker = " *" if s1 != s2 else ""
        print(f"{label:>{w}} {s1:>20} {s2:>20} {delta:>12}{diff_marker}")


def cmd_detail(runs: list[RunResult], run_id: str) -> None:
    run = next((r for r in runs if r.run_id == run_id), None)
    if not run:
        print(f"Run '{run_id}' not found"); return

    print(f"Run: {run.run_id}")
    print(f"Log: {run.log_path}")
    print()
    print("── Hyperparameters ──")
    for attr in ["model_params", "num_heads", "num_kv_heads", "train_batch_tokens",
                 "train_seq_len", "iterations", "warmup_steps", "max_wallclock_seconds",
                 "embed_lr", "head_lr", "matrix_lr", "scalar_lr", "tie_embeddings",
                 "train_shards", "world_size", "grad_accum_steps", "seed"]:
        val = getattr(run, attr, None)
        if val is not None:
            print(f"  {attr}: {val}")

    print()
    print("── Results ──")
    print(f"  Final BPB:        {fmt_bpb(run.final_val_bpb)}")
    print(f"  INT8 BPB:         {fmt_bpb(run.int8_val_bpb)}")
    print(f"  Final Val Loss:   {fmt_bpb(run.final_val_loss)}")
    print(f"  Train Time:       {fmt_time(run.final_train_time_ms)}")
    print(f"  Peak VRAM:        {fmt_int(run.peak_memory_mib)} MiB")
    print(f"  Model Size:       {fmt_mb(run.model_bytes)}")
    print(f"  Submission Size:  {fmt_mb(run.submission_bytes)}")

    if run.steps:
        print()
        print("── Step Log ──")
        print(f"  {'Step':>6} {'Train Loss':>12} {'Val Loss':>12} {'Val BPB':>12} {'Time':>10}")
        for s in run.steps:
            tl = f"{s.train_loss:.4f}" if s.train_loss is not None else ""
            vl = f"{s.val_loss:.4f}" if s.val_loss is not None else ""
            vb = f"{s.val_bpb:.4f}" if s.val_bpb is not None else ""
            print(f"  {s.step:>6} {tl:>12} {vl:>12} {vb:>12} {fmt_time(s.train_time_ms):>10}")


def cmd_report(runs: list[RunResult], open_browser: bool = True) -> None:
    from report import generate_report
    output = OUTPUT_DIR.parent / "report.html"
    generate_report(runs, output)
    if open_browser:
        webbrowser.open(f"file://{output.resolve()}")


def cmd_add(run_id: str, logs_dir: Path) -> None:
    """Read pasted log output from stdin and save as a log file."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    out = logs_dir / f"{run_id}.txt"
    print(f"Paste the training output, then press Ctrl+D (EOF) when done:")
    try:
        text = sys.stdin.read()
    except KeyboardInterrupt:
        print("\ncancelled")
        return
    # Strip terminal prompt lines and warmup noise, keep the useful stuff
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Skip shell prompt lines
        if stripped.startswith("root@") or stripped.startswith(">"):
            continue
        # Skip warmup lines
        if stripped.startswith("warmup_step:"):
            continue
        lines.append(stripped)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {out} ({len(lines)} lines)")


def cmd_pull(source: str, logs_dir: Path) -> None:
    """Pull log files from a remote host via scp."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    # Ensure source ends with / to copy contents
    if not source.endswith("/"):
        source += "/"
    cmd = ["scp", "-r", f"{source}*.txt", str(logs_dir) + "/"]
    print(f"running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        count = len(list(logs_dir.glob("*.txt")))
        print(f"done — {count} log file(s) in {logs_dir}")
    else:
        # scp glob might not work on all systems, try rsync
        cmd2 = ["rsync", "-avz", "--include=*.txt", "--exclude=*", source, str(logs_dir) + "/"]
        print(f"scp failed, trying: {' '.join(cmd2)}")
        subprocess.run(cmd2)
        count = len(list(logs_dir.glob("*.txt")))
        print(f"done — {count} log file(s) in {logs_dir}")


def cmd_plot(runs: list[RunResult], plot_type: str, param: str | None) -> None:
    try:
        from visualize import (
            plot_bpb_comparison,
            plot_loss_curves,
            plot_memory_and_size,
            plot_param_vs_bpb,
        )
    except ImportError:
        print("matplotlib is required for plots: pip install matplotlib")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    if plot_type == "loss":
        plot_loss_curves(runs, OUTPUT_DIR / "loss_curves.png", metric="train_loss")
        plot_loss_curves(runs, OUTPUT_DIR / "val_loss_curves.png", metric="val_loss")
    elif plot_type == "bpb":
        plot_bpb_comparison(runs, OUTPUT_DIR / "bpb_comparison.png")
    elif plot_type == "param":
        if not param:
            print("specify a parameter name, e.g.: python analyze.py plot param matrix_lr")
            return
        plot_param_vs_bpb(runs, param, OUTPUT_DIR / f"bpb_vs_{param}.png")
    elif plot_type == "memory":
        plot_memory_and_size(runs, OUTPUT_DIR / "memory_and_size.png")
    elif plot_type == "all":
        plot_loss_curves(runs, OUTPUT_DIR / "loss_curves.png", metric="train_loss")
        plot_loss_curves(runs, OUTPUT_DIR / "val_loss_curves.png", metric="val_loss")
        plot_bpb_comparison(runs, OUTPUT_DIR / "bpb_comparison.png")
        plot_memory_and_size(runs, OUTPUT_DIR / "memory_and_size.png")
    else:
        print(f"Unknown plot type: {plot_type}")
        print("Available: loss, bpb, param <name>, memory, all")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze parameter-golf training runs")
    parser.add_argument("--logs-dir", type=Path, default=LOGS_DIR, help="Path to logs directory")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("summary", help="Show summary table of all runs")

    p_overfit = sub.add_parser("overfit", help="Overfitting diagnostic for one or all runs")
    p_overfit.add_argument("run_id", nargs="?", help="Run ID (omit for all runs)")

    p_cmp = sub.add_parser("compare", help="Compare two runs")
    p_cmp.add_argument("run1", help="First run ID")
    p_cmp.add_argument("run2", help="Second run ID")

    p_det = sub.add_parser("detail", help="Show full detail for a run")
    p_det.add_argument("run_id", help="Run ID")

    p_plot = sub.add_parser("plot", help="Generate plots")
    p_plot.add_argument("plot_type", choices=["loss", "bpb", "param", "memory", "all"])
    p_plot.add_argument("param", nargs="?", help="Parameter name (for 'param' plot type)")

    p_report = sub.add_parser("report", help="Generate HTML report and open in browser")
    p_report.add_argument("--no-open", action="store_true", help="Don't open in browser")

    p_add = sub.add_parser("add", help="Paste training output to save as a log file")
    p_add.add_argument("run_id", help="Run ID (filename without .txt)")

    p_pull = sub.add_parser("pull", help="Pull logs from RunPod via scp/rsync")
    p_pull.add_argument("source", help="Remote path, e.g. root@1.2.3.4:/workspace/parameter-golf/logs/")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "pull":
        cmd_pull(args.source, args.logs_dir)
        return
    if args.command == "add":
        cmd_add(args.run_id, args.logs_dir)
        return

    runs = parse_all_logs(args.logs_dir)

    if args.command == "summary":
        cmd_summary(runs)
    elif args.command == "overfit":
        cmd_overfit(runs, getattr(args, "run_id", None))
    elif args.command == "compare":
        cmd_compare(runs, args.run1, args.run2)
    elif args.command == "detail":
        cmd_detail(runs, args.run_id)
    elif args.command == "plot":
        cmd_plot(runs, args.plot_type, getattr(args, "param", None))
    elif args.command == "report":
        cmd_report(runs, open_browser=not args.no_open)


if __name__ == "__main__":
    main()

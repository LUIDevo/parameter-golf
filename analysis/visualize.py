"""Visualization helpers for training runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parse_logs import RunResult


def plot_loss_curves(runs: list[RunResult], output: Path, metric: str = "train_loss") -> None:
    """Overlay training or validation loss curves for multiple runs."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for run in runs:
        steps_with_metric = []
        for s in run.steps:
            val = getattr(s, metric, None)
            if val is not None:
                steps_with_metric.append((s.step, val))
        if steps_with_metric:
            x, y = zip(*steps_with_metric)
            ax.plot(x, y, label=run.run_id, marker="." if len(steps_with_metric) < 50 else None)

    ax.set_xlabel("Step")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} by Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"saved: {output}")


def plot_bpb_comparison(runs: list[RunResult], output: Path) -> None:
    """Bar chart comparing final BPB across runs."""
    runs_with_bpb = [r for r in runs if r.int8_val_bpb is not None]
    if not runs_with_bpb:
        runs_with_bpb = [r for r in runs if r.final_val_bpb is not None]
    if not runs_with_bpb:
        print("no runs with BPB data to plot")
        return

    runs_with_bpb.sort(key=lambda r: r.int8_val_bpb or r.final_val_bpb or 999)

    labels = [r.run_id for r in runs_with_bpb]
    bpbs = [r.int8_val_bpb or r.final_val_bpb for r in runs_with_bpb]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 6))
    bars = ax.bar(labels, bpbs, color="steelblue", edgecolor="black", linewidth=0.5)

    for bar, bpb in zip(bars, bpbs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bpb:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("BPB (lower is better)")
    ax.set_title("Final BPB Comparison (INT8 quantized)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"saved: {output}")


def plot_param_vs_bpb(runs: list[RunResult], param: str, output: Path) -> None:
    """Scatter plot of a hyperparameter vs final BPB."""
    points = []
    for r in runs:
        bpb = r.int8_val_bpb or r.final_val_bpb
        val = getattr(r, param, None)
        if bpb is not None and val is not None:
            points.append((val, bpb, r.run_id))

    if not points:
        print(f"no data for {param} vs BPB")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    xs, ys, labels = zip(*points)
    ax.scatter(xs, ys, s=60, color="steelblue", edgecolors="black", linewidth=0.5)

    for x, y, label in points:
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel(param.replace("_", " ").title())
    ax.set_ylabel("BPB (lower is better)")
    ax.set_title(f"{param.replace('_', ' ').title()} vs BPB")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"saved: {output}")


def plot_memory_and_size(runs: list[RunResult], output: Path) -> None:
    """Bar chart of peak VRAM and submission size per run."""
    runs_with_data = [r for r in runs if r.peak_memory_mib is not None]
    if not runs_with_data:
        print("no memory data to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    labels = [r.run_id for r in runs_with_data]
    mem = [r.peak_memory_mib for r in runs_with_data]
    ax1.bar(labels, mem, color="coral", edgecolor="black", linewidth=0.5)
    ax1.axhline(y=81920, color="red", linestyle="--", alpha=0.5, label="H100 80GB")
    ax1.set_ylabel("Peak VRAM (MiB)")
    ax1.set_title("Peak Memory Usage")
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend()

    runs_with_size = [r for r in runs_with_data if r.submission_bytes is not None]
    if runs_with_size:
        labels2 = [r.run_id for r in runs_with_size]
        sizes_mb = [r.submission_bytes / 1024 / 1024 for r in runs_with_size]
        ax2.bar(labels2, sizes_mb, color="mediumseagreen", edgecolor="black", linewidth=0.5)
        ax2.axhline(y=16, color="red", linestyle="--", alpha=0.5, label="16MB cap")
        ax2.set_ylabel("Submission Size (MB)")
        ax2.set_title("INT8+zlib Submission Size")
        ax2.tick_params(axis="x", rotation=45)
        ax2.legend()

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"saved: {output}")

"""Generate a single HTML report with embedded charts and tables."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parse_logs import RunResult

# Palette inspired by the reference image — vivid neon on dark
NEON_GREEN = "#33ff88"
NEON_CYAN = "#00ccff"
NEON_BLUE = "#5577ff"
NEON_RED = "#ff4466"
NEON_PURPLE = "#bb77ff"
NEON_WHITE = "#e8e8f0"
NEON_PINK = "#ff66aa"
PALETTE = [NEON_GREEN, NEON_CYAN, NEON_BLUE, NEON_RED, NEON_PURPLE, NEON_WHITE, NEON_PINK]

BG = "#1a1b2e"
BG_CARD = "#1e2035"
GRID = "#2d2a4a"
BORDER = "#3a3660"
TEXT = "#d0d0e0"
TEXT_DIM = "#7a7a9a"

# Update this dict manually when you add a new run.
ABLATION_NOTES: dict[str, str] = {
    "smoke_003": "MODEL_DIM=544 NUM_LAYERS=12, 100 iters only",
    "smoke_004": "MODEL_DIM=544 NUM_LAYERS=12, 500 iters",
    "smoke_005": "baseline (default config, 1000 iters)",
    "smoke_006": "TRAIN_BATCH_TOKENS 524k→1048k, ITERS 1000→500",
    "smoke_007": "NUM_LAYERS=14 MODEL_DIM=352",
    "smoke_008": "TRAIN_SEQ_LEN=2048 NUM_LAYERS=18 MODEL_DIM=352",
}


def _apply_style():
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": TEXT_DIM,
        "ytick.color": TEXT_DIM,
        "grid.color": GRID,
        "legend.facecolor": BG_CARD,
        "legend.edgecolor": BORDER,
        "legend.labelcolor": TEXT,
        "font.family": "sans-serif",
        "font.size": 10,
    })


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _make_loss_chart(runs: list[RunResult], metric: str) -> str | None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    any_data = False
    for i, run in enumerate(runs):
        pts = [(s.step, getattr(s, metric)) for s in run.steps if getattr(s, metric, None) is not None]
        if pts:
            x, y = zip(*pts)
            ax.plot(x, y, label=run.run_id, color=PALETTE[i % len(PALETTE)],
                    marker="." if len(pts) < 50 else None, linewidth=2)
            any_data = True
    if not any_data:
        plt.close(fig)
        return None
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=9)
    ax.set_title(metric.replace("_", " ").title(), fontsize=11, loc="left", color=NEON_WHITE)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, linestyle="-")
    return _fig_to_base64(fig)


def _make_bpb_chart(runs: list[RunResult]) -> str | None:
    _apply_style()
    data = [(r.run_id, r.int8_val_bpb or r.final_val_bpb) for r in runs if (r.int8_val_bpb or r.final_val_bpb)]
    if not data:
        return None
    data.sort(key=lambda x: x[1])
    labels, bpbs = zip(*data)
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.6), 3.2))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    bars = ax.bar(labels, bpbs, color=colors, edgecolor=BORDER, linewidth=0.5, alpha=0.9)
    for bar, bpb in zip(bars, bpbs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bpb:.4f}", ha="center", va="bottom", fontsize=8, color=NEON_WHITE)
    ax.set_ylabel("BPB", fontsize=9)
    ax.set_title("BPB Comparison", fontsize=11, loc="left", color=NEON_WHITE)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    return _fig_to_base64(fig)


def _make_memory_chart(runs: list[RunResult]) -> str | None:
    _apply_style()
    data = [(r.run_id, r.peak_memory_mib, r.submission_bytes) for r in runs if r.peak_memory_mib]
    if not data:
        return None
    labels = [d[0] for d in data]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
    ax1.bar(labels, [d[1] for d in data], color=NEON_CYAN, edgecolor=BORDER, linewidth=0.5, alpha=0.85)
    ax1.axhline(y=81920, color=NEON_RED, linestyle="--", alpha=0.7, label="H100 80GB")
    ax1.set_ylabel("VRAM (MiB)", fontsize=9)
    ax1.set_title("VRAM Usage", fontsize=11, loc="left", color=NEON_WHITE)
    ax1.tick_params(axis="x", rotation=45, labelsize=8)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.2, axis="y")
    sized = [(d[0], d[2] / 1024 / 1024) for d in data if d[2]]
    if sized:
        ax2.bar([s[0] for s in sized], [s[1] for s in sized],
                color=NEON_GREEN, edgecolor=BORDER, linewidth=0.5, alpha=0.85)
        ax2.axhline(y=16, color=NEON_RED, linestyle="--", alpha=0.7, label="16MB Cap")
        ax2.set_ylabel("Size (MB)", fontsize=9)
        ax2.set_title("Submission Size", fontsize=11, loc="left", color=NEON_WHITE)
        ax2.tick_params(axis="x", rotation=45, labelsize=8)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.2, axis="y")
    return _fig_to_base64(fig)


def _fmt(val, kind="str") -> str:
    if val is None:
        return "\u2014"
    if kind == "bpb":
        return f"{val:.4f}"
    if kind == "int":
        return f"{val:,}"
    if kind == "mb":
        return f"{val / 1024 / 1024:.1f}"
    if kind == "time":
        if val < 60_000:
            return f"{val / 1000:.1f}s"
        return f"{val / 60_000:.1f}min"
    return str(val)


# ---------------------------------------------------------------------------
# Overfitting diagnosis (copied from analyze.py)
# ---------------------------------------------------------------------------

def _overfit_diagnosis(run: RunResult) -> dict | None:
    """Compute overfitting metrics for a single run.

    Returns a dict with:
      - train_losses / val_losses: aligned lists at validation steps
      - gap_start / gap_end: train-val gap at first and last validation point
      - gap_trend: positive means the gap is widening (overfitting signal)
      - val_turning_step: step where val loss first increases (None if always decreasing)
      - verdict: "ok", "mild", or "overfitting"
    """
    train_by_step = {}
    for s in run.steps:
        if s.train_loss is not None:
            train_by_step[s.step] = s.train_loss
    val_points = [(s.step, s.val_loss) for s in run.steps if s.val_loss is not None]

    if len(val_points) < 2 or not train_by_step:
        return None

    sorted_train_steps = sorted(train_by_step.keys())

    aligned = []
    for v_step, v_loss in val_points:
        tl = None
        for ts in reversed(sorted_train_steps):
            if ts <= v_step:
                tl = train_by_step[ts]
                break
        if tl is None:
            tl = train_by_step[sorted_train_steps[0]]
        aligned.append((v_step, tl, v_loss))

    steps = [a[0] for a in aligned]
    train_losses = [a[1] for a in aligned]
    val_losses = [a[2] for a in aligned]
    gaps = [v - t for t, v in zip(train_losses, val_losses)]

    gap_start = gaps[0]
    gap_end = gaps[-1]
    gap_trend = gap_end - gap_start

    val_turning_step = None
    min_val = val_losses[0]
    for i in range(1, len(val_losses)):
        if val_losses[i] < min_val:
            min_val = val_losses[i]
        elif val_losses[i] > min_val and val_turning_step is None:
            val_turning_step = steps[i]

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


# ---------------------------------------------------------------------------
# Power law helpers
# ---------------------------------------------------------------------------

def _fit_power_law(run: RunResult) -> tuple[float, float] | None:
    """Fit L(t) = a * t^b + 1.0 to val_bpb checkpoints via log-linear regression.

    Returns (a, b) or None if there is not enough data or the fit fails.
    When fewer than 3 val checkpoints exist at step > 0, the step-0
    initialization checkpoint is included (mapped to step=1) so that runs
    with sparse validation logging can still produce an extrapolation.
    """
    pts = [(s.step, s.val_bpb) for s in run.steps if s.val_bpb is not None and s.step > 0]
    if len(pts) < 3:
        # Include step-0 init checkpoint mapped to step=1 for sparse runs
        init = [(1, s.val_bpb) for s in run.steps if s.val_bpb is not None and s.step == 0]
        pts = init + pts
    if len(pts) < 2:
        return None
    steps_arr = np.array([p[0] for p in pts], dtype=float)
    bpbs_arr = np.array([p[1] for p in pts], dtype=float)
    # Need bpb strictly above 1.0 for the log transform log(bpb - 1.0)
    mask = bpbs_arr > 1.001
    if mask.sum() < 2:
        return None
    try:
        log_t = np.log(steps_arr[mask])
        log_y = np.log(bpbs_arr[mask] - 1.0)
        coeffs = np.polyfit(log_t, log_y, 1)
        b = float(coeffs[0])
        a = float(np.exp(coeffs[1]))
        return (a, b)
    except Exception:
        return None


def _pl_predict(a: float, b: float, t: float) -> float:
    return a * (t ** b) + 1.0


def _make_power_law_chart(runs: list[RunResult]) -> str | None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    any_data = False
    for i, run in enumerate(runs):
        pts = [(s.step, s.val_bpb) for s in run.steps if s.val_bpb is not None and s.step > 0]
        if not pts:
            continue
        color = PALETTE[i % len(PALETTE)]
        steps_actual = [p[0] for p in pts]
        bpbs_actual = [p[1] for p in pts]
        ax.scatter(steps_actual, bpbs_actual, color=color, s=18, zorder=3, alpha=0.85)
        params = _fit_power_law(run)
        if params:
            a, b = params
            t_max = max(max(steps_actual), 3000)
            t_curve = np.linspace(max(1, min(steps_actual) * 0.8), t_max, 400)
            y_curve = a * (t_curve ** b) + 1.0
            ax.plot(t_curve, y_curve, color=color, linewidth=1.5, linestyle="--",
                    label=run.run_id, alpha=0.85)
            for t_pred in [1000, 2000, 3000]:
                if t_pred > max(steps_actual):
                    ax.scatter([t_pred], [_pl_predict(a, b, t_pred)],
                               color=color, s=45, marker="x", zorder=4, linewidths=1.5)
        any_data = True
    if not any_data:
        plt.close(fig)
        return None
    for t_mark in [1000, 2000, 3000]:
        ax.axvline(x=t_mark, color=TEXT_DIM, linestyle=":", alpha=0.35, linewidth=0.8)
        ax.text(t_mark, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 2.0,
                f" {t_mark}", fontsize=7, color=TEXT_DIM, va="top")
    ax.set_xlabel("Step", fontsize=9)
    ax.set_ylabel("Val BPB", fontsize=9)
    ax.set_title("Power Law Extrapolation   L(t) = a·t^b + 1", fontsize=11, loc="left", color=NEON_WHITE)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, linestyle="-")
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Compute efficiency chart
# ---------------------------------------------------------------------------

def _make_compute_efficiency_chart(runs: list[RunResult]) -> str | None:
    _apply_style()
    data = []
    for r in runs:
        if r.iterations and r.train_batch_tokens and r.final_val_bpb:
            tokens = r.iterations * r.train_batch_tokens
            data.append((r.run_id, tokens, r.final_val_bpb))
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    tokens_arr = np.array([d[1] for d in data], dtype=float)
    bpbs_arr = np.array([d[2] for d in data], dtype=float)
    for i, (run_id, tokens, bpb) in enumerate(data):
        color = PALETTE[i % len(PALETTE)]
        ax.scatter([tokens], [bpb], color=color, s=60, zorder=3)
        ax.annotate(run_id, (tokens, bpb), textcoords="offset points", xytext=(6, 4),
                    fontsize=7, color=color)
    if len(data) >= 3:
        try:
            log_x = np.log(tokens_arr)
            log_y = np.log(bpbs_arr)
            coeffs = np.polyfit(log_x, log_y, 1)
            b_fit = float(coeffs[0])
            a_fit = float(np.exp(coeffs[1]))
            t_curve = np.linspace(tokens_arr.min() * 0.85, tokens_arr.max() * 1.15, 300)
            y_curve = a_fit * (t_curve ** b_fit)
            ax.plot(t_curve, y_curve, color=TEXT_DIM, linestyle="--", linewidth=1.2,
                    alpha=0.6, label=f"fit: bpb ∝ tokens^{b_fit:.2f}")
            ax.legend(fontsize=7)
        except Exception:
            pass
    ax.set_xlabel("Total Tokens Seen  (iterations × train_batch_tokens)", fontsize=9)
    ax.set_ylabel("Final Val BPB", fontsize=9)
    ax.set_title("Compute Efficiency", fontsize=11, loc="left", color=NEON_WHITE)
    ax.grid(True, alpha=0.3, linestyle="-")
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Step-time bar chart (speed efficiency)
# ---------------------------------------------------------------------------

def _make_step_time_chart(runs: list[RunResult]) -> str | None:
    _apply_style()
    data = []
    for r in runs:
        steps_with_time = [s for s in r.steps if s.step_avg_ms > 0]
        if steps_with_time:
            data.append((r.run_id, steps_with_time[-1].step_avg_ms))
    if not data:
        return None
    data.sort(key=lambda x: x[1], reverse=True)
    labels = [d[0] for d in data]
    values = [d[1] for d in data]
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.75), 3.2))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    bars = ax.bar(labels, values, color=colors, edgecolor=BORDER, linewidth=0.5, alpha=0.9)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val:.0f}ms", ha="center", va="bottom", fontsize=8, color=NEON_WHITE)
    ax.set_ylabel("Step avg (ms)", fontsize=9)
    ax.set_title("Step Time per Run  (slower = costlier per H100-min)", fontsize=11, loc="left", color=NEON_WHITE)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_report(runs: list[RunResult], output: Path) -> None:
    runs_sorted = sorted(runs, key=lambda r: r.int8_val_bpb or r.final_val_bpb or 999)

    # Build leaderboard rows (with ablation, bpb/min, step_avg_ms columns)
    rows = ""
    for i, r in enumerate(runs_sorted):
        last_step = r.steps[-1].step if r.steps else 0
        bpb = r.int8_val_bpb or r.final_val_bpb
        is_best = r == runs_sorted[0] if runs_sorted else False
        cls = ' class="best"' if is_best else ""
        color = PALETTE[i % len(PALETTE)]

        note = ABLATION_NOTES.get(r.run_id, "")

        if bpb is not None and r.final_train_time_ms and r.final_train_time_ms > 0:
            bpb_per_min = bpb / (r.final_train_time_ms / 60_000)
            bpb_per_min_str = f"{bpb_per_min:.4f}"
        else:
            bpb_per_min_str = "—"

        steps_with_time = [s for s in r.steps if s.step_avg_ms > 0]
        step_avg_str = f"{steps_with_time[-1].step_avg_ms:.0f}" if steps_with_time else "—"

        rows += f"""<tr{cls}>
            <td><span class="run-dot" style="background:{color}"></span>{r.run_id}</td>
            <td>{_fmt(bpb, 'bpb')}</td>
            <td>{_fmt(r.final_val_loss, 'bpb')}</td>
            <td>{_fmt(r.model_params, 'int')}</td>
            <td>{_fmt(r.submission_bytes, 'mb')}</td>
            <td>{_fmt(r.peak_memory_mib, 'int')}</td>
            <td>{_fmt(r.final_train_time_ms, 'time')}</td>
            <td>{last_step}</td>
            <td class="dim">{bpb_per_min_str}</td>
            <td class="dim">{step_avg_str}</td>
            <td class="dim note">{note}</td>
        </tr>\n"""

    # Build run detail cards (with 16MB headroom bar)
    details = ""
    for i, r in enumerate(runs_sorted):
        color = PALETTE[i % len(PALETTE)]
        hp_rows = ""
        for attr, label in [
            ("model_params", "Parameters"), ("num_heads", "Heads"), ("num_kv_heads", "KV Heads"),
            ("train_batch_tokens", "Batch Tokens"), ("train_seq_len", "Seq Len"),
            ("iterations", "Iterations"), ("warmup_steps", "Warmup Steps"),
            ("max_wallclock_seconds", "Wallclock Cap"),
            ("embed_lr", "Embed LR"), ("head_lr", "Head LR"),
            ("matrix_lr", "Matrix LR"), ("scalar_lr", "Scalar LR"),
            ("tie_embeddings", "Tie Embeddings"), ("train_shards", "Train Shards"),
            ("world_size", "World Size"), ("grad_accum_steps", "Grad Accum"), ("seed", "Seed"),
        ]:
            val = getattr(r, attr, None)
            if val is not None:
                hp_rows += f"<tr><td>{label}</td><td>{val}</td></tr>\n"

        step_rows = ""
        for s in r.steps:
            step_rows += f"""<tr>
                <td>{s.step}</td>
                <td>{f'{s.train_loss:.4f}' if s.train_loss is not None else ''}</td>
                <td>{f'{s.val_loss:.4f}' if s.val_loss is not None else ''}</td>
                <td>{f'{s.val_bpb:.4f}' if s.val_bpb is not None else ''}</td>
                <td>{_fmt(s.train_time_ms, 'time')}</td>
            </tr>\n"""

        # 16MB headroom bar
        cap_bytes = 16_000_000
        if r.submission_bytes is not None:
            sb = r.submission_bytes
            pct = min(sb / cap_bytes * 100, 100)
            mb_used = sb / 1_000_000
            mb_remaining = (cap_bytes - sb) / 1_000_000
            if sb < 13_000_000:
                bar_color = NEON_GREEN
                bar_label_color = NEON_GREEN
            elif sb < 15_000_000:
                bar_color = "#ffaa00"
                bar_label_color = "#ffaa00"
            else:
                bar_color = NEON_RED
                bar_label_color = NEON_RED
            headroom_html = f"""
            <div class="headroom">
                <div class="headroom-label" style="color:{bar_label_color}">
                    {mb_used:.1f} MB / 16.0 MB — {mb_remaining:.1f} MB remaining
                </div>
                <div class="headroom-track">
                    <div class="headroom-fill" style="width:{pct:.1f}%;background:{bar_color}"></div>
                </div>
            </div>"""
        else:
            headroom_html = ""

        details += f"""
        <div class="run-detail" id="detail-{r.run_id}" style="border-left: 3px solid {color}">
            <h3><span class="run-dot" style="background:{color}"></span>{r.run_id}</h3>
            {headroom_html}
            <div class="detail-grid">
                <div>
                    <h4>Hyperparameters</h4>
                    <table class="detail-table"><tbody>{hp_rows}</tbody></table>
                </div>
                <div>
                    <h4>Results</h4>
                    <table class="detail-table"><tbody>
                        <tr><td>Final BPB</td><td>{_fmt(r.final_val_bpb, 'bpb')}</td></tr>
                        <tr><td>INT8 BPB</td><td>{_fmt(r.int8_val_bpb, 'bpb')}</td></tr>
                        <tr><td>Val Loss</td><td>{_fmt(r.final_val_loss, 'bpb')}</td></tr>
                        <tr><td>Train Time</td><td>{_fmt(r.final_train_time_ms, 'time')}</td></tr>
                        <tr><td>Peak VRAM</td><td>{_fmt(r.peak_memory_mib, 'int')} MiB</td></tr>
                        <tr><td>Model Size</td><td>{_fmt(r.model_bytes, 'mb')} MB</td></tr>
                        <tr><td>Submission</td><td>{_fmt(r.submission_bytes, 'mb')} MB / 16 MB</td></tr>
                    </tbody></table>
                </div>
            </div>
            <details><summary>Step log ({len(r.steps)} entries)</summary>
            <table class="step-table">
                <thead><tr><th>Step</th><th>Train Loss</th><th>Val Loss</th><th>Val BPB</th><th>Time</th></tr></thead>
                <tbody>{step_rows}</tbody>
            </table>
            </details>
        </div>"""

    # Existing charts
    charts_html = ""
    chart_items = [
        ("BPB Comparison", _make_bpb_chart(runs)),
        ("Training Loss", _make_loss_chart(runs, "train_loss")),
        ("Validation Loss", _make_loss_chart(runs, "val_loss")),
        ("Memory & Submission Size", _make_memory_chart(runs)),
    ]
    active = [(label, img) for label, img in chart_items if img]
    for label, img_data in active:
        charts_html += f'<div class="chart"><img src="data:image/png;base64,{img_data}" alt="{label}"></div>\n'

    # --- Section 1: Power law extrapolation ---
    pl_table_rows = ""
    for i, r in enumerate(runs_sorted):
        params = _fit_power_law(r)
        color = PALETTE[i % len(PALETTE)]
        if params:
            a, b = params
            p1 = f"{_pl_predict(a, b, 1000):.4f}"
            p2 = f"{_pl_predict(a, b, 2000):.4f}"
            p3 = f"{_pl_predict(a, b, 3000):.4f}"
            a_str = f"{a:.4f}"
            b_str = f"{b:.4f}"
        else:
            p1 = p2 = p3 = a_str = b_str = "—"
        pl_table_rows += f"""<tr>
            <td><span class="run-dot" style="background:{color}"></span>{r.run_id}</td>
            <td class="mono">{a_str}</td>
            <td class="mono">{b_str}</td>
            <td class="mono">{p1}</td>
            <td class="mono">{p2}</td>
            <td class="mono">{p3}</td>
        </tr>\n"""

    pl_chart_img = _make_power_law_chart(runs)
    pl_chart_html = ""
    if pl_chart_img:
        pl_chart_html = f'<div class="chart chart-wide"><img src="data:image/png;base64,{pl_chart_img}" alt="Power Law Extrapolation"></div>'

    pl_section = f"""
    <h2>Power Law Extrapolation</h2>
    <p class="section-note">Fit: L(t) = a · t<sup>b</sup> + 1.0 &nbsp;·&nbsp; Extrapolated val BPB at fixed step counts.</p>
    <table>
        <thead><tr>
            <th>Run ID</th><th>a</th><th>b</th>
            <th>BPB @ 1000</th><th>BPB @ 2000</th><th>BPB @ 3000</th>
        </tr></thead>
        <tbody>{pl_table_rows}</tbody>
    </table>
    {pl_chart_html}
    """

    # --- Section 2: Compute efficiency ---
    ce_img = _make_compute_efficiency_chart(runs)
    ce_section = ""
    if ce_img:
        ce_section = f"""
    <h2>Compute Efficiency</h2>
    <p class="section-note">Normalizes runs with different batch sizes — x-axis is total tokens seen.</p>
    <div class="chart chart-wide"><img src="data:image/png;base64,{ce_img}" alt="Compute Efficiency"></div>
    """

    # --- Section 3: Overfitting ---
    overfit_rows = ""
    for i, r in enumerate(runs_sorted):
        color = PALETTE[i % len(PALETTE)]
        diag = _overfit_diagnosis(r)
        if diag is None:
            overfit_rows += f"""<tr>
                <td><span class="run-dot" style="background:{color}"></span>{r.run_id}</td>
                <td colspan="5" class="dim">insufficient data (&lt;2 val checkpoints)</td>
            </tr>\n"""
            continue
        verdict = diag["verdict"]
        verdict_color = {"ok": NEON_GREEN, "mild": "#ffaa00", "overfitting": NEON_RED}[verdict]
        verdict_label = {"ok": "OK", "mild": "MILD", "overfitting": "OVERFIT"}[verdict]
        turn = str(diag["val_turning_step"]) if diag["val_turning_step"] else "—"
        overfit_rows += f"""<tr>
            <td><span class="run-dot" style="background:{color}"></span>{r.run_id}</td>
            <td class="mono">{diag['gap_start']:+.4f}</td>
            <td class="mono">{diag['gap_end']:+.4f}</td>
            <td class="mono">{diag['gap_trend']:+.4f}</td>
            <td class="mono">{turn}</td>
            <td><span class="verdict" style="color:{verdict_color};border-color:{verdict_color}">{verdict_label}</span></td>
        </tr>\n"""

    overfit_section = f"""
    <h2>Overfitting</h2>
    <table>
        <thead><tr>
            <th>Run ID</th><th>Gap Start</th><th>Gap End</th>
            <th>Gap Trend</th><th>Val Turn Step</th><th>Verdict</th>
        </tr></thead>
        <tbody>{overfit_rows}</tbody>
    </table>
    """

    # --- Section 4: Learning velocity (derivative) ---
    deriv_rows = ""
    for i, r in enumerate(runs_sorted):
        color = PALETTE[i % len(PALETTE)]
        params = _fit_power_law(r)
        if params is None:
            deriv_rows += f"""<tr>
                <td><span class="run-dot" style="background:{color}"></span>{r.run_id}</td>
                <td colspan="4" class="dim">insufficient data</td>
            </tr>\n"""
            continue
        a, b = params
        last_step = max((s.step for s in r.steps if s.val_bpb is not None and s.step > 0), default=0)
        if last_step == 0:
            deriv_rows += f"""<tr>
                <td><span class="run-dot" style="background:{color}"></span>{r.run_id}</td>
                <td colspan="4" class="dim">no val steps</td>
            </tr>\n"""
            continue
        deriv = a * b * (last_step ** (b - 1))
        drop_per_1k = deriv * 1000
        # Color: more negative = greener (more headroom)
        if drop_per_1k < -0.5:
            d_color = NEON_GREEN
        elif drop_per_1k < -0.1:
            d_color = NEON_CYAN
        else:
            d_color = TEXT_DIM
        deriv_rows += f"""<tr>
            <td><span class="run-dot" style="background:{color}"></span>{r.run_id}</td>
            <td class="mono">{last_step}</td>
            <td class="mono">{deriv:.6f}</td>
            <td class="mono" style="color:{d_color}">{drop_per_1k:+.4f}</td>
        </tr>\n"""

    deriv_section = f"""
    <h2>Learning Velocity (dBPB/dt)</h2>
    <p class="section-note">Instantaneous BPB derivative at the final step via power law fit.
    More negative = still improving fast = more headroom for longer training.</p>
    <table>
        <thead><tr>
            <th>Run ID</th><th>Last Step</th><th>dBPB/dt</th><th>Per 1k Steps</th>
        </tr></thead>
        <tbody>{deriv_rows}</tbody>
    </table>
    """

    # --- Section 5 (Speed Efficiency bar chart) ---
    st_img = _make_step_time_chart(runs)
    speed_section = ""
    if st_img:
        speed_section = f"""
    <h2>Speed Efficiency</h2>
    <p class="section-note">step_avg_ms from last recorded step. Slower steps cost more per H100-minute.
    BPB/min column in the leaderboard normalizes final BPB by training time.</p>
    <div class="chart chart-wide"><img src="data:image/png;base64,{st_img}" alt="Step Time"></div>
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Parameter Golf</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        background: {BG};
        color: {TEXT};
        padding: 28px 32px;
        line-height: 1.6;
        font-size: 14px;
        max-width: 1300px;
        margin: 0 auto;
    }}
    h1 {{
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 2px;
        color: {NEON_WHITE};
        letter-spacing: -0.5px;
    }}
    h2 {{
        font-size: 15px;
        font-weight: 600;
        color: {TEXT_DIM};
        margin: 36px 0 14px;
        padding-bottom: 8px;
        border-bottom: 1px solid {BORDER};
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }}
    h3 {{
        font-size: 14px;
        font-weight: 600;
        color: {NEON_WHITE};
        margin-bottom: 10px;
    }}
    h4 {{
        font-size: 11px;
        font-weight: 600;
        color: {TEXT_DIM};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }}
    .subtitle {{
        color: {TEXT_DIM};
        font-size: 13px;
        margin-bottom: 28px;
    }}
    .section-note {{
        color: {TEXT_DIM};
        font-size: 12px;
        margin-bottom: 12px;
    }}
    .run-dot {{
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        vertical-align: middle;
    }}

    table {{ border-collapse: collapse; width: 100%; margin-bottom: 14px; }}
    th, td {{
        padding: 9px 14px;
        text-align: left;
        border-bottom: 1px solid {GRID};
        font-size: 13px;
    }}
    th {{
        background: {BG_CARD};
        color: {TEXT_DIM};
        font-weight: 600;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
        position: sticky;
        top: 0;
    }}
    td {{ font-variant-numeric: tabular-nums; }}
    td.dim {{ color: {TEXT_DIM}; font-size: 12px; }}
    td.note {{ font-size: 11px; max-width: 240px; }}
    td.mono {{ font-family: monospace; font-size: 12px; }}
    tr:hover {{ background: {BG_CARD}; }}
    tr.best {{ background: rgba(51, 255, 136, 0.06); }}
    tr.best td:nth-child(2) {{ color: {NEON_GREEN}; font-weight: 700; }}

    .verdict {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}

    .charts-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
        margin: 14px 0;
    }}
    .chart {{
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 6px;
        background: {BG_CARD};
    }}
    .chart-wide {{
        grid-column: 1 / -1;
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 6px;
        background: {BG_CARD};
        margin: 10px 0;
    }}
    .chart img, .chart-wide img {{
        width: 100%;
        border-radius: 6px;
        display: block;
    }}

    .run-detail {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 18px;
        margin: 12px 0;
    }}
    .detail-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    .detail-table td:first-child {{ color: {TEXT_DIM}; }}
    .step-table {{ font-size: 12px; }}
    .step-table td {{ color: {TEXT}cc; }}
    details {{ margin-top: 12px; }}
    summary {{
        cursor: pointer;
        color: {NEON_CYAN};
        padding: 6px 0;
        font-size: 13px;
    }}
    summary:hover {{ color: {NEON_GREEN}; }}

    .headroom {{
        margin: 8px 0 14px;
    }}
    .headroom-label {{
        font-size: 12px;
        margin-bottom: 4px;
        font-variant-numeric: tabular-nums;
    }}
    .headroom-track {{
        background: {GRID};
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
    }}
    .headroom-fill {{
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s;
    }}

    .footer {{
        color: {GRID};
        margin-top: 48px;
        text-align: center;
        font-size: 12px;
    }}
    @media (max-width: 900px) {{
        .charts-grid {{ grid-template-columns: 1fr; }}
        .detail-grid {{ grid-template-columns: 1fr; }}
    }}
</style>
</head>
<body>
    <h1>Parameter Golf</h1>
    <p class="subtitle">{len(runs)} run(s) analyzed</p>

    <h2>Leaderboard</h2>
    <table>
        <thead><tr>
            <th>Run ID</th><th>BPB</th><th>Val Loss</th><th>Params</th>
            <th>Size MB</th><th>VRAM MiB</th><th>Time</th><th>Steps</th>
            <th>BPB/min</th><th>Step ms</th><th>Changed</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>

    <h2>Charts</h2>
    <div class="charts-grid">
    {charts_html}
    </div>

    {pl_section}
    {deriv_section}
    {ce_section}
    {overfit_section}
    {speed_section}

    <h2>Run Details</h2>
    {details}

    <p class="footer">generated by analysis/report.py</p>
</body>
</html>"""

    output.write_text(html, encoding="utf-8")
    print(f"report saved: {output}")

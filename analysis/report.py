"""Generate a single HTML report with embedded charts and tables."""

from __future__ import annotations

import base64
import io
from pathlib import Path

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


def generate_report(runs: list[RunResult], output: Path) -> None:
    runs_sorted = sorted(runs, key=lambda r: r.int8_val_bpb or r.final_val_bpb or 999)

    rows = ""
    for i, r in enumerate(runs_sorted):
        last_step = r.steps[-1].step if r.steps else 0
        bpb = r.int8_val_bpb or r.final_val_bpb
        is_best = r == runs_sorted[0] if runs_sorted else False
        cls = ' class="best"' if is_best else ""
        color = PALETTE[i % len(PALETTE)]
        rows += f"""<tr{cls}>
            <td><span class="run-dot" style="background:{color}"></span>{r.run_id}</td>
            <td>{_fmt(bpb, 'bpb')}</td>
            <td>{_fmt(r.final_val_loss, 'bpb')}</td>
            <td>{_fmt(r.model_params, 'int')}</td>
            <td>{_fmt(r.submission_bytes, 'mb')}</td>
            <td>{_fmt(r.peak_memory_mib, 'int')}</td>
            <td>{_fmt(r.final_train_time_ms, 'time')}</td>
            <td>{last_step}</td>
        </tr>\n"""

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

        details += f"""
        <div class="run-detail" id="detail-{r.run_id}" style="border-left: 3px solid {color}">
            <h3><span class="run-dot" style="background:{color}"></span>{r.run_id}</h3>
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
        max-width: 1200px;
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
    tr:hover {{ background: {BG_CARD}; }}
    tr.best {{ background: rgba(51, 255, 136, 0.06); }}
    tr.best td:nth-child(2) {{ color: {NEON_GREEN}; font-weight: 700; }}

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
    .chart img {{
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
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>

    <h2>Charts</h2>
    <div class="charts-grid">
    {charts_html}
    </div>

    <h2>Run Details</h2>
    {details}

    <p class="footer">generated by analysis/report.py</p>
</body>
</html>"""

    output.write_text(html, encoding="utf-8")
    print(f"report saved: {output}")

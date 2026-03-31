"""Generate a standalone interactive simulator HTML for parameter-golf."""

from __future__ import annotations

import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Palette (matches report.py)
# ---------------------------------------------------------------------------
NEON_GREEN  = "#33ff88"
NEON_CYAN   = "#00ccff"
NEON_BLUE   = "#5577ff"
NEON_RED    = "#ff4466"
NEON_AMBER  = "#ffaa00"
NEON_PURPLE = "#bb77ff"
NEON_WHITE  = "#e8e8f0"
BG          = "#1a1b2e"
BG_CARD     = "#1e2035"
GRID        = "#2d2a4a"
BORDER      = "#3a3660"
TEXT        = "#d0d0e0"
TEXT_DIM    = "#7a7a9a"

# ---------------------------------------------------------------------------
# ARCH_CONFIGS (mirrored from bo_tune.py)
# ---------------------------------------------------------------------------
ARCH_CONFIGS = [
    (384,  6,  3),
    (384,  8,  4),
    (448,  8,  4),
    (512,  8,  4),
    (512,  8,  8),
    (512, 16,  4),
    (576,  8,  4),
    (576, 12,  4),
    (640,  8,  4),
    (640, 10,  5),
    (768,  8,  4),
    (768, 12,  4),
    (768, 16,  8),
]

# Run IDs known from ABLATION_NOTES in report.py
KNOWN_RUNS = ["smoke_005", "smoke_006", "smoke_007", "smoke_008"]

# ---------------------------------------------------------------------------
# Activation function SVG helpers
# ---------------------------------------------------------------------------

def _act_svg(fn: str, w: int = 96, h: int = 48) -> str:
    n = 120
    xs = [i * 6 / (n - 1) - 3 for i in range(n)]

    if fn == "tanh":
        ys = [math.tanh(x) for x in xs]
    elif fn == "relu":
        ys = [max(0.0, x) for x in xs]
    elif fn == "gelu":
        ys = [x * 0.5 * (1 + math.erf(x / math.sqrt(2))) for x in xs]
    elif fn == "swiglu":          # SiLU gate (the activation in SwiGLU)
        ys = [x / (1 + math.exp(-x)) for x in xs]
    elif fn == "leakyrelu":
        ys = [x if x >= 0 else 0.1 * x for x in xs]
    else:
        ys = xs

    x_lo, x_hi = -3.0, 3.0
    y_lo, y_hi = -1.4, 2.0

    def px(x: float) -> float:
        return (x - x_lo) / (x_hi - x_lo) * w

    def py(y: float) -> float:
        yc = min(max(y, y_lo), y_hi)
        return h - (yc - y_lo) / (y_hi - y_lo) * h

    axis_y = py(0.0)
    axis_x = px(0.0)
    pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))

    return (
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'style="background:{BG};border-radius:3px;border:1px solid {BORDER};display:block">'
        f'<line x1="0" y1="{axis_y:.1f}" x2="{w}" y2="{axis_y:.1f}" '
        f'stroke="{GRID}" stroke-width="1"/>'
        f'<line x1="{axis_x:.1f}" y1="0" x2="{axis_x:.1f}" y2="{h}" '
        f'stroke="{GRID}" stroke-width="1"/>'
        f'<polyline points="{pts}" fill="none" stroke="{NEON_GREEN}" '
        f'stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>'
        f'</svg>'
    )


# ---------------------------------------------------------------------------
# Arch table rows (Python-computed, embedded in HTML)
# ---------------------------------------------------------------------------

def _arch_table_rows() -> str:
    rows = ""
    vocab = 1024
    for i, (dim, heads, kv) in enumerate(ARCH_CONFIGS):
        layers_ref = 9
        mlp_ref = 4
        # Accurate param estimate
        head_dim = dim // heads
        attn = dim * heads * head_dim + 2 * dim * kv * head_dim + heads * head_dim * dim
        mlp = 2 * dim * (mlp_ref * dim)
        ln = 4 * dim
        params = layers_ref * (attn + mlp + ln) + vocab * dim
        size_mb = params * 0.75 / 1e6  # INT8+zlib estimate
        fits = f'<span style="color:{NEON_GREEN}">✓</span>' if size_mb < 16 else f'<span style="color:{NEON_RED}">✗</span>'
        rows += (
            f"<tr>"
            f"<td>{i}</td><td>{dim}</td><td>{heads}</td><td>{kv}</td>"
            f"<td>{params/1e6:.2f}M</td><td>{size_mb:.1f}</td><td>{fits}</td>"
            f"</tr>\n"
        )
    return rows


# ---------------------------------------------------------------------------
# HTML template  (uses __MARKER__ tokens; no f-string brace escaping needed)
# ---------------------------------------------------------------------------

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Parameter Golf — Simulator</title>
<style>
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

:root {
  --bg: __BG__;
  --bg-card: __BG_CARD__;
  --grid: __GRID__;
  --border: __BORDER__;
  --text: __TEXT__;
  --dim: __TEXT_DIM__;
  --green: __NEON_GREEN__;
  --cyan: __NEON_CYAN__;
  --red: __NEON_RED__;
  --amber: __NEON_AMBER__;
  --white: __NEON_WHITE__;
  --blue: __NEON_BLUE__;
  --purple: __NEON_PURPLE__;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  font-size: 14px;
  line-height: 1.5;
  min-height: 100vh;
}

h1 { font-size: 22px; font-weight: 700; color: var(--white); letter-spacing: -0.3px; }
h2 { font-size: 13px; font-weight: 600; color: var(--dim); text-transform: uppercase;
     letter-spacing: 1.4px; margin: 24px 0 12px; padding-bottom: 7px;
     border-bottom: 1px solid var(--border); }
h3 { font-size: 13px; font-weight: 600; color: var(--white); margin-bottom: 8px; }

.page-header {
  padding: 20px 28px 16px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: baseline; gap: 16px;
}
.page-header .sub { color: var(--dim); font-size: 13px; }

/* Tabs */
.tab-bar {
  display: flex; gap: 0;
  border-bottom: 1px solid var(--border);
  padding: 0 28px;
  background: var(--bg-card);
}
.tab-btn {
  padding: 11px 20px;
  background: none; border: none; border-bottom: 2px solid transparent;
  color: var(--dim); font-size: 13px; font-weight: 500;
  cursor: pointer; margin-bottom: -1px;
  transition: color 0.15s, border-color 0.15s;
}
.tab-btn:hover { color: var(--text); }
.tab-btn.active { color: var(--green); border-bottom-color: var(--green); }

.tab-panel { display: none; padding: 24px 28px 48px; }
.tab-panel.active { display: block; }

/* Two-column layout */
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.cols3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
@media (max-width: 900px) { .cols, .cols3 { grid-template-columns: 1fr; } }

/* Cards */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
}

/* Slider rows */
.field { margin-bottom: 12px; }
.field label {
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: 12px; color: var(--dim); margin-bottom: 4px;
}
.field label .val { color: var(--white); font-weight: 600; font-variant-numeric: tabular-nums; }
input[type=range] {
  width: 100%; height: 4px; cursor: pointer;
  accent-color: var(--green);
}
select, input[type=text], input[type=number], textarea {
  background: var(--bg); border: 1px solid var(--border);
  color: var(--text); padding: 6px 10px; border-radius: 5px;
  font-size: 13px; width: 100%;
}
select:focus, input:focus, textarea:focus {
  outline: none; border-color: var(--green);
}
textarea { font-family: monospace; font-size: 12px; resize: vertical; }

/* Metrics panel */
.metrics-grid {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 8px; margin-bottom: 10px;
}
.metric {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px 12px;
}
.metric .label { font-size: 10px; color: var(--dim); text-transform: uppercase;
                  letter-spacing: 0.8px; margin-bottom: 2px; }
.metric .value { font-size: 18px; font-weight: 700; color: var(--white);
                  font-variant-numeric: tabular-nums; }
.metric .value.green { color: var(--green); }
.metric .value.red   { color: var(--red); }
.metric .value.amber { color: var(--amber); }
.metric .sub-value { font-size: 11px; color: var(--dim); margin-top: 1px; }

/* Command box */
.cmd-box {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px 14px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12px;
  color: var(--cyan);
  white-space: pre;
  overflow-x: auto;
  line-height: 1.8;
  position: relative;
}
.copy-btn {
  position: absolute; top: 8px; right: 8px;
  background: var(--border); border: none; border-radius: 4px;
  color: var(--text); font-size: 11px; padding: 3px 8px;
  cursor: pointer;
}
.copy-btn:hover { background: var(--green); color: var(--bg); }

/* Range bar for BO params */
.range-bar-row { margin-bottom: 10px; }
.range-bar-row .name { font-size: 12px; color: var(--white); margin-bottom: 3px;
                        display: flex; justify-content: space-between; }
.range-bar-row .name span { color: var(--dim); font-size: 11px; }
.range-bar-track {
  background: var(--grid); border-radius: 3px; height: 6px;
  position: relative;
}
.range-bar-fill {
  background: var(--blue); border-radius: 3px; height: 100%;
}

/* Tables */
table { border-collapse: collapse; width: 100%; margin-bottom: 12px; font-size: 13px; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--grid); }
th { background: var(--bg); color: var(--dim); font-size: 10px; font-weight: 600;
     text-transform: uppercase; letter-spacing: 0.9px; }
tr:hover { background: var(--bg-card); }
td { font-variant-numeric: tabular-nums; }

/* Verdict badges */
.badge {
  display: inline-block; padding: 2px 7px; border-radius: 4px;
  font-size: 10px; font-weight: 700; letter-spacing: 0.4px;
}
.badge-green { background: rgba(51,255,136,0.12); color: var(--green); }
.badge-red   { background: rgba(255,68,102,0.12);  color: var(--red); }
.badge-amber { background: rgba(255,170,0,0.12);   color: var(--amber); }

/* Activation row */
.act-row { display: flex; align-items: center; gap: 10px; }
.act-row select { flex: 1; }
.act-svg-wrap { flex-shrink: 0; }

/* Experiment card output */
.exp-card-out {
  background: var(--bg);
  border: 1px solid var(--green);
  border-radius: 6px;
  padding: 14px;
  font-family: monospace;
  font-size: 12px;
  color: var(--text);
  white-space: pre-wrap;
  display: none;
  margin-top: 14px;
}

.btn {
  background: var(--green); color: var(--bg);
  border: none; border-radius: 5px;
  padding: 8px 16px; font-size: 13px; font-weight: 600;
  cursor: pointer; margin-top: 8px;
}
.btn:hover { opacity: 0.85; }
.btn-outline {
  background: none; border: 1px solid var(--border);
  color: var(--text); border-radius: 5px;
  padding: 7px 14px; font-size: 12px; cursor: pointer;
}
.btn-outline:hover { border-color: var(--green); color: var(--green); }

.warn { color: var(--amber); font-size: 12px; margin-top: 6px; }
.info { color: var(--dim);   font-size: 12px; margin-top: 6px; }
.section-note { color: var(--dim); font-size: 12px; margin: -8px 0 12px; }

/* nproc row */
.inline-row { display: flex; gap: 10px; align-items: center; }
.inline-row label { font-size: 12px; color: var(--dim); white-space: nowrap; }
.inline-row select { width: auto; }
</style>
</head>
<body>

<div class="page-header">
  <h1>Parameter Golf — Simulator</h1>
  <span class="sub">interactive design tool · dark matter edition</span>
</div>

<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('arch')">Architecture &amp; Training</button>
  <button class="tab-btn"        onclick="switchTab('bo')">Bayesian Optimizer</button>
  <button class="tab-btn"        onclick="switchTab('exp')">Experiment Planner</button>
</div>

<!-- ═══════════════════════════════════════════════════════════════
     TAB 1 — Architecture & Training Simulator
     ═══════════════════════════════════════════════════════════════ -->
<div id="tab-arch" class="tab-panel active">
  <div class="cols">

    <!-- LEFT: sliders -->
    <div>

      <h2>Architecture</h2>
      <div class="card">
        <div class="field">
          <label>MODEL_DIM <span class="val" id="v-dim">512</span></label>
          <input type="range" id="s-dim" min="256" max="1024" step="64" value="512"
                 oninput="update()">
        </div>
        <div class="field">
          <label>NUM_LAYERS <span class="val" id="v-layers">9</span></label>
          <input type="range" id="s-layers" min="4" max="24" step="1" value="9"
                 oninput="update()">
        </div>
        <div class="field">
          <label>NUM_HEADS <span class="val" id="v-heads">8</span></label>
          <select id="s-heads" onchange="syncKvHeads(); update()">
            <option value="4">4</option>
            <option value="6">6</option>
            <option value="8" selected>8</option>
            <option value="12">12</option>
            <option value="16">16</option>
          </select>
        </div>
        <div class="field">
          <label>NUM_KV_HEADS <span class="val" id="v-kvheads">4</span></label>
          <select id="s-kvheads" onchange="update()">
            <option value="4" selected>4</option>
          </select>
        </div>
        <div class="field">
          <label>MLP_MULT <span class="val" id="v-mlp">4</span></label>
          <select id="s-mlp" onchange="update()">
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4" selected>4</option>
          </select>
        </div>
        <div class="field">
          <label>Activation</label>
          <div class="act-row">
            <select id="s-act" onchange="updateActSvg()">
              <option value="gelu"     selected>GeLU (default)</option>
              <option value="swiglu">SwiGLU (LLaMA)</option>
              <option value="tanh">tanh (micrograd)</option>
              <option value="relu">ReLU</option>
              <option value="leakyrelu">LeakyReLU</option>
            </select>
            <div class="act-svg-wrap">
              <div id="svg-gelu"      class="act-svg">__SVG_GELU__</div>
              <div id="svg-swiglu"    class="act-svg" style="display:none">__SVG_SWIGLU__</div>
              <div id="svg-tanh"      class="act-svg" style="display:none">__SVG_TANH__</div>
              <div id="svg-relu"      class="act-svg" style="display:none">__SVG_RELU__</div>
              <div id="svg-leakyrelu" class="act-svg" style="display:none">__SVG_LEAKYRELU__</div>
            </div>
          </div>
        </div>
      </div>

      <h2>Training</h2>
      <div class="card">
        <div class="field">
          <label>ITERATIONS <span class="val" id="v-iters">1000</span></label>
          <input type="range" id="s-iters" min="200" max="5000" step="100" value="1000"
                 oninput="update()">
        </div>
        <div class="field">
          <label>TRAIN_BATCH_TOKENS <span class="val" id="v-batch">524,288</span></label>
          <select id="s-batch" onchange="update()">
            <option value="131072">131,072</option>
            <option value="262144">262,144</option>
            <option value="524288" selected>524,288</option>
            <option value="1048576">1,048,576</option>
          </select>
        </div>
        <div class="field">
          <label>TRAIN_SEQ_LEN <span class="val" id="v-seqlen">1024</span></label>
          <select id="s-seqlen" onchange="update()">
            <option value="512">512</option>
            <option value="1024" selected>1024</option>
            <option value="2048">2048</option>
            <option value="4096">4096</option>
          </select>
        </div>
        <div class="field">
          <label>WARMUP_STEPS <span class="val" id="v-warmup">100</span></label>
          <input type="range" id="s-warmup" min="10" max="500" step="10" value="100"
                 oninput="update()">
        </div>
        <div class="field">
          <label>WARMDOWN_ITERS <span class="val" id="v-warmdown">300</span></label>
          <input type="range" id="s-warmdown" min="100" max="2000" step="100" value="300"
                 oninput="update()">
        </div>
      </div>

      <h2>Optimizer</h2>
      <div class="card">
        <div class="field">
          <label>Optimizer type</label>
          <select id="s-optim" onchange="update()">
            <option value="muon" selected>Muon (current)</option>
            <option value="adamw">AdamW</option>
            <option value="sgd">SGD + momentum</option>
          </select>
        </div>
        <div class="field">
          <label>MATRIX_LR <span class="val" id="v-mlr">0.0500</span></label>
          <input type="range" id="s-mlr" min="-6.9" max="-1.6" step="0.05" value="-2.996"
                 oninput="update()">
        </div>
        <div class="field">
          <label>SCALAR_LR <span class="val" id="v-slr">0.0500</span></label>
          <input type="range" id="s-slr" min="-6.9" max="-1.6" step="0.05" value="-2.996"
                 oninput="update()">
        </div>
        <div class="field">
          <label>TIED_EMBED_LR <span class="val" id="v-elr">0.0500</span></label>
          <input type="range" id="s-elr" min="-6.9" max="-1.6" step="0.05" value="-2.996"
                 oninput="update()">
        </div>
        <div class="field">
          <label>MUON_MOMENTUM <span class="val" id="v-mom">0.950</span></label>
          <input type="range" id="s-mom" min="0.85" max="0.99" step="0.005" value="0.95"
                 oninput="update()">
        </div>
        <div class="field">
          <label>Weight decay <span class="val" id="v-wd">0.000</span></label>
          <input type="range" id="s-wd" min="0" max="0.1" step="0.005" value="0"
                 oninput="update()">
        </div>
      </div>

    </div><!-- /LEFT -->

    <!-- RIGHT: live metrics + command -->
    <div>

      <h2>Live Metrics</h2>
      <div class="metrics-grid">
        <div class="metric">
          <div class="label">Parameters</div>
          <div class="value" id="m-params">—</div>
        </div>
        <div class="metric">
          <div class="label">FP32 size</div>
          <div class="value" id="m-fp32">—</div>
          <div class="sub-value">uncompressed</div>
        </div>
        <div class="metric">
          <div class="label">INT8+zlib size</div>
          <div class="value" id="m-int8">—</div>
          <div class="sub-value">estimated submission</div>
        </div>
        <div class="metric">
          <div class="label">Fits 16 MB cap?</div>
          <div class="value" id="m-fits16">—</div>
        </div>
        <div class="metric">
          <div class="label">Tokens / sec (1×H100)</div>
          <div class="value" id="m-tps">—</div>
        </div>
        <div class="metric">
          <div class="label">Time · 1×H100</div>
          <div class="value" id="m-t1">—</div>
        </div>
        <div class="metric">
          <div class="label">Time · 8×H100</div>
          <div class="value" id="m-t8">—</div>
        </div>
        <div class="metric">
          <div class="label">Fits 10 min wall?</div>
          <div class="value" id="m-fits10">—</div>
        </div>
        <div class="metric" style="grid-column:1/-1">
          <div class="label">Est. final BPB  (power law from smoke_005)</div>
          <div class="value" id="m-bpb">—</div>
          <div class="sub-value" id="m-bpb-sub">BPB ≈ 3.8 · tokens^(−0.12) + 1.0</div>
        </div>
      </div>

      <h2>Ready-to-Run Command</h2>
      <div class="card">
        <div class="inline-row" style="margin-bottom:10px">
          <label>RUN_ID</label>
          <input type="text" id="s-runid" value="experiment_001"
                 style="width:180px" oninput="update()">
          <label>nproc</label>
          <select id="s-nproc" onchange="update()" style="width:70px">
            <option value="1" selected>1</option>
            <option value="2">2</option>
            <option value="4">4</option>
            <option value="8">8</option>
          </select>
        </div>
        <div style="position:relative">
          <pre class="cmd-box" id="cmd-out"></pre>
          <button class="copy-btn" onclick="copyCmd()">Copy</button>
        </div>
        <p class="info" id="optim-note" style="margin-top:8px"></p>
      </div>

    </div><!-- /RIGHT -->

  </div><!-- /cols -->
</div><!-- /tab-arch -->


<!-- ═══════════════════════════════════════════════════════════════
     TAB 2 — Bayesian Optimizer Dashboard
     ═══════════════════════════════════════════════════════════════ -->
<div id="tab-bo" class="tab-panel">
  <div class="cols">

    <!-- LEFT: search space range bars -->
    <div>
      <h2>Search Space  <span style="font-weight:400;text-transform:none;letter-spacing:0">(editable bounds)</span></h2>
      <div class="card" id="bo-space-card">

        <div class="range-bar-row" id="bor-num_layers">
          <div class="name">num_layers <span>int · 6 – 16</span></div>
          <div style="display:flex;gap:8px;align-items:center;margin-bottom:4px">
            <input type="number" id="bo-num_layers-lo" value="6"  min="1" max="24"
                   style="width:60px" onchange="renderBoSpace()">
            <div class="range-bar-track" style="flex:1">
              <div class="range-bar-fill" id="bof-num_layers" style="width:100%"></div>
            </div>
            <input type="number" id="bo-num_layers-hi" value="16" min="1" max="24"
                   style="width:60px" onchange="renderBoSpace()">
          </div>
        </div>

        <div class="range-bar-row">
          <div class="name">mlp_mult <span>categorical · [2, 3, 4]</span></div>
          <div style="display:flex;gap:6px;margin-top:4px">
            <label style="font-size:12px;color:var(--dim)">
              <input type="checkbox" id="bo-mlp-2" checked onchange="renderBoCmd()"> 2
            </label>
            <label style="font-size:12px;color:var(--dim)">
              <input type="checkbox" id="bo-mlp-3" checked onchange="renderBoCmd()"> 3
            </label>
            <label style="font-size:12px;color:var(--dim)">
              <input type="checkbox" id="bo-mlp-4" checked onchange="renderBoCmd()"> 4
            </label>
          </div>
        </div>

        <div class="range-bar-row">
          <div class="name">matrix_lr <span>log-float · 0.01 – 0.12</span></div>
          <div style="display:flex;gap:8px;align-items:center;margin-bottom:4px">
            <input type="number" id="bo-matrix_lr-lo" value="0.01" min="0.001" max="0.5" step="0.005"
                   style="width:70px" onchange="renderBoSpace()">
            <div class="range-bar-track" style="flex:1">
              <div class="range-bar-fill" id="bof-matrix_lr" style="width:100%"></div>
            </div>
            <input type="number" id="bo-matrix_lr-hi" value="0.12" min="0.001" max="0.5" step="0.005"
                   style="width:70px" onchange="renderBoSpace()">
          </div>
        </div>

        <div class="range-bar-row">
          <div class="name">scalar_lr <span>log-float · 0.01 – 0.12</span></div>
          <div style="display:flex;gap:8px;align-items:center;margin-bottom:4px">
            <input type="number" id="bo-scalar_lr-lo" value="0.01" min="0.001" max="0.5" step="0.005"
                   style="width:70px" onchange="renderBoSpace()">
            <div class="range-bar-track" style="flex:1">
              <div class="range-bar-fill" id="bof-scalar_lr" style="width:100%"></div>
            </div>
            <input type="number" id="bo-scalar_lr-hi" value="0.12" min="0.001" max="0.5" step="0.005"
                   style="width:70px" onchange="renderBoSpace()">
          </div>
        </div>

        <div class="range-bar-row">
          <div class="name">tied_embed_lr <span>log-float · 0.01 – 0.15</span></div>
          <div style="display:flex;gap:8px;align-items:center;margin-bottom:4px">
            <input type="number" id="bo-tied_embed_lr-lo" value="0.01" min="0.001" max="0.5" step="0.005"
                   style="width:70px" onchange="renderBoSpace()">
            <div class="range-bar-track" style="flex:1">
              <div class="range-bar-fill" id="bof-tied_embed_lr" style="width:100%"></div>
            </div>
            <input type="number" id="bo-tied_embed_lr-hi" value="0.15" min="0.001" max="0.5" step="0.005"
                   style="width:70px" onchange="renderBoSpace()">
          </div>
        </div>

        <div class="range-bar-row">
          <div class="name">muon_momentum <span>float · 0.90 – 0.98</span></div>
          <div style="display:flex;gap:8px;align-items:center;margin-bottom:4px">
            <input type="number" id="bo-muon_momentum-lo" value="0.90" min="0.5" max="0.999" step="0.01"
                   style="width:70px" onchange="renderBoSpace()">
            <div class="range-bar-track" style="flex:1">
              <div class="range-bar-fill" id="bof-muon_momentum" style="width:100%"></div>
            </div>
            <input type="number" id="bo-muon_momentum-hi" value="0.98" min="0.5" max="0.999" step="0.01"
                   style="width:70px" onchange="renderBoSpace()">
          </div>
        </div>

        <div class="range-bar-row">
          <div class="name">qk_gain_init <span>float · 0.5 – 3.0</span></div>
          <div style="display:flex;gap:8px;align-items:center;margin-bottom:4px">
            <input type="number" id="bo-qk_gain_init-lo" value="0.5" min="0.1" max="5.0" step="0.1"
                   style="width:70px" onchange="renderBoSpace()">
            <div class="range-bar-track" style="flex:1">
              <div class="range-bar-fill" id="bof-qk_gain_init" style="width:100%"></div>
            </div>
            <input type="number" id="bo-qk_gain_init-hi" value="3.0" min="0.1" max="5.0" step="0.1"
                   style="width:70px" onchange="renderBoSpace()">
          </div>
        </div>

        <div class="range-bar-row">
          <div class="name">warmdown_iters <span>int · 200 – 2000 step 200</span></div>
          <div style="display:flex;gap:8px;align-items:center;margin-bottom:4px">
            <input type="number" id="bo-warmdown_iters-lo" value="200"  min="100" max="3000" step="200"
                   style="width:70px" onchange="renderBoSpace()">
            <div class="range-bar-track" style="flex:1">
              <div class="range-bar-fill" id="bof-warmdown_iters" style="width:100%"></div>
            </div>
            <input type="number" id="bo-warmdown_iters-hi" value="2000" min="100" max="3000" step="200"
                   style="width:70px" onchange="renderBoSpace()">
          </div>
        </div>

      </div><!-- /card -->

      <h2>BO Run Planner</h2>
      <div class="card">
        <div class="field">
          <label>n_trials <span class="val" id="v-ntrials">15</span></label>
          <input type="range" id="s-ntrials" min="5" max="50" step="1" value="15"
                 oninput="updateBo()">
        </div>
        <div class="field">
          <label>iters_per_trial <span class="val" id="v-boiters">200</span></label>
          <input type="range" id="s-boiters" min="50" max="500" step="50" value="200"
                 oninput="updateBo()">
        </div>
        <div class="metrics-grid" style="margin-top:12px">
          <div class="metric">
            <div class="label">Est. GPU-hours</div>
            <div class="value" id="bo-m-hours">—</div>
          </div>
          <div class="metric">
            <div class="label">Est. cost @ $2/hr H100</div>
            <div class="value" id="bo-m-cost">—</div>
          </div>
        </div>
        <h3 style="margin-top:4px">Command</h3>
        <div style="position:relative">
          <pre class="cmd-box" id="bo-cmd-out"></pre>
          <button class="copy-btn" onclick="copyBoCmd()">Copy</button>
        </div>
      </div>

    </div><!-- /LEFT -->

    <!-- RIGHT: arch configs table -->
    <div>
      <h2>ARCH_CONFIGS  <span style="font-weight:400;text-transform:none;letter-spacing:0">(13 pre-validated entries)</span></h2>
      <p class="section-note">Param estimate at 9 layers, MLP_MULT=4. INT8 size ≈ params × 0.75 B.</p>
      <div class="card" style="overflow-x:auto">
        <table>
          <thead>
            <tr>
              <th>#</th><th>dim</th><th>heads</th><th>kv</th>
              <th>params</th><th>INT8 MB</th><th>≤16MB</th>
            </tr>
          </thead>
          <tbody>
            __ARCH_ROWS__
          </tbody>
        </table>
      </div>
    </div><!-- /RIGHT -->

  </div><!-- /cols -->
</div><!-- /tab-bo -->


<!-- ═══════════════════════════════════════════════════════════════
     TAB 3 — Experiment Planner
     ═══════════════════════════════════════════════════════════════ -->
<div id="tab-exp" class="tab-panel">
  <div class="cols">

    <div>
      <h2>Experiment Card</h2>
      <div class="card">

        <div class="field">
          <label>Experiment name</label>
          <input type="text" id="exp-name" placeholder="e.g. wider_mlp_test" oninput="expChanged()">
        </div>

        <div class="field">
          <label>Baseline run</label>
          <select id="exp-baseline" onchange="expChanged()">
            __RUN_OPTIONS__
          </select>
        </div>

        <div class="field">
          <label>Hypothesis — what do you think will happen and why?</label>
          <textarea id="exp-hyp" rows="3"
            placeholder="e.g. Wider MLP (mult=4→6) should give more representational capacity per layer, reducing BPB by ~0.02 at the cost of ~20% more params..."
            oninput="expChanged()"></textarea>
        </div>

        <div class="field">
          <label>ONE thing changing</label>
          <input type="text" id="exp-change"
            placeholder="e.g. MLP_MULT 4→6  (one change only!)"
            oninput="expChanged()">
        </div>

        <div class="field">
          <label>Expected BPB direction</label>
          <select id="exp-dir" onchange="expChanged()">
            <option value="improve">Should improve (lower BPB)</option>
            <option value="hurt">Should hurt (higher BPB)</option>
            <option value="unclear">Unclear — exploring</option>
          </select>
        </div>

        <div class="field" style="display:flex;align-items:center;gap:10px">
          <input type="checkbox" id="exp-t4" onchange="expChanged()">
          <label for="exp-t4" style="color:var(--text);font-size:13px">
            T4 smoke test first? (Colab — ~30–60× slower, cheap sanity check)
          </label>
        </div>

        <div class="field">
          <label>Iterations for this experiment</label>
          <input type="number" id="exp-iters" value="1000" min="100" max="5000" step="100"
                 oninput="expChanged()">
        </div>

        <div id="exp-cost-note" class="info"></div>
        <div id="exp-multi-warn" class="warn" style="display:none">
          ⚠ Multiple variables detected in the "one thing" field. Ablations should change exactly one thing.
        </div>

        <button class="btn" onclick="generateExpCard()">Generate experiment card</button>

        <pre class="exp-card-out" id="exp-card-out"></pre>
      </div>
    </div>

    <div>
      <h2>Ablation Discipline</h2>
      <div class="card">
        <h3>Rules</h3>
        <ul style="color:var(--dim);font-size:13px;padding-left:18px;line-height:2">
          <li>Change exactly <strong style="color:var(--white)">one</strong> thing per run</li>
          <li>Always record the baseline BPB before changing anything</li>
          <li>Run a T4 smoke test first (200 iters, ~5 min) before committing H100 budget</li>
          <li>Write the hypothesis <em style="color:var(--cyan)">before</em> running, not after</li>
          <li>If the result surprises you, it's more valuable than if it doesn't</li>
        </ul>

        <h3 style="margin-top:18px">Context</h3>
        <table>
          <tbody>
            <tr><td style="color:var(--dim)">Current best BPB</td><td style="color:var(--green);font-weight:700">1.1194</td></tr>
            <tr><td style="color:var(--dim)">Our best (smoke_005)</td><td>1.3714</td></tr>
            <tr><td style="color:var(--dim)">Gap to close</td><td style="color:var(--amber)">~0.25 BPB</td></tr>
            <tr><td style="color:var(--dim)">Baseline params</td><td>17M</td></tr>
            <tr><td style="color:var(--dim)">Baseline config</td><td>9L · 512d · 8H · 4KV</td></tr>
            <tr><td style="color:var(--dim)">Budget per run</td><td>~10 min · 8×H100 ≈ $2.67</td></tr>
            <tr><td style="color:var(--dim)">smoke_005 converged?</td><td style="color:var(--amber)">No — still improving at step 1000</td></tr>
          </tbody>
        </table>

        <h3 style="margin-top:18px">Quick cost estimator</h3>
        <div class="metrics-grid" style="margin-top:8px">
          <div class="metric">
            <div class="label">1×H100 cost / 1000 iters</div>
            <div class="value">~$0.33</div>
            <div class="sub-value">@ $2/hr, ~10min/1k iters baseline</div>
          </div>
          <div class="metric">
            <div class="label">8×H100 cost / 1000 iters</div>
            <div class="value">~$2.67</div>
            <div class="sub-value">full training budget</div>
          </div>
        </div>
      </div>
    </div>

  </div><!-- /cols -->
</div><!-- /tab-exp -->


<script>
// ─────────────────────────────────────────────────────────────────
// Tab switching
// ─────────────────────────────────────────────────────────────────
function switchTab(id) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  event.target.classList.add('active');
}

// ─────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────
function fmtM(n) {
  if (n >= 1e9) return (n/1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n/1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'k';
  return n.toString();
}

function fmtMB(bytes) {
  return (bytes / 1e6).toFixed(1) + ' MB';
}

function fmtMin(min) {
  if (min < 1) return (min * 60).toFixed(0) + 's';
  if (min >= 60) return (min / 60).toFixed(1) + 'h';
  return min.toFixed(1) + 'min';
}

// ─────────────────────────────────────────────────────────────────
// Param estimation
// ─────────────────────────────────────────────────────────────────
function estimateParams(dim, layers, heads, kvHeads, mlpMult) {
  const vocab = 1024;
  const headDim = dim / heads;
  // Q: dim×(heads×headDim)=dim²  K: dim×(kv×headDim)  V: same  O: dim²
  const attn = dim * dim + 2 * dim * kvHeads * headDim + dim * dim;
  // MLP: two matmuls of dim→mlp_mult*dim→dim
  const mlp = 2 * dim * (mlpMult * dim);
  // LayerNorm: 2 params * dim * 2 norms per layer (small but included)
  const ln = 4 * dim;
  return Math.round(layers * (attn + mlp + ln) + vocab * dim);
}

// tokens/sec on 1×H100 (empirical calibration vs smoke_005 baseline)
function estimateTPS(dim, layers) {
  return 500000 / Math.pow(dim / 512, 2) / (layers / 9);
}

// Power law BPB from smoke_005 fit: L(t) = 3.8 * t^(-0.12) + 1.0
function estimateBPB(iters, batchTokens) {
  const t = iters * batchTokens;
  return 3.8 * Math.pow(t, -0.12) + 1.0;
}

// ─────────────────────────────────────────────────────────────────
// KV heads sync (must be divisor of num_heads)
// ─────────────────────────────────────────────────────────────────
function syncKvHeads() {
  const heads = parseInt(document.getElementById('s-heads').value);
  const sel = document.getElementById('s-kvheads');
  const prev = parseInt(sel.value) || heads;
  sel.innerHTML = '';
  for (let kv = 1; kv <= heads; kv++) {
    if (heads % kv === 0) {
      const opt = document.createElement('option');
      opt.value = kv;
      opt.textContent = kv;
      if (kv === prev && heads % prev === 0) opt.selected = true;
      sel.appendChild(opt);
    }
  }
  // default to half if previous invalid
  if (!sel.value || heads % parseInt(sel.value) !== 0) {
    sel.value = Math.floor(heads / 2) || 1;
  }
  document.getElementById('v-heads').textContent = heads;
  document.getElementById('v-kvheads').textContent = sel.value;
}

// ─────────────────────────────────────────────────────────────────
// Activation SVG toggle
// ─────────────────────────────────────────────────────────────────
const ACT_NAMES = ['gelu','swiglu','tanh','relu','leakyrelu'];
function updateActSvg() {
  const chosen = document.getElementById('s-act').value;
  ACT_NAMES.forEach(n => {
    document.getElementById('svg-' + n).style.display = (n === chosen) ? '' : 'none';
  });
}

// ─────────────────────────────────────────────────────────────────
// Main update (Tab 1)
// ─────────────────────────────────────────────────────────────────
function update() {
  const dim      = parseInt(document.getElementById('s-dim').value);
  const layers   = parseInt(document.getElementById('s-layers').value);
  const heads    = parseInt(document.getElementById('s-heads').value);
  const kvHeads  = parseInt(document.getElementById('s-kvheads').value);
  const mlpMult  = parseInt(document.getElementById('s-mlp').value);
  const iters    = parseInt(document.getElementById('s-iters').value);
  const batch    = parseInt(document.getElementById('s-batch').value);
  const seqLen   = parseInt(document.getElementById('s-seqlen').value);
  const warmup   = parseInt(document.getElementById('s-warmup').value);
  const warmdown = parseInt(document.getElementById('s-warmdown').value);
  const mlrLog   = parseFloat(document.getElementById('s-mlr').value);
  const slrLog   = parseFloat(document.getElementById('s-slr').value);
  const elrLog   = parseFloat(document.getElementById('s-elr').value);
  const mom      = parseFloat(document.getElementById('s-mom').value);
  const wd       = parseFloat(document.getElementById('s-wd').value);
  const optim    = document.getElementById('s-optim').value;
  const act      = document.getElementById('s-act').value;
  const runId    = document.getElementById('s-runid').value || 'experiment_001';
  const nproc    = parseInt(document.getElementById('s-nproc').value);

  const mlr = Math.exp(mlrLog);
  const slr = Math.exp(slrLog);
  const elr = Math.exp(elrLog);

  // Display slider values
  document.getElementById('v-dim').textContent     = dim;
  document.getElementById('v-layers').textContent  = layers;
  document.getElementById('v-kvheads').textContent = kvHeads;
  document.getElementById('v-mlp').textContent     = mlpMult;
  document.getElementById('v-iters').textContent   = iters;
  document.getElementById('v-batch').textContent   = batch.toLocaleString();
  document.getElementById('v-seqlen').textContent  = seqLen;
  document.getElementById('v-warmup').textContent  = warmup;
  document.getElementById('v-warmdown').textContent = warmdown;
  document.getElementById('v-mlr').textContent     = mlr.toFixed(4);
  document.getElementById('v-slr').textContent     = slr.toFixed(4);
  document.getElementById('v-elr').textContent     = elr.toFixed(4);
  document.getElementById('v-mom').textContent     = mom.toFixed(3);
  document.getElementById('v-wd').textContent      = wd.toFixed(3);

  // Metrics
  const params    = estimateParams(dim, layers, heads, kvHeads, mlpMult);
  const fp32B     = params * 4;
  const int8B     = params * 1 * 0.75;
  const fits16    = int8B <= 16e6;
  const tps       = estimateTPS(dim, layers);
  const totalTok  = iters * batch;
  const t1min     = totalTok / tps / 60;
  const t8min     = t1min / 8;
  const fits10    = t8min <= 10.0;
  const bpb       = estimateBPB(iters, batch);

  document.getElementById('m-params').textContent = fmtM(params);
  document.getElementById('m-fp32').textContent   = fmtMB(fp32B);
  document.getElementById('m-int8').textContent   = fmtMB(int8B);

  const fits16El = document.getElementById('m-fits16');
  fits16El.textContent = fits16 ? 'YES ✓' : 'NO ✗';
  fits16El.className = 'value ' + (fits16 ? 'green' : 'red');

  document.getElementById('m-tps').textContent = fmtM(Math.round(tps));
  document.getElementById('m-t1').textContent  = fmtMin(t1min);
  document.getElementById('m-t8').textContent  = fmtMin(t8min);

  const fits10El = document.getElementById('m-fits10');
  fits10El.textContent = fits10 ? 'YES ✓' : 'NO ✗';
  fits10El.className = 'value ' + (fits10 ? 'green' : (t8min <= 12 ? 'amber' : 'red'));

  const bpbEl = document.getElementById('m-bpb');
  bpbEl.textContent = bpb.toFixed(4);
  bpbEl.className = 'value ' + (bpb < 1.25 ? 'green' : bpb < 1.5 ? 'amber' : 'red');
  document.getElementById('m-bpb-sub').textContent =
    `tokens seen: ${(totalTok/1e6).toFixed(1)}M  ·  power law smoke_005 fit`;

  // Command
  const cmd = [
    `RUN_ID=${runId}`,
    `MODEL_DIM=${dim}`,
    `NUM_LAYERS=${layers}`,
    `NUM_HEADS=${heads}`,
    `NUM_KV_HEADS=${kvHeads}`,
    `MLP_MULT=${mlpMult}`,
    `ITERATIONS=${iters}`,
    `TRAIN_BATCH_TOKENS=${batch}`,
    `TRAIN_SEQ_LEN=${seqLen}`,
    `WARMUP_STEPS=${warmup}`,
    `WARMDOWN_ITERS=${warmdown}`,
    `MATRIX_LR=${mlr.toFixed(4)}`,
    `SCALAR_LR=${slr.toFixed(4)}`,
    `TIED_EMBED_LR=${elr.toFixed(4)}`,
    `MUON_MOMENTUM=${mom.toFixed(3)}`,
    (wd > 0 ? `WEIGHT_DECAY=${wd.toFixed(3)}` : null),
    `ACTIVATION=${act}`,
    `torchrun --standalone --nproc_per_node=${nproc} train_gpt.py`,
  ].filter(Boolean).join(' \\\n  ');
  document.getElementById('cmd-out').textContent = cmd;

  // Optimizer note
  const noteEl = document.getElementById('optim-note');
  if (optim !== 'muon') {
    noteEl.textContent =
      `⚠ ${optim.toUpperCase()} selected — train_gpt.py uses Muon by default. Switching requires code changes.`;
  } else {
    noteEl.textContent = '';
  }
}

function copyCmd() {
  navigator.clipboard.writeText(document.getElementById('cmd-out').textContent);
}

// ─────────────────────────────────────────────────────────────────
// Tab 2 — BO Dashboard
// ─────────────────────────────────────────────────────────────────
function renderBoSpace() {
  // Just keeps the bars full-width since min/max are user-edited — no best known
  const bars = ['num_layers','matrix_lr','scalar_lr','tied_embed_lr',
                 'muon_momentum','qk_gain_init','warmdown_iters'];
  bars.forEach(p => {
    const bar = document.getElementById('bof-' + p);
    if (bar) bar.style.width = '100%';
  });
  renderBoCmd();
}

function renderBoCmd() {
  const nTrials = parseInt(document.getElementById('s-ntrials').value);
  const boIters = parseInt(document.getElementById('s-boiters').value);

  document.getElementById('v-ntrials').textContent = nTrials;
  document.getElementById('v-boiters').textContent = boIters;

  // Cost estimate: ~100 iters/min on 1×H100 at baseline
  const trialMin = boIters / 100;
  const totalMin = nTrials * trialMin;
  const totalHrs = totalMin / 60;
  const cost = totalHrs * 2;

  document.getElementById('bo-m-hours').textContent = totalHrs.toFixed(2) + 'h';
  document.getElementById('bo-m-cost').textContent  = '$' + cost.toFixed(2);

  const cmd = `python analysis/bo_tune.py --n-trials ${nTrials} --iters ${boIters}`;
  document.getElementById('bo-cmd-out').textContent = cmd;
}

function updateBo() { renderBoCmd(); }

function copyBoCmd() {
  navigator.clipboard.writeText(document.getElementById('bo-cmd-out').textContent);
}

// ─────────────────────────────────────────────────────────────────
// Tab 3 — Experiment Planner
// ─────────────────────────────────────────────────────────────────
function expChanged() {
  const iters = parseInt(document.getElementById('exp-iters').value) || 1000;
  // cost: 1×H100 ~10min/1000iters @ $2/hr → $0.33/1k iters; 8×H100 → $2.67/1k iters
  const min1  = iters / 100;
  const cost1 = (min1 / 60) * 2;
  const cost8 = cost1 * 8;
  document.getElementById('exp-cost-note').textContent =
    `Est. cost: 1×H100 ~${min1.toFixed(0)}min ($${cost1.toFixed(2)})  ·  8×H100 ~${(min1/8).toFixed(0)}min ($${cost8.toFixed(2)})`;

  // Detect multiple variables changed
  const change = document.getElementById('exp-change').value;
  const warnEl = document.getElementById('exp-multi-warn');
  const arrows  = (change.match(/→|->|=/g) || []).length;
  const commas  = (change.match(/,/g) || []).length;
  const multi   = arrows > 1 || commas >= 2;
  warnEl.style.display = multi ? '' : 'none';
}

function generateExpCard() {
  const name     = document.getElementById('exp-name').value     || '(unnamed)';
  const baseline = document.getElementById('exp-baseline').value || '—';
  const hyp      = document.getElementById('exp-hyp').value      || '(no hypothesis written)';
  const change   = document.getElementById('exp-change').value   || '(not specified)';
  const dir      = document.getElementById('exp-dir').value;
  const t4       = document.getElementById('exp-t4').checked;
  const iters    = parseInt(document.getElementById('exp-iters').value) || 1000;

  const dirStr = { improve: 'Should IMPROVE (lower BPB)',
                   hurt:    'Should HURT (higher BPB)',
                   unclear: 'Unclear — exploring' }[dir];
  const t4Str = t4 ? 'YES ✓' : 'NO — skipping (budget risk)';

  const min8  = iters / 8 / 100;
  const cost8 = (min8 / 60) * 2 * 8;

  const now = new Date().toISOString().slice(0, 10);

  const card = [
    '═'.repeat(60),
    `EXPERIMENT: ${name}`,
    `Date:       ${now}`,
    `Baseline:   ${baseline}`,
    '─'.repeat(60),
    `Hypothesis:\n  ${hyp}`,
    '',
    `ONE change: ${change}`,
    `Direction:  ${dirStr}`,
    `T4 smoke:   ${t4Str}`,
    `Iterations: ${iters}`,
    `Est. cost:  ~$${cost8.toFixed(2)} (8×H100)`,
    '═'.repeat(60),
  ].join('\n');

  const outEl = document.getElementById('exp-card-out');
  outEl.textContent = card;
  outEl.style.display = '';
}

// ─────────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────────
syncKvHeads();
update();
renderBoSpace();
renderBoCmd();
expChanged();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_simulator(output: Path) -> None:
    svgs = {fn: _act_svg(fn) for fn in ["tanh", "relu", "gelu", "swiglu", "leakyrelu"]}
    arch_rows = _arch_table_rows()
    run_options = "\n".join(
        f'            <option value="{r}">{r}</option>' for r in KNOWN_RUNS
    )

    html = _TEMPLATE
    html = html.replace("__BG__",         BG)
    html = html.replace("__BG_CARD__",    BG_CARD)
    html = html.replace("__GRID__",       GRID)
    html = html.replace("__BORDER__",     BORDER)
    html = html.replace("__TEXT__",       TEXT)
    html = html.replace("__TEXT_DIM__",   TEXT_DIM)
    html = html.replace("__NEON_GREEN__",  NEON_GREEN)
    html = html.replace("__NEON_CYAN__",   NEON_CYAN)
    html = html.replace("__NEON_RED__",    NEON_RED)
    html = html.replace("__NEON_AMBER__",  NEON_AMBER)
    html = html.replace("__NEON_WHITE__",  NEON_WHITE)
    html = html.replace("__NEON_BLUE__",   NEON_BLUE)
    html = html.replace("__NEON_PURPLE__", NEON_PURPLE)
    html = html.replace("__SVG_TANH__",       svgs["tanh"])
    html = html.replace("__SVG_RELU__",       svgs["relu"])
    html = html.replace("__SVG_GELU__",       svgs["gelu"])
    html = html.replace("__SVG_SWIGLU__",     svgs["swiglu"])
    html = html.replace("__SVG_LEAKYRELU__",  svgs["leakyrelu"])
    html = html.replace("__ARCH_ROWS__",   arch_rows)
    html = html.replace("__RUN_OPTIONS__", run_options)

    output.write_text(html, encoding="utf-8")
    print(f"simulator saved: {output}")


if __name__ == "__main__":
    out = Path(__file__).parent / "simulator.html"
    generate_simulator(out)

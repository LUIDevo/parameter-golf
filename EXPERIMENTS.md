# Fixed 1,000-step experiments

The root CUDA trainer defaults to 1,000 optimizer steps, a 100-step linear cooldown,
and no wallclock cutoff. The architecture is still the nine-layer ReLU² baseline.
Block weights use per-row int6 QAT and packed int6 export. The same quantizer,
including FP16 scale rounding, runs during training and export. Embeddings and
output heads stay FP16 on export and do not receive QAT.

EMA is off. Attention is fully causal and uses PyTorch SDPA, so the default run
does not require the separate `flash-attn` package. `ATTENTION_WINDOW` enables
local attention only when explicitly set to a nonnegative value. The old
`SWA_WINDOW` variable remains an alias; unset it when running controls.

## Matched controls

Use the same GPU, data, seed, batch size, architecture, and evaluation settings.
For these fixed-step comparisons, keep the wallclock cutoff disabled. If you set
one, training cools down toward whichever limit comes first.

```bash
export ITERATIONS=1000 WARMDOWN_ITERS=100 MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=250 SEED=1337 EMA_ENABLED=0 ATTENTION_WINDOW=-1 EVAL_STRIDE=0

# Baseline-style training, int8 block export.
RUN_ID=control_int8 QAT_ENABLED=0 QUANT_BITS=8 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py

# Isolate the cost of int6 export.
RUN_ID=control_int6 QAT_ENABLED=0 QUANT_BITS=6 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py

# Measure whether matched QAT recovers that cost.
RUN_ID=matched_int6_qat QAT_ENABLED=1 QUANT_BITS=6 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py
```

These are matched controls for the corrected trainer, not exact reproductions of
old logs: they share the corrected cooldown, adaptive quantizer, FP16 embeddings,
and complete validation-token accounting. Compare final roundtrip BPB and total
artifact bytes. Repeat promising comparisons across seeds before claiming a win.
`DEV_MODE=1` is only for debugging; it still forces 100 iterations.

## EMA diagnostics

`EMA_ENABLED=1` adds a late EMA without changing periodic validation or replacing
the primary artifact. Its defaults are `EMA_START_STEP=900` and `EMA_DECAY=0.9`
for a 1,000-step run with a 100-step cooldown. It initializes from the trained
weights at that step, not from initialization or compilation warmup.

At the end, logs contain clearly labeled scores for raw and EMA weights before
and after export. Raw weights are saved as `final_model.pt` and
`final_model.int6.ptz`. EMA weights are saved separately as `final_model.ema.pt`
and `final_model.ema.int6.ptz`. `QUANT_BITS=8` changes the artifact suffix to
`.int8.ptz`. If a time cutoff occurs before EMA starts, only raw weights are saved.

Each artifact logs its actual code-plus-model byte count and whether it fits the
decimal 16,000,000-byte cap. Run logs include the Git commit, trainer SHA-256,
actual model shape, feature settings, and training LR multipliers.

```bash
RUN_ID=ema_diagnostic EMA_ENABLED=1 QAT_ENABLED=1 QUANT_BITS=6 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py
python analysis/analyze.py detail ema_diagnostic
```

## Optional overlapping-window evaluation

`EVAL_STRIDE=64` adds a final overlapping-window score on each exported variant.
It does not shorten attention or change training. Each target is scored once,
including the first window and final partial window. Standard non-overlapping
scores remain the primary result for comparison; the additional score is labeled
`final_sliding_window`. This costs extra evaluation time, not training steps. Time it on the target hardware
before assuming it fits the challenge's separate evaluation budget.

Validation now includes the tail previously dropped by sequence-length rounding,
so use fresh baseline measurements rather than expecting exact equality with
historical BPB values.

## Checks

With the training dependencies installed, run CPU regression checks:

```bash
python -B -m unittest -v test_train_gpt
```

The checks cover bit packing, QAT/export equivalence, embedding preservation,
STE gradients, cooldowns, EMA snapshots, validation-token/byte accounting,
full-graph capture with the eager backend, CPU training, and log compatibility.
They do not establish GPU throughput, CUDA/FlashAttention compatibility, or a
quality improvement over the baseline. The optional Apple MLX trainer remains
int8-only; its incorrect int6 metadata and undefined variable were removed.

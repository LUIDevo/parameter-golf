"""Parse train_gpt.py log files into structured run data."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StepRecord:
    step: int
    total_steps: int
    train_loss: float | None = None
    val_loss: float | None = None
    val_bpb: float | None = None
    train_time_ms: float = 0.0
    step_avg_ms: float = 0.0


@dataclass
class RunResult:
    run_id: str
    log_path: str

    # Hyperparameters
    num_heads: int | None = None
    num_kv_heads: int | None = None
    model_params: int | None = None
    train_batch_tokens: int | None = None
    train_seq_len: int | None = None
    iterations: int | None = None
    warmup_steps: int | None = None
    max_wallclock_seconds: float | None = None
    seed: int | None = None
    embed_lr: float | None = None
    head_lr: float | None = None
    matrix_lr: float | None = None
    scalar_lr: float | None = None
    tie_embeddings: bool | None = None
    train_shards: int | None = None
    world_size: int | None = None
    grad_accum_steps: int | None = None

    # Per-step metrics
    steps: list[StepRecord] = field(default_factory=list)

    # Final results
    final_val_loss: float | None = None
    final_val_bpb: float | None = None
    final_train_time_ms: float | None = None
    peak_memory_mib: int | None = None
    reserved_memory_mib: int | None = None
    model_bytes: int | None = None
    int8_bytes: int | None = None
    submission_bytes: int | None = None
    int8_val_loss: float | None = None
    int8_val_bpb: float | None = None


STEP_RE = re.compile(
    r"step:(\d+)/(\d+)\s+"
    r"(?:train_loss:([\d.]+)\s+)?"
    r"(?:val_loss:([\d.]+)\s+val_bpb:([\d.]+)\s+)?"
    r"train_time:([\d.]+)ms\s+step_avg:([\d.]+)ms"
)


def parse_log(path: Path) -> RunResult:
    run_id = path.stem
    result = RunResult(run_id=run_id, log_path=str(path))
    text = path.read_text(encoding="utf-8", errors="replace")

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Hyperparameters
        if m := re.search(r"train_shards:(\d+)", line):
            result.train_shards = int(m.group(1))
        if m := re.search(r"model_params:(\d+)", line):
            result.model_params = int(m.group(1))
        if m := re.search(r"world_size:(\d+)\s+grad_accum_steps:(\d+)", line):
            result.world_size = int(m.group(1))
            result.grad_accum_steps = int(m.group(2))
        if m := re.search(r"num_heads:(\d+)\s+num_kv_heads:(\d+)", line):
            result.num_heads = int(m.group(1))
            result.num_kv_heads = int(m.group(2))
        if m := re.search(r"tie_embeddings:(\w+)", line):
            result.tie_embeddings = m.group(1) == "True"
        if m := re.search(r"embed_lr:([\d.]+)", line):
            result.embed_lr = float(m.group(1))
        if m := re.search(r"head_lr:([\d.]+)", line):
            result.head_lr = float(m.group(1))
        if m := re.search(r"matrix_lr:([\d.]+)", line):
            result.matrix_lr = float(m.group(1))
        if m := re.search(r"scalar_lr:([\d.]+)", line):
            result.scalar_lr = float(m.group(1))
        if m := re.search(r"train_batch_tokens:(\d+)\s+train_seq_len:(\d+)\s+iterations:(\d+)\s+warmup_steps:(\d+)\s+max_wallclock_seconds:([\d.]+)", line):
            result.train_batch_tokens = int(m.group(1))
            result.train_seq_len = int(m.group(2))
            result.iterations = int(m.group(3))
            result.warmup_steps = int(m.group(4))
            result.max_wallclock_seconds = float(m.group(5))
        if m := re.search(r"^seed:(\d+)$", line):
            result.seed = int(m.group(1))

        # Step records
        if m := STEP_RE.search(line):
            rec = StepRecord(
                step=int(m.group(1)),
                total_steps=int(m.group(2)),
                train_loss=float(m.group(3)) if m.group(3) else None,
                val_loss=float(m.group(4)) if m.group(4) else None,
                val_bpb=float(m.group(5)) if m.group(5) else None,
                train_time_ms=float(m.group(6)),
                step_avg_ms=float(m.group(7)),
            )
            result.steps.append(rec)

        # Peak memory
        if m := re.search(r"peak memory allocated:\s*(\d+)\s*MiB\s+reserved:\s*(\d+)\s*MiB", line):
            result.peak_memory_mib = int(m.group(1))
            result.reserved_memory_mib = int(m.group(2))

        # Serialization
        if m := re.search(r"^Serialized model:\s*(\d+)\s*bytes", line):
            result.model_bytes = int(m.group(1))
        if m := re.search(r"^Serialized model int8\+zlib:\s*(\d+)\s*bytes", line):
            result.int8_bytes = int(m.group(1))
        if m := re.search(r"^Total submission size int8\+zlib:\s*(\d+)\s*bytes", line):
            result.submission_bytes = int(m.group(1))

        # Final int8 roundtrip
        if m := re.search(r"final_int8_zlib_roundtrip_exact\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)", line):
            result.int8_val_loss = float(m.group(1))
            result.int8_val_bpb = float(m.group(2))

    # Extract final metrics from last validation step
    val_steps = [s for s in result.steps if s.val_bpb is not None]
    if val_steps:
        last = val_steps[-1]
        result.final_val_loss = last.val_loss
        result.final_val_bpb = last.val_bpb
        result.final_train_time_ms = last.train_time_ms

    return result


def parse_all_logs(logs_dir: Path) -> list[RunResult]:
    if not logs_dir.is_dir():
        return []
    results = []
    for f in sorted(logs_dir.glob("*.txt")):
        try:
            results.append(parse_log(f))
        except Exception as e:
            print(f"warning: failed to parse {f}: {e}")
    return results

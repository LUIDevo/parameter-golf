"""CPU regression checks. Run: python -B -m unittest -v test_train_gpt"""

import io
import math
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

import train_gpt as trainer
from analysis.parse_logs import parse_log


torch.set_num_threads(1)


class TrainingRegressionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1337)

    def test_int6_packing_is_lossless_with_padding(self):
        for length in (0, 1, 3, 4, 5, 63, 64, 65, 1001):
            with self.subTest(length=length):
                q = (torch.arange(length) % 64 - 32).to(torch.int8)
                packed = trainer.pack_int6(q)
                self.assertEqual(packed.numel(), 3 * ((length + 3) // 4))
                torch.testing.assert_close(trainer.unpack_int6(packed, length), q)

    def test_qat_matches_serialized_export_and_keeps_gradients(self):
        # Large enough to take the quantized path, with very different row ranges.
        w = torch.randn(257, 257) * torch.linspace(0, 0.1, 257)[:, None]
        state = {"blocks.0.mlp.fc.weight": w, "tok_emb.weight": w.clone(),
                 "lm_head.weight": w.clone(), "blocks.0.attn_scale": torch.ones(257),
                 "counter": torch.tensor(7)}
        for bits in (6, 8):
            with self.subTest(bits=bits):
                obj, _ = trainer.quantize_state_dict_int8(state, bits)
                buf = io.BytesIO()
                torch.save(obj, buf)
                blob = zlib.compress(buf.getvalue())
                loaded = torch.load(io.BytesIO(zlib.decompress(blob)), weights_only=True)
                restored = trainer.dequantize_state_dict_int8(loaded)
                leaf = w.clone().requires_grad_()
                fake = trainer.fake_sym_quant(leaf, bits)
                torch.testing.assert_close(restored["blocks.0.mlp.fc.weight"], fake, rtol=0, atol=1e-8)
                fake.sum().backward()
                torch.testing.assert_close(leaf.grad, torch.ones_like(leaf))
                self.assertTrue(torch.isfinite(fake).all())
                for name in ("tok_emb.weight", "lm_head.weight"):
                    self.assertNotIn(name, obj["quantized"])
                    self.assertEqual(obj["passthrough"][name].dtype, torch.float16)
                    torch.testing.assert_close(restored[name], state[name].half().float())
                torch.testing.assert_close(restored["counter"], state["counter"])

    def test_quantizer_preserves_small_row_resolution(self):
        w = torch.linspace(-0.1, 0.1, 10001)[None, :]
        q, scale = trainer.quantize_float_tensor(w, 6)
        self.assertEqual(q.unique().numel(), 63)
        self.assertLess(scale.item(), 0.004)
        for bits in (6, 8):
            for shape in ((2, 3), (7,), ()):
                value = trainer.fake_sym_quant(torch.zeros(shape), bits)
                torch.testing.assert_close(value, torch.zeros(shape))

    def test_qat_only_changes_weights_that_export_quantizes(self):
        for width in (256, 257):
            layer = trainer.CastedLinear(width, width, bias=False)
            x = torch.randn(2, width)
            expected = layer.weight if width == 256 else trainer.fake_sym_quant(layer.weight)
            torch.testing.assert_close(layer(x), torch.nn.functional.linear(x, expected))
            layer.qat_enabled = False
            torch.testing.assert_close(layer(x), torch.nn.functional.linear(x, layer.weight))
        with self.assertRaises(ValueError):
            trainer.quantize_state_dict_int8({}, bits=7)

    def test_cooldown_finishes_at_the_step_budget_even_with_a_time_guard(self):
        lr = trainer.learning_rate_multiplier
        self.assertEqual(lr(899, 1000, 100), 1.0)
        self.assertAlmostEqual(lr(950, 1000, 100), 0.5)
        self.assertAlmostEqual(lr(999, 1000, 100), 0.01)
        self.assertEqual(lr(1000, 1000, 100), 0.0)
        self.assertEqual(lr(1001, 1000, 100), 0.0)
        self.assertAlmostEqual(lr(450, 500, 100, 180_000, 600_000), 0.5)
        self.assertLess(lr(500, 1000, 100, 599_000, 600_000), 0.01)
        self.assertEqual(lr(0, 10, 100), 1.0)
        self.assertEqual(lr(100, 1000, 0), 1.0)

    def test_ema_starts_from_the_supplied_trained_snapshot(self):
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(4)
        ema = trainer.EMA(model, decay=0.5)
        with torch.no_grad():
            model.weight.fill_(6)
        ema.update()
        raw = model.weight.detach().clone()
        ema.apply_shadow()
        torch.testing.assert_close(model.weight, torch.full_like(raw, 5))
        # The saved raw snapshot is independent of the shadow and can be restored exactly.
        with torch.no_grad():
            model.weight.copy_(raw)
        torch.testing.assert_close(model.weight, torch.full_like(raw, 6))
        with self.assertRaises(ValueError):
            trainer.EMA(model, decay=1.0)

    def test_validation_windows_cover_prefix_and_tail_once_across_ranks(self):
        for total in (1, 7, 8, 9, 19, 32, 33):
            for stride in (0, 1, 3, 8):
                with self.subTest(total=total, stride=stride):
                    scored = []
                    for rank in range(3):
                        for batch in trainer.validation_batches(total, 8, stride, 2, rank, 3):
                            self.assertLessEqual(len(batch), 2)
                            self.assertEqual(len({(end - start, offset) for start, end, offset in batch}), 1)
                            for start, end, offset in batch:
                                self.assertTrue(0 <= start < end <= total)
                                self.assertLessEqual(end - start, 8)
                                scored.extend(range(start + offset, end))
                    self.assertEqual(sorted(scored), list(range(total)))
        with self.assertRaises(ValueError):
            list(trainer.validation_batches(19, 8, 9, 2))

    def test_evaluation_counts_the_same_tokens_and_bytes_for_every_stride(self):
        class TokenLoss(nn.Module):
            def forward(self, x, y, score_start=0):
                return y[:, score_start:].double().mean()

        tokens = torch.arange(34)
        args = SimpleNamespace(val_batch_size=16, train_seq_len=8)
        byte_lut = (torch.arange(34) % 3 + 1).short()
        spaces = torch.arange(34) % 2 == 0
        boundaries = torch.arange(34) % 5 == 0
        expected_loss = tokens[1:].double().mean().item()
        expected_bytes = (byte_lut[1:] + (spaces[1:] & ~boundaries[:-1]).short()).sum().item()
        expected_bpb = expected_loss / math.log(2) * 33 / expected_bytes
        model = TokenLoss().eval()
        for stride in (0, 1, 3, 8):
            loss, bpb = trainer.eval_val(args, model, 0, 1, torch.device("cpu"), 1,
                                        tokens, byte_lut, spaces, boundaries, stride)
            self.assertAlmostEqual(loss, expected_loss)
            self.assertAlmostEqual(bpb, expected_bpb)
            self.assertFalse(model.training)

    def test_validation_loader_does_not_drop_a_partial_sequence(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "val.bin"
            header = np.zeros(256, dtype="<i4")
            header[:3] = (20240520, 1, 20)
            shard.write_bytes(header.tobytes() + np.arange(20, dtype="<u2").tobytes())
            tokens = trainer.load_validation_tokens(str(shard), 8)
            self.assertEqual(tokens.numel(), 20)

    def test_short_cpu_training_supports_fullgraph_qat_and_roundtrip(self):
        model = trainer.GPT(vocab_size=32, num_layers=2, model_dim=256, num_heads=4,
                            num_kv_heads=2, mlp_mult=2, tie_embeddings=False,
                            tied_embed_init_std=0.005, logit_softcap=30,
                            rope_base=10000, qk_gain_init=1.5, window_size=-1)
        self.assertIsNone(model.blocks[0].attn.flash_attention)
        self.assertFalse(model.lm_head.qat_enabled)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x, y = torch.randint(32, (2, 8)), torch.randint(32, (2, 8))
        for _ in range(2):
            optimizer.zero_grad()
            loss = compiled(x, y)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
        raw = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.eval()
        self.assertTrue(torch.isfinite(compiled(x, y, score_start=3)))
        obj, _ = trainer.quantize_state_dict_int8(raw)
        model.load_state_dict(trainer.dequantize_state_dict_int8(obj), strict=True)
        self.assertTrue(torch.isfinite(model(x, y)))
        model.load_state_dict(raw, strict=True)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, raw[name], rtol=0, atol=0)

    def test_new_and_historical_logs_keep_primary_scores_separate(self):
        text = """git_commit:abc123 source_sha256:deadbeef
experiment:quant_bits:6 qat_enabled:True embed_export:fp16 ema_enabled:True attention_window:-1 eval_stride:64
step:1000/1000 val_loss:2.3 val_bpb:1.4 train_time:123000ms step_avg:123.0ms
Serialized model quantized+zlib: 10000000 bytes
Total submission size quantized+zlib: 10050000 bytes
diagnostic weights:raw stage:float val_loss:2.3 val_bpb:1.4
diagnostic weights:raw stage:quantized val_loss:2.4 val_bpb:1.42
final_quantized_roundtrip_exact val_loss:2.4 val_bpb:1.42
final_sliding_window weights:raw eval_stride:64 val_loss:2.2 val_bpb:1.3
diagnostic weights:ema stage:float val_loss:2.6 val_bpb:1.5
diagnostic weights:ema stage:quantized val_loss:2.7 val_bpb:1.6
final_sliding_window weights:ema eval_stride:64 val_loss:2.5 val_bpb:1.45
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new.txt"
            path.write_text(text)
            run = parse_log(path)
        self.assertEqual(run.quant_bits, 6)
        self.assertTrue(run.ema_enabled)
        self.assertEqual(run.attention_window, -1)
        self.assertEqual(run.git_commit, "abc123")
        self.assertEqual(run.submission_bytes, 10_050_000)
        self.assertEqual(run.int8_val_bpb, 1.42)
        self.assertEqual(run.sliding_val_bpb, 1.3)
        self.assertEqual(len(run.diagnostics), 4)
        self.assertEqual(run.diagnostics["ema_quantized"]["val_bpb"], 1.6)
        historical = parse_log(Path(__file__).parent / "logs" / "baseline.txt")
        self.assertEqual(historical.int8_val_bpb, 1.36075333)
        self.assertEqual(historical.submission_bytes, 14_585_467)


if __name__ == "__main__":
    unittest.main()

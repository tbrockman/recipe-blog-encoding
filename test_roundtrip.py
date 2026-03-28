#!/usr/bin/env python3
"""
Test arithmetic coding encode/decode round-trip.

Uses a deterministic byte-level mock model to verify the AC logic
independently of any real LLM. Optionally tests with a real GGUF model.

    python test_roundtrip.py                          # mock model only
    python test_roundtrip.py --model ./model.gguf     # also test real model
"""
import argparse
import io
import sys
import contextlib
import numpy as np
import torch

from main import encode, decode, StegoModel


def _djb2(seq) -> int:
    """Deterministic hash for integer sequences."""
    h = 5381
    for x in seq:
        h = ((h << 5) + h + x) & 0xFFFFFFFF
    return h


class MockModel:
    """Byte-level mock model: token ID = byte value (0-127).

    Produces deterministic logits seeded by the full context, so both
    encoder and decoder see identical CDFs at every step.
    """

    VOCAB = 130  # 128 byte tokens + EOS(128) + BOS(129)

    def __init__(self):
        self.device = "cpu"

    def tokenize(self, text: str) -> torch.Tensor:
        bos = 129
        byte_tokens = [b for b in text.encode("utf-8") if b < 128]
        return torch.tensor([[bos] + byte_tokens])

    def tokenize_no_special(self, text: str) -> list[int]:
        return [b for b in text.encode("utf-8") if b < 128]

    def detokenize(self, token_ids: list[int]) -> str:
        return bytes(t for t in token_ids if t < 128).decode("ascii", errors="replace")

    def detokenize_bytes(self, token_ids: list[int]) -> bytes:
        return bytes(t for t in token_ids if t < 128)

    def logits_at(self, input_ids: torch.Tensor) -> torch.Tensor:
        seed = _djb2(input_ids[0].tolist()) % (2**31)
        rng = np.random.RandomState(seed)
        logits = np.full(self.VOCAB, -50.0, dtype=np.float32)
        # Only printable ASCII (32-126) gets real logits
        logits[32:127] = rng.randn(95).astype(np.float32) * 3.0
        return torch.from_numpy(logits)

    @property
    def eos_id(self) -> int:
        return 128


def compare_bytes(expected: bytes, got: bytes) -> str:
    """Human-readable diff of the first mismatched byte."""
    for i in range(max(len(expected), len(got))):
        a = expected[i] if i < len(expected) else None
        b = got[i] if i < len(got) else None
        if a != b:
            if a is not None and b is not None:
                xor = a ^ b
                bits = [7 - j for j in range(8) if xor & (1 << j)]
                ac = chr(a) if 32 <= a < 127 else "."
                bc = chr(b) if 32 <= b < 127 else "."
                return (
                    f"byte {i}: 0x{a:02X} ('{ac}') vs 0x{b:02X} ('{bc}') "
                    f"— bit(s) {bits} flipped"
                )
            return f"byte {i}: length mismatch ({len(expected)} vs {len(got)})"
    return "identical"


def run_test(model, message, prompt, top_k=64, max_tail_tokens=0, label="", quiet=True):
    """Encode then decode a message; return True if round-trip is lossless."""
    secret = message.encode("utf-8") if isinstance(message, str) else message

    cm = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with cm:
        stego_text = encode(
            model, secret, prompt,
            top_k=top_k,
            max_tail_tokens=max_tail_tokens,
        )
        recovered = decode(model, stego_text, prompt, top_k=top_k)

    tag = f"[{label}] " if label else ""
    display = repr(message)
    if len(display) > 50:
        display = display[:47] + "..."

    if recovered == secret:
        print(f"  PASS {tag}{display}")
        return True

    print(f"  FAIL {tag}{display}")
    print(f"       {compare_bytes(secret, recovered)}")
    return False


MESSAGES = [
    "x",
    "hi",
    "hello world",
    "attack at dawn",
    "https://www.nokings.org/",
    "The quick brown fox jumps over the lazy dog",
    "a" * 100,
    "b" * 200,
]


def main():
    parser = argparse.ArgumentParser(description="AC round-trip tests")
    parser.add_argument("--model", help="GGUF model path for real-model tests")
    parser.add_argument("--message", help="Test a single message instead of the suite")
    parser.add_argument(
        "--prompt", default="Write a blog post: ",
        help="Prompt (default: ASCII test prompt)",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show encode/decode output")
    args = parser.parse_args()

    messages = [args.message] if args.message else MESSAGES
    passed = failed = 0

    # --- Mock model tests ---
    print("=" * 60)
    print("Mock model (byte-level) — pure AC logic")
    print("=" * 60)

    mock = MockModel()
    quiet = not args.verbose

    print("\n--- No tail ---")
    for msg in messages:
        ok = run_test(mock, msg, args.prompt, max_tail_tokens=0,
                      label="no-tail", quiet=quiet)
        passed += ok
        failed += not ok

    print("\n--- With tail (50 tokens) ---")
    for msg in messages:
        ok = run_test(mock, msg, args.prompt, max_tail_tokens=50,
                      label="tail=50", quiet=quiet)
        passed += ok
        failed += not ok

    print("\n--- top_k=8 (high bits/token) ---")
    for msg in messages[:4]:
        ok = run_test(mock, msg, args.prompt, top_k=8,
                      label="k=8", quiet=quiet)
        passed += ok
        failed += not ok

    print("\n--- top_k=128 (low bits/token) ---")
    for msg in messages[:4]:
        ok = run_test(mock, msg, args.prompt, top_k=128,
                      label="k=128", quiet=quiet)
        passed += ok
        failed += not ok

    # --- Real model tests ---
    if args.model:
        print(f"\n{'=' * 60}")
        print(f"Real model: {args.model}")
        print("=" * 60)
        real = StegoModel(args.model)
        for msg in messages:
            ok = run_test(
                real, msg, args.prompt,
                top_k=256, max_tail_tokens=0,
                label="real", quiet=quiet,
            )
            passed += ok
            failed += not ok

    # --- Summary ---
    total = passed + failed
    status = "ALL PASSED" if failed == 0 else f"{failed} FAILED"
    print(f"\n{'=' * 60}")
    print(f"{status}  ({passed}/{total})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

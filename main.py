#!/usr/bin/env python3
"""
Arithmetic Steganography via Language Model
============================================

Hides secret messages in natural-looking text by exploiting the probability
distributions of a language model. Each generated token encodes bits of the
secret message by constraining which token is selected from the model's
predicted distribution, using arithmetic coding for optimal efficiency.

Requirements:
    pip install torch transformers bitsandbytes accelerate numpy

Usage:
    # Encode a secret message
    python stego.py encode \
        --message "attack at dawn" \
        --prompt "The best thing about making scrambled eggs is: "

    # Decode from stego text (prompt must match!)
    python stego.py decode \
        --stego-file stego_output.txt \
        --prompt "The best thing about making scrambled eggs is: "

    # Use a specific model
    python stego.py encode \
        --model "mistralai/Mistral-7B-v0.3" \
        --message "secret message" \
        --prompt "My grandmother always"

CRITICAL: Both encoder and decoder must use the exact same model, quantization,
          top-k value, and prompt. Any difference causes decoding failure.
"""

import argparse
import struct
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Arithmetic coding constants
# We use 64-bit integer precision. The interval [lo, hi] lives in [0, WHOLE).
# Renormalization keeps the interval width >= QUARTER, so we never run out
# of precision regardless of message length.
# ---------------------------------------------------------------------------
PRECISION = 48  # bits of precision in the interval
WHOLE = 1 << PRECISION
HALF = 1 << (PRECISION - 1)
QUARTER = 1 << (PRECISION - 2)
THREE_QUARTER = 3 * QUARTER
MASK = WHOLE - 1


# ---------------------------------------------------------------------------
# Bit I/O helpers
# ---------------------------------------------------------------------------
class BitReader:
    """Reads individual bits from a byte sequence, MSB first."""

    def __init__(self, data: bytes):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0
        self._total_read = 0

    def read(self) -> int:
        """Read one bit. Returns alternating bits after data is exhausted."""
        if self.byte_pos >= len(self.data):
            self._total_read += 1
            # Alternating padding prevents the value register from
            # converging to zero, which would lock token selection onto
            # the first CDF entry and stall renormalization.
            return self._total_read & 1
        bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.bit_pos = 0
            self.byte_pos += 1
        self._total_read += 1
        return bit

    @property
    def total_bits_read(self) -> int:
        return self._total_read


class BitWriter:
    """Collects individual bits and converts to bytes."""

    def __init__(self):
        self.bits: list[int] = []

    def write(self, bit: int):
        self.bits.append(bit & 1)

    def write_with_pending(self, bit: int, pending: int):
        """Write a bit followed by `pending` copies of its complement."""
        self.write(bit)
        for _ in range(pending):
            self.write(bit ^ 1)

    def to_bytes(self) -> bytes:
        # Pad to a full byte
        padded = self.bits + [0] * ((8 - len(self.bits) % 8) % 8)
        out = bytearray()
        for i in range(0, len(padded), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | padded[i + j]
            out.append(byte)
        return bytes(out)

    def __len__(self):
        return len(self.bits)


# ---------------------------------------------------------------------------
# CDF construction from logits
# ---------------------------------------------------------------------------
def build_cdf(
    logits: torch.Tensor,
    top_k: int = 256,
    cdf_total: int = 1 << 16,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build an integer cumulative frequency table from model logits.

    1. Take top-k tokens by logit value.
    2. Softmax over those to get probabilities.
    3. Quantize to integer frequencies summing to `cdf_total`.
    4. Build cumulative frequency array.

    Returns:
        token_ids:  int array of shape (K,)    — the token IDs in this CDF
        cum_freq:   int array of shape (K+1,)  — cumulative frequencies
                    cum_freq[0] = 0, cum_freq[-1] = cdf_total
        cdf_total:  the total frequency mass
    """
    # Top-k filtering
    values, indices = torch.topk(logits.float(), min(top_k, logits.shape[-1]))

    # Softmax to probabilities
    probs = torch.softmax(values, dim=-1)

    # Quantize to integer frequencies, minimum 1 per token
    freqs = torch.clamp((probs * cdf_total).floor().long(), min=1)

    # Correct rounding error so frequencies sum to exactly cdf_total
    diff = cdf_total - freqs.sum().item()
    if diff > 0:
        # Distribute surplus to highest-probability tokens
        freqs[0] += diff
    elif diff < 0:
        # Remove from highest-prob tokens (preserving min=1)
        for i in range(len(freqs)):
            take = min(-diff, freqs[i].item() - 1)
            freqs[i] -= take
            diff += take
            if diff == 0:
                break

    # Build cumulative frequencies
    cum_freq = np.zeros(len(freqs) + 1, dtype=np.int64)
    freq_np = freqs.cpu().numpy()
    np.cumsum(freq_np, out=cum_freq[1:])
    cum_freq[-1] = cdf_total  # ensure exact

    token_ids = indices.cpu().numpy()
    return token_ids, cum_freq, cdf_total


# ---------------------------------------------------------------------------
# Model wrapper — supports HuggingFace Transformers and GGUF via llama-cpp
# ---------------------------------------------------------------------------
class StegoModel:
    """Wraps a causal LM for token-by-token logit access.

    Pass a HuggingFace model name for the Transformers backend, or a path
    ending in .gguf to use the llama-cpp-python backend (which supports
    partial GPU offloading via --n-gpu-layers).
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        quantize_4bit: bool = True,
        n_gpu_layers: int | None = None,
    ):
        self._backend = "gguf" if model_name.endswith(".gguf") else "hf"

        if self._backend == "gguf":
            self._init_gguf(model_name, n_gpu_layers)
        else:
            self._init_hf(model_name, device, quantize_4bit)

    # -- GGUF backend (llama-cpp-python) ------------------------------------

    def _init_gguf(self, model_path: str, n_gpu_layers: int | None):
        from llama_cpp import Llama

        if n_gpu_layers is None:
            n_gpu_layers = -1  # offload all layers that fit

        print(f"Loading GGUF model: {model_path}")
        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=4096,
            logits_all=True,
            verbose=True,
        )
        self.device = "cpu"
        self._n_eval = 0
        print("Model loaded.\n")

    # -- HuggingFace backend ------------------------------------------------

    def _init_hf(self, model_name: str, device: str, quantize_4bit: bool):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading tokenizer: {model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        print(f"Loading model: {model_name} ({'4-bit' if quantize_4bit else 'fp16'})")
        load_kwargs = dict(torch_dtype=torch.float16, device_map=device)

        if device == "auto" and torch.cuda.is_available():
            max_memory = {}
            for i in range(torch.cuda.device_count()):
                free_bytes = torch.cuda.mem_get_info(i)[0]
                headroom_gb = max(1, free_bytes // (1024**3) - 2)
                max_memory[i] = f"{headroom_gb}GiB"
            max_memory["cpu"] = "48GiB"
            load_kwargs["max_memory"] = max_memory

        if quantize_4bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=True,
                )
            except ImportError:
                print("Warning: bitsandbytes not found, loading in fp16")

        self._model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self._model.eval()
        self.device = self._model.device
        print("Model loaded.\n")

    # -- Unified interface --------------------------------------------------

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize with BOS / special tokens. Resets GGUF KV cache."""
        if self._backend == "gguf":
            self._llm.reset()
            self._n_eval = 0
            tokens = self._llm.tokenize(text.encode(), add_bos=True)
            return torch.tensor([tokens])
        return self._tokenizer.encode(text, return_tensors="pt").to(self.device)

    def tokenize_no_special(self, text: str) -> list[int]:
        if self._backend == "gguf":
            return self._llm.tokenize(text.encode(), add_bos=False)
        return self._tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, token_ids: list[int]) -> str:
        if self._backend == "gguf":
            return self._llm.detokenize(token_ids).decode("utf-8", errors="replace")
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def detokenize_bytes(self, token_ids: list[int]) -> bytes:
        """Raw bytes without UTF-8 decoding — safe for incremental accumulation."""
        if self._backend == "gguf":
            return self._llm.detokenize(token_ids)
        return self._tokenizer.decode(token_ids, skip_special_tokens=True).encode("utf-8")

    def logits_at(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get logits for the next token given input_ids context."""
        if self._backend == "gguf":
            tokens = input_ids[0].tolist()
            new_tokens = tokens[self._n_eval:]
            self._llm.eval(new_tokens)
            self._n_eval = len(tokens)
            logits_np = np.array(self._llm.scores[self._n_eval - 1], dtype=np.float32)
            return torch.from_numpy(logits_np)
        with torch.no_grad():
            return self._model(input_ids).logits[0, -1, :]

    @property
    def eos_id(self) -> int:
        if self._backend == "gguf":
            return self._llm.token_eos()
        return self._tokenizer.eos_token_id


# ---------------------------------------------------------------------------
# Canonical BPE filtering
#
# The arithmetic coder selects tokens based on the bit stream, not BPE merge
# rules. This can produce non-canonical token sequences (e.g. ["Th", "ey"]
# instead of ["They"]) that collapse when re-tokenized, breaking decoding.
# We filter the CDF at each step to only include tokens that maintain a
# canonical BPE tokenization, so the text round-trip is lossless.
# ---------------------------------------------------------------------------
def filter_canonical_tokens(
    model: StegoModel,
    generated_so_far: list[int],
    bytes_so_far: bytes,
    token_ids: np.ndarray,
    cum_freq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Filter CDF to only include tokens whose selection maintains canonical
    BPE tokenization, ensuring the round-trip through text is lossless.

    Uses raw bytes (not decoded strings) to avoid corrupting multi-byte
    UTF-8 sequences when tokens are decoded individually.
    """
    n = len(generated_so_far)
    # Decode the accumulated bytes as a whole — safe for multi-byte chars
    text_so_far = bytes_so_far.decode("utf-8", errors="replace")
    mask = np.ones(len(token_ids), dtype=bool)

    for i, tid in enumerate(token_ids):
        candidate_bytes = model.detokenize_bytes([int(tid)])
        new_text = (bytes_so_far + candidate_bytes).decode("utf-8", errors="replace")
        retokenized = model.tokenize_no_special(new_text)
        if len(retokenized) != n + 1 or retokenized[n] != int(tid):
            mask[i] = False

    if not mask.any():
        # Shouldn't happen — fall back to unfiltered CDF
        return token_ids, cum_freq, int(cum_freq[-1])

    # Extract individual frequencies and filter
    freqs = np.diff(cum_freq)
    filtered_ids = token_ids[mask]
    filtered_freqs = freqs[mask]

    # Rebuild cumulative frequencies
    new_cum_freq = np.zeros(len(filtered_freqs) + 1, dtype=np.int64)
    np.cumsum(filtered_freqs, out=new_cum_freq[1:])
    new_total = int(new_cum_freq[-1])

    return filtered_ids, new_cum_freq, new_total


# ---------------------------------------------------------------------------
# Encoder: secret bits → stego text
#
# This is conceptually an arithmetic DECODER: it reads bits from the secret
# message and emits symbols (tokens). The secret bitstream defines a point
# in [0, 1), and we "decode" that point into a token sequence using the
# model's probability distributions as the codebook.
# ---------------------------------------------------------------------------
def encode(
    model: StegoModel,
    secret: bytes,
    prompt: str,
    top_k: int = 256,
    max_tokens: int | None = None,
    max_tail_tokens: int = 200,
) -> str:
    """
    Encode `secret` bytes into natural text, guided by `prompt`.

    The output text is statistically close to normal model output for the
    given prompt — each token was a plausible next token in context.
    """
    # Prepend a 4-byte big-endian length header so the decoder knows
    # how many bytes to extract.
    payload = struct.pack(">I", len(secret)) + secret
    reader = BitReader(payload)
    total_payload_bits = len(payload) * 8

    # --- Arithmetic coder state ---
    lo = 0
    hi = WHOLE - 1

    # Fill the value register with the first PRECISION bits of the message.
    # This register represents "where in the current interval our target
    # point lies."
    value = 0
    for _ in range(PRECISION):
        value = (value << 1) | reader.read()

    # --- Generation loop ---
    input_ids = model.tokenize(prompt)
    generated: list[int] = []
    bytes_so_far = b""
    encoding_complete = False
    # Mirror the decoder's "pending" counter so we know when all
    # payload bits have been decided during renormalization.
    dec_pending = 0

    step = 0
    while max_tokens is None or step < max_tokens:
        logits = model.logits_at(input_ids)
        token_ids, cum_freq, freq_total = build_cdf(logits, top_k)

        # Filter to tokens that preserve canonical BPE tokenization
        token_ids, cum_freq, freq_total = filter_canonical_tokens(
            model, generated, bytes_so_far, token_ids, cum_freq,
        )

        rng = hi - lo + 1

        # Normal AC token selection: find the token whose
        # sub-interval contains our value.
        scaled = ((value - lo + 1) * freq_total - 1) // rng
        idx = int(np.searchsorted(cum_freq, scaled, side="right")) - 1
        idx = max(0, min(idx, len(token_ids) - 1))

        # Narrow the interval to this token's sub-interval
        sym_lo = int(cum_freq[idx])
        sym_hi = int(cum_freq[idx + 1])
        hi = lo + (rng * sym_hi) // freq_total - 1
        lo = lo + (rng * sym_lo) // freq_total

        # Emit the token
        token_id = int(token_ids[idx])
        generated.append(token_id)
        bytes_so_far += model.detokenize_bytes([token_id])

        # Extend context for next step
        next_tok = torch.tensor([[token_id]], device=model.device)
        input_ids = torch.cat([input_ids, next_tok], dim=1)

        # --- Renormalization ---
        # Shift out the MSBs that lo and hi agree on, and read new
        # message bits into the value register to maintain precision.
        while True:
            if hi < HALF:
                # Both start with 0 — shift out
                dec_pending = 0
                lo = (lo << 1) & MASK
                hi = ((hi << 1) | 1) & MASK
                value = ((value << 1) | reader.read()) & MASK
            elif lo >= HALF:
                # Both start with 1 — subtract HALF, shift, restore
                dec_pending = 0
                lo = ((lo - HALF) << 1) & MASK
                hi = (((hi - HALF) << 1) | 1) & MASK
                value = (((value - HALF) << 1) | reader.read()) & MASK
            elif lo >= QUARTER and hi < THREE_QUARTER:
                # Underflow / near-convergence — shift out second MSB
                dec_pending += 1
                lo = ((lo - QUARTER) << 1) & MASK
                hi = (((hi - QUARTER) << 1) | 1) & MASK
                value = (((value - QUARTER) << 1) | reader.read()) & MASK
            else:
                break

        # Completion: all payload bits consumed AND the decoder's
        # pending counter is zero, meaning every payload bit has been
        # decided during renormalization (not left to the flush).
        if reader.total_bits_read >= total_payload_bits + PRECISION:
            if dec_pending == 0:
                encoding_complete = True
        if encoding_complete:
            break

        # Stop on EOS
        if token_id == model.eos_id:
            break

        # Progress indicator
        bits_encoded = min(reader.total_bits_read, total_payload_bits)
        if step % 20 == 0:
            pct = bits_encoded / total_payload_bits * 100
            bpt = bits_encoded / (step + 1) if step > 0 else 0
            print(
                f"  step {step:4d} | {bits_encoded}/{total_payload_bits} bits "
                f"({pct:.0f}%) | {bpt:.1f} bits/token"
            )

        step += 1

    # --- Tail generation ---
    # After the payload is fully encoded, continue generating greedily so the
    # text concludes naturally instead of cutting off mid-sentence. The decoder
    # ignores tail tokens (the 4-byte length header tells it when to stop).
    encoding_tokens = len(generated)
    if not encoding_complete:
        bits_done = min(reader.total_bits_read, total_payload_bits)
        print(
            f"\n  WARNING: encoding incomplete — {bits_done}/{total_payload_bits} bits "
            f"in {encoding_tokens} tokens."
        )
    if encoding_complete and max_tail_tokens > 0 and token_id != model.eos_id:
        print(f"\n  Payload encoded at {encoding_tokens} tokens, generating tail...")
        tail_ids: list[int] = []
        NGRAM = 6  # stop if an n-gram of this length repeats
        for _ in range(max_tail_tokens):
            logits = model.logits_at(input_ids)
            token_ids, cum_freq, freq_total = build_cdf(logits, top_k)
            token_ids, cum_freq, freq_total = filter_canonical_tokens(
                model, generated, bytes_so_far, token_ids, cum_freq,
            )

            # Greedy: highest-probability canonical token
            token_id = int(token_ids[0])
            if token_id == model.eos_id:
                break

            tail_ids.append(token_id)
            generated.append(token_id)
            bytes_so_far += model.detokenize_bytes([token_id])
            next_tok = torch.tensor([[token_id]], device=model.device)
            input_ids = torch.cat([input_ids, next_tok], dim=1)

            # Detect repetition: if the last NGRAM tokens appeared earlier
            # in the tail, the model is looping — roll back to the first
            # occurrence and stop.
            if len(tail_ids) >= NGRAM * 2:
                ngram = tuple(tail_ids[-NGRAM:])
                for i in range(len(tail_ids) - NGRAM * 2 + 1):
                    if tuple(tail_ids[i : i + NGRAM]) == ngram:
                        # Roll back: keep only up to the first occurrence
                        keep = encoding_tokens + i + NGRAM
                        generated = generated[:keep]
                        bytes_so_far = model.detokenize_bytes(generated)
                        print("  Tail: repetition detected, stopping.")
                        token_id = model.eos_id
                        break
                if token_id == model.eos_id:
                    break

    stego_text = bytes_so_far.decode("utf-8", errors="replace")

    # Final stats
    bits_encoded = min(reader.total_bits_read, total_payload_bits)
    n_tokens = len(generated)
    tail_tokens = n_tokens - encoding_tokens
    print(f"\n  Done: {encoding_tokens} tokens encoding + {tail_tokens} tokens tail = {n_tokens} total")
    print(f"  ~{bits_encoded / encoding_tokens:.2f} bits/token")
    print(f"  Payload: {len(secret)} bytes in {n_tokens} tokens ({len(stego_text)} chars)")

    return stego_text


# ---------------------------------------------------------------------------
# Decoder: stego text → secret bits
#
# This is conceptually an arithmetic ENCODER: it reads symbols (tokens from
# the stego text) and outputs bits (the secret message). It narrows the
# interval exactly as the encoder did, and the MSBs that become decided
# during renormalization ARE the recovered message bits.
# ---------------------------------------------------------------------------
def decode(
    model: StegoModel,
    stego_text: str,
    prompt: str,
    top_k: int = 256,
) -> bytes:
    """
    Decode a secret message from stego text.

    The model and prompt must be identical to those used for encoding.
    """
    stego_ids = model.tokenize_no_special(stego_text)
    input_ids = model.tokenize(prompt)
    decoded_so_far: list[int] = []
    bytes_so_far = b""

    # --- Arithmetic coder state ---
    lo = 0
    hi = WHOLE - 1
    pending = 0  # pending bits for underflow resolution

    writer = BitWriter()

    for i, token_id in enumerate(stego_ids):
        logits = model.logits_at(input_ids)
        token_ids, cum_freq, freq_total = build_cdf(logits, top_k)

        # Apply same canonical BPE filter as encoder
        token_ids, cum_freq, freq_total = filter_canonical_tokens(
            model, decoded_so_far, bytes_so_far, token_ids, cum_freq,
        )

        # Find this token in our filtered vocabulary
        matches = np.where(token_ids == token_id)[0]
        if len(matches) == 0:
            # Token fell outside filtered top-k. This means the encoder
            # wouldn't have chosen it, so something is wrong — but we
            # continue gracefully.
            print(f"  Warning: token {token_id!r} not in top-{top_k} at position {i}")
            decoded_so_far.append(token_id)
            bytes_so_far += model.detokenize_bytes([token_id])
            next_tok = torch.tensor([[token_id]], device=model.device)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
            continue

        idx = int(matches[0])

        # Narrow interval identically to the encoder
        rng = hi - lo + 1
        sym_lo = int(cum_freq[idx])
        sym_hi = int(cum_freq[idx + 1])
        hi = lo + (rng * sym_hi) // freq_total - 1
        lo = lo + (rng * sym_lo) // freq_total

        # Update context and tracking
        decoded_so_far.append(token_id)
        bytes_so_far += model.detokenize_bytes([token_id])
        next_tok = torch.tensor([[token_id]], device=model.device)
        input_ids = torch.cat([input_ids, next_tok], dim=1)

        # --- Renormalization ---
        # When the MSBs of lo and hi agree, those bits are decided —
        # they are recovered message bits. Pending bits handle the
        # underflow case where lo and hi straddle a boundary.
        while True:
            if hi < HALF:
                # MSB is 0 for both
                writer.write_with_pending(0, pending)
                pending = 0
                lo = (lo << 1) & MASK
                hi = ((hi << 1) | 1) & MASK
            elif lo >= HALF:
                # MSB is 1 for both
                writer.write_with_pending(1, pending)
                pending = 0
                lo = ((lo - HALF) << 1) & MASK
                hi = (((hi - HALF) << 1) | 1) & MASK
            elif lo >= QUARTER and hi < THREE_QUARTER:
                # Underflow: lo = 01..., hi = 10... — MSBs don't agree yet
                # but will on the next decided bit. Track as pending.
                pending += 1
                lo = ((lo - QUARTER) << 1) & MASK
                hi = (((hi - QUARTER) << 1) | 1) & MASK
            else:
                break

    # Flush: emit final bits to resolve any remaining pending bits.
    # The +1 ensures the last pending underflow bit is properly resolved.
    pending += 1
    if lo < QUARTER:
        writer.write_with_pending(0, pending)
    else:
        writer.write_with_pending(1, pending)

    # Convert recovered bits to bytes and parse the length-prefixed message
    raw = writer.to_bytes()

    if len(raw) < 4:
        raise ValueError(f"Decoded only {len(raw)} bytes — not enough for length header")

    msg_len = struct.unpack(">I", raw[:4])[0]
    print(f"  Decoded length header: {msg_len} bytes")
    print(f"  Total recovered: {len(raw)} bytes ({len(writer)} bits)")

    if msg_len > len(raw) - 4:
        print(f"  Warning: expected {msg_len} bytes but only have {len(raw) - 4}")
        msg_len = len(raw) - 4

    return raw[4 : 4 + msg_len]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Arithmetic Steganography via Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stego.py encode --message "meet me at noon" \\
      --prompt "The best scrambled eggs I ever had"

  python stego.py decode --stego-file stego_output.txt \\
      --prompt "The best scrambled eggs I ever had"

  python stego.py encode --message "hello" \\
      --model mistralai/Mistral-7B-v0.3 \\
      --prompt "Every Sunday morning my grandmother would"
        """,
    )

    parser.add_argument("mode", choices=["encode", "decode"])
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-3B",
        help="HuggingFace model name/path (default: Llama-3.2-3B)",
    )
    parser.add_argument(
        "--prompt",
        default="You are writing a typical online recipe preamble, aiming to gamify SEO. Write your recipe about scrambled eggs: ",
        help="Context prompt (MUST match between encode/decode)",
    )
    parser.add_argument("--message", help="Secret message to encode (encode mode)")
    parser.add_argument("--stego-text", help="Stego text string to decode")
    parser.add_argument(
        "--stego-file",
        default="stego_output.txt",
        help="Stego text file path (output for encode, input for decode)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=256,
        help="Top-k vocabulary size for CDF (default: 256)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate (default: unlimited)",
    )
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (uses fp16; needs more VRAM)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=None,
        help="Layers to offload to GPU (GGUF only, default: all that fit)",
    )
    parser.add_argument(
        "--max-tail-tokens",
        type=int,
        default=200,
        help="Max tokens for greedy tail after encoding (default: 200, 0 to disable)",
    )

    args = parser.parse_args()

    # Load model
    stego_model = StegoModel(
        args.model,
        device=args.device,
        quantize_4bit=not args.no_4bit,
        n_gpu_layers=args.n_gpu_layers,
    )

    if args.mode == "encode":
        secret_msg = args.message or input("Enter secret message: ")
        secret_bytes = secret_msg.encode("utf-8")

        print(f"Secret: {len(secret_bytes)} bytes ({len(secret_bytes) * 8} bits)")
        print(f"Prompt: {args.prompt!r}")
        print(f"Top-k:  {args.top_k}")
        print()

        stego_text = encode(
            stego_model,
            secret_bytes,
            args.prompt,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            max_tail_tokens=args.max_tail_tokens,
        )

        print("\n" + "=" * 70)
        print("STEGO TEXT (paste this wherever recipe preambles go):")
        print("=" * 70)
        print(stego_text)
        print("=" * 70)

        with open(args.stego_file, "w") as f:
            f.write(stego_text)
        print(f"\nSaved to {args.stego_file}")

    elif args.mode == "decode":
        if args.stego_text:
            stego_text = args.stego_text
        elif args.stego_file:
            with open(args.stego_file) as f:
                stego_text = f.read()
        else:
            print("Paste stego text (Ctrl-D when done):")
            stego_text = sys.stdin.read()

        print(f"Stego text: {len(stego_text)} chars")
        print(f"Prompt: {args.prompt!r}")
        print(f"Top-k:  {args.top_k}")
        print()

        try:
            recovered = decode(
                stego_model,
                stego_text,
                args.prompt,
                top_k=args.top_k,
            )
            print("\n" + "=" * 70)
            print("RECOVERED SECRET MESSAGE:")
            print("=" * 70)
            print(recovered.decode("utf-8", errors="replace"))
            print("=" * 70)
        except Exception as e:
            print(f"\nDecode failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
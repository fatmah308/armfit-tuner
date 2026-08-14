"""
Measures quality degradation from quantization using llama.cpp's
llama-perplexity tool against a small held-out text sample.

Perplexity is measured on the HOST machine (not on-device) since it
only affects the "how much quality did we give up" question, not the
"how fast is it on Arm" question. This keeps the pipeline faster and
avoids pushing large evaluation text files to the phone.
"""

import subprocess
from pathlib import Path
from typing import Dict

from armfit.binaries import find_host_binary

_SAMPLE_PARAGRAPH = """The quick brown fox jumps over the lazy dog. Artificial intelligence
systems are increasingly deployed on resource-constrained edge devices, where memory,
compute, and battery life impose hard limits on model size and inference speed. Efficient
quantization techniques aim to reduce these costs while preserving as much of the
original model's output quality as possible. Evaluating this tradeoff requires a
consistent held-out text sample so that perplexity scores are comparable across
different quantization levels of the same underlying model."""

# llama-perplexity computes PPL using a sliding window over the FULL context
# size and produces no usable output at all if the input has fewer tokens
# than that window, no error, just silence where the "Final estimate" line
# should be. That was the actual cause of every reading coming back nan: the
# original ~100-token sample was well under the tool's default context.
# Repeating the paragraph guarantees well over a thousand tokens regardless
# of tokenizer, and PPL_CTX_SIZE is passed explicitly so this doesn't depend
# on whatever default happens to be compiled into a given llama.cpp build.
_SAMPLE_TEXT = (_SAMPLE_PARAGRAPH + "\n\n") * 20
_PPL_CTX_SIZE = 256


def _find_perplexity_binary(llama_dir: Path) -> Path:
    return find_host_binary(llama_dir, "llama-perplexity")


def perplexity_all(quantized_paths: Dict[str, Path], llama_dir: Path, work_dir: Path) -> Dict[str, float]:
    ppl_bin = _find_perplexity_binary(llama_dir)

    sample_path = work_dir / "ppl_sample.txt"
    sample_path.write_text(_SAMPLE_TEXT, encoding="utf-8")

    results: Dict[str, float] = {}
    for quant_name, gguf_path in quantized_paths.items():
        cmd = [str(ppl_bin), "-m", str(gguf_path), "-f", str(sample_path), "-c", str(_PPL_CTX_SIZE)]
        print(f"      Measuring perplexity for {quant_name} ...")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        combined_output = result.stdout + result.stderr
        ppl = _parse_perplexity(combined_output)

        if ppl != ppl:  # NaN check (nan != nan is always True), avoids importing math for one check
            print(f"      WARNING: could not parse a perplexity value for {quant_name}. "
                  f"Last lines of llama-perplexity output:")
            for line in combined_output.strip().splitlines()[-8:]:
                print(f"        {line}")

        results[quant_name] = ppl

    return results


def _parse_perplexity(output: str) -> float:
    # Only match the definitive final-summary line, e.g.
    # "Final estimate: PPL = 12.3456 +/- 0.1234". Matching any line that
    # merely contains "PPL" is too loose, llama-perplexity can print that
    # substring in earlier diagnostic/setup lines too, which was silently
    # extracting the wrong number (an early diagnostic value, not the real
    # aggregate perplexity) and produced implausible results where
    # quantized models appeared to have BETTER perplexity than F16, which
    # is not something quantization actually does.
    for line in output.splitlines():
        if "Final estimate" in line:
            for token in line.replace("=", " ").split():
                try:
                    return float(token)
                except ValueError:
                    continue
    return float("nan")
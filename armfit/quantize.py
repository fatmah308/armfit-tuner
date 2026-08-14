"""
Quantize an F16 GGUF file to multiple levels using llama.cpp's
llama-quantize binary.

Note: KleidiAI's accelerated kernels specifically target Q4_0 and
Q8_0 weight layouts, so those two quant levels matter most for the
"Arm-specific speedup" comparison. F16 is kept as the unquantized
baseline for size/quality comparison.
"""

import subprocess
from pathlib import Path
from typing import Dict, List

from armfit.binaries import find_host_binary


class QuantizeError(RuntimeError):
    pass


def _find_quantize_binary(llama_dir: Path) -> Path:
    try:
        return find_host_binary(llama_dir, "llama-quantize")
    except FileNotFoundError as e:
        raise QuantizeError(str(e)) from e


def quantize_all(
    f16_gguf: Path,
    llama_dir: Path,
    quant_levels: List[str],
    work_dir: Path,
) -> Dict[str, Path]:
    """
    Returns a dict mapping quant level name -> path to quantized .gguf file.
    "F16" maps to the original f16_gguf unchanged.
    """
    results: Dict[str, Path] = {}
    quantize_bin = None

    for level in quant_levels:
        level = level.upper()
        if level in ("F16", "FP16"):
            results["F16"] = f16_gguf
            continue

        if quantize_bin is None:
            quantize_bin = _find_quantize_binary(llama_dir)

        out_path = work_dir / f"{f16_gguf.stem}-{level}.gguf"
        cmd = [str(quantize_bin), str(f16_gguf), str(out_path), level]
        print(f"      Quantizing -> {level}: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise QuantizeError(f"Quantization to {level} failed:\n{result.stderr}")
        if not out_path.exists():
            raise QuantizeError(f"llama-quantize reported success but {out_path} was not created.")

        results[level] = out_path

    return results

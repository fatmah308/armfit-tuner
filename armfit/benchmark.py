"""
Runs llama-bench on-device via adb and parses its output into
structured prefill/decode speed numbers.

llama-bench prints a markdown-style table, e.g.:

| model | size | params | backend | threads | test  | t/s          |
|-------|------|--------|---------|---------|-------|--------------|
| qwen2 1.5B Q4_K | 1.04 GiB | 1.78 B | CPU | 8 | pp512 | 64.60 ± 0.27 |
| qwen2 1.5B Q4_K | 1.04 GiB | 1.78 B | CPU | 8 | tg128 | 36.29 ± 0.16 |

"pp" rows are prefill (prompt processing) speed, "tg" rows are
decode (token generation) speed.

Parsing is deliberately column-order-independent: rather than
matching the whole row against a fixed column count, we split each
row on "|" and independently look for whichever cell matches the
pp<N>/tg<N> test-name pattern and whichever cell matches the
"<number> ± <number>" speed pattern. This survives llama-bench
adding/removing/reordering columns between versions, which a
strict positional regex would not.
"""

import re
from typing import Dict, Optional

from armfit.device import DeviceError

_TEST_CELL_RE = re.compile(r"^(pp|tg)(\d+)$")
_TPS_CELL_RE = re.compile(r"([\d.]+)\s*±")


def _parse_llama_bench_output(output: str) -> Dict[str, Optional[float]]:
    prefill_tps = None
    decode_tps = None

    for line in output.splitlines():
        if "|" not in line or "±" not in line:
            continue

        cells = [c.strip() for c in line.strip().strip("|").split("|")]

        test_cell = next((c for c in cells if _TEST_CELL_RE.match(c)), None)
        if test_cell is None:
            continue

        tps_cell = next((c for c in cells if _TPS_CELL_RE.search(c)), None)
        if tps_cell is None:
            continue

        tps = float(_TPS_CELL_RE.search(tps_cell).group(1))
        if test_cell.startswith("pp"):
            prefill_tps = tps
        elif test_cell.startswith("tg"):
            decode_tps = tps

    if prefill_tps is None and decode_tps is None:
        raise DeviceError(
            "Could not parse any pp/tg rows from llama-bench output. This usually "
            "means the llama-bench table format changed between versions, or the "
            "binary crashed/printed an error instead of a results table.\n"
            f"Raw output was:\n{output}"
        )

    return {"prefill_tps": prefill_tps, "decode_tps": decode_tps}


def run_llama_bench(
    dev,
    bin_remote_path: str,
    model_remote_path: str,
    n_prompt: int,
    n_gen: int,
    threads: int,
    repeats: int = 3,
) -> Dict[str, Optional[float]]:
    """
    Runs llama-bench `repeats` times and averages the result, since
    on-device timings vary run to run depending on thermal state and
    background load.

    Returns prefill_tps, decode_tps, and ttft_ms. TTFT is DERIVED from
    prefill speed (n_prompt / prefill_tps), not measured with a separate
    on-device call. A separate wall-clock round trip would reload the
    model from storage each time and would be dominated by adb
    communication overhead and disk I/O rather than actual inference
    latency, which would overstate TTFT substantially. Deriving it from
    the already-measured prefill throughput is the more defensible
    number: time-to-first-token is, by definition, the time to process
    the prompt and emit the first generated token, which prefill speed
    directly measures.
    """
    all_runs = []
    bin_dir = "/".join(bin_remote_path.split("/")[:-1])
    bin_name = bin_remote_path.split("/")[-1]

    for _ in range(repeats):
        cmd = (
            f"cd {bin_dir} && LD_LIBRARY_PATH={bin_dir} ./{bin_name} "
            f"-m {model_remote_path} -p {n_prompt} -n {n_gen} -t {threads}"
        )
        output = dev.shell(cmd)
        all_runs.append(_parse_llama_bench_output(output))

    prefill_vals = [r["prefill_tps"] for r in all_runs if r["prefill_tps"] is not None]
    decode_vals = [r["decode_tps"] for r in all_runs if r["decode_tps"] is not None]

    prefill_tps = round(sum(prefill_vals) / len(prefill_vals), 2) if prefill_vals else None
    decode_tps = round(sum(decode_vals) / len(decode_vals), 2) if decode_vals else None

    ttft_ms = round((n_prompt / prefill_tps) * 1000, 1) if prefill_tps else None

    return {
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "ttft_ms": ttft_ms,
    }

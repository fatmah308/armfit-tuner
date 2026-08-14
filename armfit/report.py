"""
Scores every (model, quant level, build) combination and produces a
recommendation per model, then writes a markdown/JSON report.

Scoring approach (kept simple and transparent on purpose, since a
black-box score is not useful to someone deciding whether to trust
the recommendation):

  1. Any combination whose size exceeds a safety margin of the
     device's free RAM is disqualified from the primary ranking (it
     would risk an OOM kill mid-conversation).
  2. Among the remaining candidates, rank by decode tokens/sec
     (the metric users actually feel), breaking ties by lower
     perplexity delta vs the F16 baseline.
  3. If NOTHING fits within the safety margin, we still rank by real
     decode speed among whatever produced a valid reading, rather than
     falling back to an arbitrary alphabetical pick, and we attach an
     explicit warning that gets surfaced in both the report and the
     CLI output rather than failing silently.

Multi-model runs: results are grouped by "model" and scored
independently per group (RAM fit and perplexity delta are only
meaningful relative to that model's own F16 baseline). The report
additionally computes, per model, which quant level shows the
largest KleidiAI-vs-baseline decode speedup, so that running a
second (or third) model size makes it immediately visible whether
the same optimization pattern holds across model sizes rather than
being a one-off.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RAM_SAFETY_MARGIN = 1.4  # require free_ram >= model_size_mb * this margin


def _fmt(value, suffix: str = "") -> str:
    """None-safe formatter: treats None as 'n/a' regardless of whether
    the key was missing or just empty, unlike dict.get's default."""
    if value is None:
        return "n/a"
    return f"{value}{suffix}"


def _rank_key(r: Dict):
    tps = r["decode_tps"] if r.get("decode_tps") is not None else -1
    ppl = r["ppl_delta"] if r.get("ppl_delta") is not None else 0
    return (-tps, ppl)


def _score_one_model(results: List[Dict], free_ram_mb: int) -> Tuple[Dict, List[Dict]]:
    """Original single-model scoring logic, unchanged, just extracted so
    it can be called once per model in a multi-model run."""
    f16_ppl = next((r["perplexity"] for r in results if r["quant"] == "F16" and r.get("perplexity")), None)

    # Quantization should never meaningfully IMPROVE perplexity, precision
    # loss doesn't make a model better. A negative delta beyond ordinary
    # measurement noise (small sample, sliding-window variance) is a strong
    # signal something upstream mis-measured or mis-parsed, not a real
    # result, flagged here rather than silently reported as fact.
    PPL_IMPROVEMENT_SUSPICION_THRESHOLD = -0.5

    scored = []
    for r in results:
        r = dict(r)
        fits = (r["size_mb"] * RAM_SAFETY_MARGIN) <= free_ram_mb
        r["fits_in_ram"] = fits
        if f16_ppl and r.get("perplexity"):
            r["ppl_delta"] = round(r["perplexity"] - f16_ppl, 4)
            r["quality_suspicious"] = r["ppl_delta"] < PPL_IMPROVEMENT_SUSPICION_THRESHOLD
        else:
            r["ppl_delta"] = None
            r["quality_suspicious"] = False
        scored.append(r)

    fitting = [r for r in scored if r["fits_in_ram"] and r.get("decode_tps") is not None]

    warning: Optional[str] = None
    if fitting:
        candidates = fitting
    else:
        candidates = [r for r in scored if r.get("decode_tps") is not None]
        warning = (
            f"No (quant, build) combination fit within the RAM safety margin "
            f"(size x {RAM_SAFETY_MARGIN} <= {free_ram_mb} MB free). Showing the fastest "
            f"option regardless of fit. Running this on-device risks the OS killing the "
            f"process under memory pressure."
        )

    if not candidates:
        candidates = scored
        warning = warning or (
            "No combination produced a valid decode tok/s reading. Check that "
            "llama-bench's output format matches what benchmark.py expects."
        )

    ranked_candidates = sorted(candidates, key=_rank_key)
    recommendation = dict(ranked_candidates[0])
    if warning:
        recommendation["warning"] = warning

    ranked = sorted(scored, key=lambda r: (r["quant"], r["build"]))
    return recommendation, ranked


def score_and_rank_all(
    results: List[Dict],
    free_ram_mb: int,
) -> Tuple[Dict[str, Dict], List[Dict]]:
    """
    Groups results by model and scores each group independently.
    Returns {model_id: recommendation} and the full combined, sorted
    results list (each row annotated with fits_in_ram / ppl_delta).
    """
    by_model: Dict[str, List[Dict]] = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)

    recommendations: Dict[str, Dict] = {}
    ranked_all: List[Dict] = []
    for model_id, model_results in by_model.items():
        rec, ranked = _score_one_model(model_results, free_ram_mb)
        recommendations[model_id] = rec
        ranked_all.extend(ranked)

    ranked_all.sort(key=lambda r: (r["model"], r["quant"], r["build"]))
    return recommendations, ranked_all


def _best_kleidiai_speedup(model_results: List[Dict]) -> Optional[Tuple[str, float]]:
    """
    For a single model's results, finds the quant level with the largest
    kleidiai-vs-baseline decode speedup. Used to auto-generate a
    cross-model pattern check without hardcoding which quant level to
    compare (different runs may use different --quant-levels).
    """
    by_quant: Dict[str, Dict[str, Dict]] = {}
    for r in model_results:
        by_quant.setdefault(r["quant"], {})[r["build"]] = r

    best = None
    for quant, builds in by_quant.items():
        base = builds.get("baseline")
        kleidiai = builds.get("kleidiai")
        if base and kleidiai and base.get("decode_tps") and kleidiai.get("decode_tps"):
            ratio = kleidiai["decode_tps"] / base["decode_tps"]
            if best is None or ratio > best[1]:
                best = (quant, ratio)
    return best


def write_report(
    output_path: Path,
    device_serial: str,
    free_ram_mb: int,
    ranked: List[Dict],
    recommendations: Dict[str, Dict],
) -> None:
    model_ids = list(recommendations.keys())

    if output_path.suffix == ".json":
        output_path.write_text(json.dumps({
            "models": model_ids,
            "device": device_serial,
            "free_ram_mb": free_ram_mb,
            "results": ranked,
            "recommendations": recommendations,
        }, indent=2), encoding="utf-8")
        return

    lines = [
        "# ArmFit Tuner Report",
        "",
        f"- **Model(s):** {', '.join(model_ids)}",
        f"- **Device:** {device_serial}",
        f"- **Free RAM at test time:** ~{free_ram_mb} MB",
        "",
    ]

    for model_id, rec in recommendations.items():
        if rec.get("warning"):
            lines += [f"> **Warning ({model_id}):** {rec['warning']}", ""]

    has_quality = any(r.get("perplexity") is not None for r in ranked)

    lines += ["## Results", ""]
    if has_quality:
        lines += [
            "| Model | Quant | Build | Size (MB) | Prefill tok/s | Decode tok/s | TTFT (ms) | Perplexity | Δ vs F16 | Fits RAM |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
    else:
        lines += [
            "| Model | Quant | Build | Size (MB) | Prefill tok/s | Decode tok/s | TTFT (ms) | Fits RAM |",
            "|---|---|---|---|---|---|---|---|",
        ]

    any_suspicious = False
    for r in ranked:
        base_cells = (
            f"| {r['model']} | {r['quant']} | {r['build']} | {_fmt(r.get('size_mb'))} | "
            f"{_fmt(r.get('prefill_tps'))} | {_fmt(r.get('decode_tps'))} | {_fmt(r.get('ttft_ms'))} | "
        )
        if has_quality:
            delta_str = _fmt(r.get("ppl_delta"))
            if r.get("quality_suspicious"):
                delta_str = f"⚠ {delta_str}"
                any_suspicious = True
            lines.append(base_cells + f"{_fmt(r.get('perplexity'))} | {delta_str} | {'yes' if r['fits_in_ram'] else 'no'} |")
        else:
            lines.append(base_cells + f"{'yes' if r['fits_in_ram'] else 'no'} |")

    if any_suspicious:
        lines += [
            "",
            "> ⚠ **Rows marked above show quantization apparently IMPROVING perplexity, "
            "which quantization does not do.** This points to a measurement or parsing "
            "issue, not a real result. Do not report these quality numbers until "
            "re-verified against raw `llama-perplexity` output.",
        ]

    lines += ["", "## Recommendation per model", ""]
    for model_id, rec in recommendations.items():
        quality_suffix = f", perplexity delta {_fmt(rec.get('ppl_delta'))} vs F16" if has_quality else ""
        lines.append(
            f"- **{model_id}**: {rec['quant']} quantization, {rec['build']} build — "
            f"{_fmt(rec.get('decode_tps'), ' tok/s decode')}, "
            f"{_fmt(rec.get('size_mb'), ' MB on disk')}"
            f"{quality_suffix}."
        )

    if len(model_ids) > 1:
        lines += ["", "## Cross-model pattern check", ""]
        by_model: Dict[str, List[Dict]] = {}
        for r in ranked:
            by_model.setdefault(r["model"], []).append(r)

        for model_id, model_results in by_model.items():
            best = _best_kleidiai_speedup(model_results)
            if best:
                quant, ratio = best
                lines.append(
                    f"- **{model_id}**: largest KleidiAI speedup at {quant} — "
                    f"{ratio:.2f}x decode tok/s vs the same quant level without KleidiAI"
                )
            else:
                lines.append(f"- **{model_id}**: not enough matching baseline/kleidiai pairs to compare")

        lines += [
            "",
            "If these ratios are in the same ballpark across models, that's evidence the "
            "KleidiAI speedup is a real, repeatable property of the hardware/optimization, "
            "not a one-off result from a single model.",
        ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
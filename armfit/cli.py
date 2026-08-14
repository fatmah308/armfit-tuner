"""
ArmFit Tuner — CLI entry point.

Orchestrates the full pipeline:
  1. Download a Hugging Face model
  2. Convert it to GGUF
  3. Quantize it to multiple levels (F16 / Q8_0 / Q4_0)
  4. Push each variant to a connected Arm Android device
  5. Benchmark speed (tokens/sec, TTFT) using both a generic llama.cpp
     build and a KleidiAI-optimized build
  6. Measure quality drop (perplexity) at each quant level
  7. Score every (quant, build) combination against the device's free
     RAM and produce a recommendation + report

Usage:
    python -m armfit.cli run \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --device <adb-serial> \
        --llama-dir /path/to/llama.cpp \
        --baseline-bin /path/to/llama-cli-baseline \
        --kleidiai-bin /path/to/llama-cli-kleidiai \
        --quant-levels F16,Q8_0,Q4_0 \
        --output report.md
"""

import argparse
import os
import sys
import webbrowser
from pathlib import Path

from armfit import convert, quantize, device, benchmark, quality, report, dashboard


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="armfit",
        description="Benchmark and recommend the best quantization + Arm build for a given model and device.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the full benchmark pipeline")
    run.add_argument("--model", required=True, help="Hugging Face model id, or comma-separated list of ids for a multi-model comparison (e.g. 'Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct')")
    run.add_argument("--device", required=True, help="adb device serial (see `adb devices`)")
    run.add_argument("--llama-dir", required=True, type=Path, help="Path to a cloned llama.cpp repo (for convert/quantize scripts)")
    run.add_argument("--baseline-bin", required=True, type=Path, help="Path to a llama.cpp build WITHOUT KleidiAI enabled")
    run.add_argument("--kleidiai-bin", required=True, type=Path, help="Path to a llama.cpp build WITH GGML_CPU_KLEIDIAI=ON")
    run.add_argument("--quant-levels", default="F16,Q8_0,Q4_0", help="Comma-separated quantization levels to test")
    run.add_argument("--n-prompt", type=int, default=64, help="Prompt tokens for the prefill benchmark")
    run.add_argument("--n-gen", type=int, default=128, help="Tokens to generate for the decode benchmark")
    run.add_argument("--threads", type=int, default=4, help="CPU threads to use on-device")
    run.add_argument("--work-dir", type=Path, default=Path("./armfit_work"), help="Local scratch directory")
    run.add_argument("--remote-dir", default="/data/local/tmp/armfit", help="On-device working directory")
    run.add_argument("--output", type=Path, default=Path("armfit_report.md"), help="Report output path (.md or .json)")
    run.add_argument("--skip-quality", action="store_true", help="Skip perplexity measurement (faster, less rigorous)")
    run.add_argument("--no-browser", action="store_true", help="Don't auto-open the HTML dashboard when the run finishes")

    return parser


def run_pipeline(args: argparse.Namespace) -> int:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    quant_levels = [q.strip().upper() for q in args.quant_levels.split(",") if q.strip()]
    model_ids = [m.strip() for m in args.model.split(",") if m.strip()]

    # Step numbers reflect what's ACTUALLY running this invocation, not a
    # fixed count with a step silently skipped. 5 steps normally, 4 when
    # quality measurement is off, so the printed progress is always accurate
    # rather than announcing something as "skipped".
    total_steps = 5 if args.skip_quality else 6

    def step(n, label):
        print(f"[{n}/{total_steps}] {label}")

    step(1, f"Checking device {args.device} ...")
    dev = device.AdbDevice(serial=args.device, remote_dir=args.remote_dir)
    dev.ensure_ready()
    free_ram_mb = dev.free_ram_mb()
    print(f"      Device reports ~{free_ram_mb} MB free RAM")

    builds = {
        "baseline": args.baseline_bin,
        "kleidiai": args.kleidiai_bin,
    }
    all_results = []

    for model_idx, model_id in enumerate(model_ids, start=1):
        print(f"\n=== Model {model_idx}/{len(model_ids)}: {model_id} ===")

        step(2, f"Downloading + converting model '{model_id}' to GGUF (F16) ...")
        f16_gguf = convert.hf_to_gguf(
            model_id=model_id,
            llama_dir=args.llama_dir,
            work_dir=args.work_dir,
        )

        step(3, f"Quantizing to: {', '.join(quant_levels)} ...")
        quantized_paths = quantize.quantize_all(
            f16_gguf=f16_gguf,
            llama_dir=args.llama_dir,
            quant_levels=quant_levels,
            work_dir=args.work_dir,
        )

        step(4, "Benchmarking each (quant level x build) combination on-device ...")
        for quant_name, gguf_path in quantized_paths.items():
            size_mb = gguf_path.stat().st_size / (1024 * 1024)
            remote_model_path = dev.push_model(gguf_path)

            for build_name, bin_path in builds.items():
                remote_bin_path = dev.push_build(bin_path, build_name)
                print(f"      -> {quant_name} / {build_name} ...")
                bench_result = benchmark.run_llama_bench(
                    dev=dev,
                    bin_remote_path=remote_bin_path,
                    model_remote_path=remote_model_path,
                    n_prompt=args.n_prompt,
                    n_gen=args.n_gen,
                    threads=args.threads,
                )
                all_results.append({
                    "model": model_id,
                    "quant": quant_name,
                    "build": build_name,
                    "size_mb": round(size_mb, 1),
                    **bench_result,
                })

        if not args.skip_quality:
            step(5, "Measuring quality (perplexity) per quant level ...")
            ppl_by_quant = quality.perplexity_all(
                quantized_paths=quantized_paths,
                llama_dir=args.llama_dir,
                work_dir=args.work_dir,
            )
            for r in all_results:
                if r["model"] == model_id:
                    r["perplexity"] = ppl_by_quant.get(r["quant"])
        else:
            for r in all_results:
                if r["model"] == model_id:
                    r["perplexity"] = None

    print()
    step(total_steps, "Scoring combinations and writing report ...")
    recommendations, ranked = report.score_and_rank_all(all_results, free_ram_mb=free_ram_mb)
    report.write_report(
        output_path=args.output,
        device_serial=args.device,
        free_ram_mb=free_ram_mb,
        ranked=ranked,
        recommendations=recommendations,
    )

    dashboard_dir = args.output.parent / (args.output.stem + "_dashboard")
    resolved_path = dashboard.generate_dashboard(
        model_ids=model_ids,
        device_serial=args.device,
        free_ram_mb=free_ram_mb,
        ranked=ranked,
        recommendations=recommendations,
        output_dir=dashboard_dir,
    ).resolve()
    print(f"      Dashboard written to: {resolved_path.parent}")
    print(f"      Open: {resolved_path}")

    if not args.no_browser:
        # webbrowser.open() is known to fail silently on some Windows setups.
        # os.startfile() (Windows-only, stdlib) is the OS-native "open with
        # default app" call and is far more reliable there. Every path is
        # wrapped so a failure here can NEVER crash a finished, successful
        # run, the file already exists at the printed path regardless.
        opened = False
        try:
            if sys.platform == "win32":
                os.startfile(str(resolved_path))
                opened = True
            else:
                opened = webbrowser.open(resolved_path.as_uri())
        except Exception as e:
            print(f"      Auto-open failed ({e}).")
        if not opened:
            print(f"      Open it manually: double-click {resolved_path}")

    print()
    print("=" * 60)
    for model_id, recommendation in recommendations.items():
        decode_str = (
            f"{recommendation['decode_tps']:.1f}" if recommendation.get("decode_tps") is not None else "n/a"
        )
        ppl_delta_str = (
            recommendation["ppl_delta"] if recommendation.get("ppl_delta") is not None else "n/a"
        )
        print(f"{model_id}: {recommendation['quant']} / {recommendation['build']}")
        print(f"  {decode_str} tok/s decode, "
              f"{recommendation['size_mb']:.0f} MB, "
              f"perplexity delta {ppl_delta_str}")
        if recommendation.get("warning"):
            print(f"  WARNING: {recommendation['warning']}")
    print(f"Full report written to {args.output}")
    print("=" * 60)
    return 0


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        try:
            return run_pipeline(args)
        except device.DeviceError as e:
            print(f"[armfit] Device error: {e}", file=sys.stderr)
            return 1
        except (convert.ConvertError, quantize.QuantizeError) as e:
            print(f"[armfit] Model prep error: {e}", file=sys.stderr)
            return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
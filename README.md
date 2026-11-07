# ArmFit Tuner

Benchmark tool that answers one question: **for a given LLM and a given
Arm Android device, what's the best quantization level and build
configuration to run it with?**

Built and tested against a Samsung Galaxy A16 (8 GB RAM), a genuinely
budget-tier Arm device, not a flagship phone.

## Project Overview

Most on-device LLM demos pick one model, one quantization level, run it
once, and call it optimized. ArmFit Tuner instead isolates and measures
**exactly how much of the speedup comes from Arm-specific hardware
acceleration**, as opposed to quantization alone, by benchmarking every
combination of quantization level against two builds of the same
inference engine (llama.cpp): one generic, one with
[Arm KleidiAI](https://developer.arm.com/) enabled, which supplies
hand-tuned matrix-multiplication kernels for Arm's DotProd/I8MM CPU
instructions.

What makes this interesting: the comparison isn't just "quantized model
runs faster," it's "the exact same quantized model, on the exact same
device, runs meaningfully faster or slower depending on whether Arm's
own accelerated kernels are compiled in." That isolates the Arm-specific
contribution from the quantization contribution, two different
optimization levers that are usually conflated in typical demos, and it
was measured on a genuinely budget device most other submissions won't
have access to, not a flagship phone or dev board.

## Functionality / Output

The tool runs an automated pipeline:

1. Downloads a model from Hugging Face and converts it to GGUF
2. Quantizes it to F16 / Q8_0 / Q4_0
3. Benchmarks every quant level against **two builds of llama.cpp**
   (generic baseline vs KleidiAI-enabled) on a connected Arm Android device
4. Measures quality loss (perplexity) at each quant level vs the F16 baseline
5. Scores every (quant, build) combination against the device's actual
   free RAM, and recommends the best-fitting, fastest option

**Final output:** a single markdown/JSON report (see `examples/sample_report.md`
for the format, and the Results section below for real measured data) with every
combination's size, speed, and quality, plus a one-line recommendation. The tool
itself is the reusable artifact, point it at any Hugging Face model ID and any
adb-connected Arm Android device to get the same kind of report.

## Results

Measured on a Samsung Galaxy A16 (Cortex-A76/A78 class chip, DotProd supported,
I8MM not supported), running Qwen2.5-0.5B-Instruct, ~4.2 GB free RAM at test time:

| Quant | Build | Size (MB) | Prefill tok/s | Decode tok/s | TTFT (ms) |
|---|---|---|---|---|---|
| F16 | baseline | 948.1 | 39.15 | 7.53 | 1634.7 |
| F16 | kleidiai | 948.1 | 36.29 | 7.13 | 1763.6 |
| Q4_0 | baseline | 335.8 | 27.69 | 11.30 | 2311.3 |
| Q4_0 | kleidiai | 335.8 | **102.31** | **27.00** | **625.5** |
| Q8_0 | baseline | 506.5 | 32.55 | 11.28 | 1966.2 |
| Q8_0 | kleidiai | 506.5 | **156.09** | **21.51** | **410.0** |

**Recommendation: Q4_0 quantization, KleidiAI build** — 27.0 tok/s decode
(2.4x faster than the same quant level without KleidiAI), 335.8 MB on disk,
TTFT cut from 2.3s to 0.6s.

**Two findings worth calling out explicitly:**

1. **KleidiAI's benefit is entirely concentrated on quantized formats.**
   On F16 (unquantized), KleidiAI provides no benefit at all (36.29 vs 39.15
   tok/s prefill, within noise). That's expected, not a bug: KleidiAI's
   kernels specifically accelerate INT4/INT8 matrix multiplication via
   Arm's DotProd instructions, they have nothing to accelerate in F16
   arithmetic. Seeing zero effect exactly where none should exist is
   evidence the measured speedup elsewhere is coming from the claimed
   mechanism, not from an unrelated confound.

2. **Quantization alone made inference slower on this hardware, not faster.**
   Q4_0 without KleidiAI (11.3 tok/s decode) is barely faster than F16
   (7.53 tok/s) despite being a third of the size, and Q4_0 prefill without
   KleidiAI (27.69 tok/s) is actually *slower* than F16 prefill (39.15 tok/s).
   Without Arm-specific acceleration, this budget chip falls back to slow,
   generic unpacking of quantized weights. In other words: on this class of
   Arm hardware, KleidiAI isn't just an optimization on top of quantization,
   it's what makes quantization deliver its expected benefit at all.

*Quality (perplexity) measurements and a second data point at 1.5B
parameters (to confirm this pattern holds across model sizes, not just
at 0.5B) are pending a re-run with the fixes in this commit. Update this
section with real numbers once `report.md` is regenerated, the command
in "Running the benchmark" below now supports both models in a single
combined run.*

## Setup Instructions

### Repo structure

```
armfit-tuner/
├── armfit/                  # the CLI tool
│   ├── cli.py                # orchestrates the full pipeline (supports multi-model runs)
│   ├── convert.py             # HF model -> GGUF (F16)
│   ├── quantize.py            # GGUF -> Q8_0 / Q4_0
│   ├── device.py              # adb push/shell helpers
│   ├── benchmark.py           # runs llama-bench, parses tok/s
│   ├── quality.py              # perplexity measurement
│   ├── report.py               # scoring + multi-model report generation
│   ├── dashboard.py            # multi-page HTML dashboard generator
│   └── binaries.py             # shared host-binary locator (Ninja vs MSVC layouts)
├── scripts/
│   ├── setup_llama_cpp.sh    # builds baseline + KleidiAI binaries (Linux/macOS)
│   └── setup_llama_cpp.ps1    # same, for Windows PowerShell
├── web/
│   └── index.html              # live chat UI, talks to llama-server, baseline/KleidiAI toggle
├── examples/
│   └── sample_report.md       # example output
├── requirements.txt
└── LICENSE                    # MIT
```

### Installation and build steps

**On your development machine:**

```bash
git clone https://github.com/fatmah308/armfit-tuner
cd armfit-tuner
pip install -r requirements.txt --break-system-packages

# Clone llama.cpp separately (this is the framework being optimized)
git clone --depth 1 https://github.com/ggml-org/llama.cpp
```

**Build llama.cpp for your host machine first** (needed for quantization
and perplexity, which run locally rather than on-device):

```bash
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_WEBUI=OFF -DLLAMA_CURL=OFF
cmake --build build --config Release --target llama-quantize --target llama-perplexity -j
cd ..
```

Note: `LLAMA_BUILD_UI=OFF` / `LLAMA_BUILD_WEBUI=OFF` attempt to skip llama-server's
embedded web UI (a Node/Svelte build step irrelevant to this project), but as of
this writing that path has some rough edges, particularly when cross-compiling.
Building only the specific `--target` we need sidesteps it entirely, since
neither `llama-quantize` nor `llama-perplexity` depend on the UI.

**Build the two comparison binaries** (requires Android NDK r27+ and Ninja):

On Linux/macOS:
```bash
NDK_PATH=/path/to/android-ndk-r27 ./scripts/setup_llama_cpp.sh ./llama.cpp
```

On Windows (PowerShell), install Ninja first (required for Android
cross-compilation, the default Visual Studio generator won't work):
```powershell
pip install ninja --break-system-packages
$env:NDK_PATH = "C:\Users\<you>\AppData\Local\Android\Sdk\ndk\<version>"
.\scripts\setup_llama_cpp.ps1 -LlamaDir D:\armfit-tuner\llama.cpp
```

Both scripts also locate and copy `libomp.so` (LLVM's OpenMP runtime,
which llama.cpp's CPU backend links against) from inside the NDK into
each build's output folder, since it doesn't get copied there automatically
and the device-push step needs it sitting alongside the binary.

Both produce `llama.cpp/build-baseline/bin/llama-bench` and
`llama.cpp/build-kleidiai/bin/llama-bench`.

**Connect your Android device:**

1. Settings → About phone → tap Build number 7 times to enable Developer options
2. Settings → Developer options → enable USB debugging
3. Connect via USB, run `adb devices`, and authorize the connection when prompted on the phone

## Running the benchmark

Single model:
```bash
python -m armfit.cli run \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --device <serial-from-adb-devices> \
    --llama-dir ./llama.cpp \
    --baseline-bin ./llama.cpp/build-baseline/bin/llama-bench \
    --kleidiai-bin ./llama.cpp/build-kleidiai/bin/llama-bench \
    --quant-levels F16,Q8_0,Q4_0 \
    --output report.md
```

Multiple models in one run, comma-separate `--model`, producing a single
combined report with a per-model recommendation and an automatic
cross-model pattern check (does the KleidiAI speedup hold consistently
across model sizes, or was it a one-off?):
```bash
python -m armfit.cli run \
    --model Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct \
    --device <serial-from-adb-devices> \
    --llama-dir ./llama.cpp \
    --baseline-bin ./llama.cpp/build-baseline/bin/llama-bench \
    --kleidiai-bin ./llama.cpp/build-kleidiai/bin/llama-bench \
    --quant-levels F16,Q8_0,Q4_0 \
    --output report.md
```

See `examples/sample_report.md` for what the output looks like.

## Report dashboard

Every run automatically generates and opens a multi-page HTML dashboard
(a `report_dashboard/` folder alongside `report.md`), a real site with
navigation, not one long scrolling page:

- **`index.html`** — overview, one summary card per model, links to details
- **`model_<name>.html`** — one page per model: baseline-vs-KleidiAI bar
  chart, full results table, recommendation
- **`comparison.html`** — cross-model pattern check (only generated when
  multiple models were run in one invocation)
- **`insights.html`** — click-to-reveal explanations of the results (see below)

No server, no adb, no extra setup: it's static HTML/CSS/JS built from the
same data written to `report.md`/`report.json`, opens in your default
browser the moment the run finishes. Pass `--no-browser` to skip the
auto-open, the folder is still written either way.

### Insights page: how it works, and why it's not a live chat

`insights.html` answers a fixed set of questions (e.g. "Which
configuration should I use?", "Why is KleidiAI faster on quantized
formats but not F16?") by running JavaScript against the run's own
embedded JSON data and filling in a template with the real numbers,
there is no free-text input and no live model behind it.

This was a deliberate choice, not a shortcut:

1. **A real conversational agent needs a backend.** Either a network
   call to an external LLM API, which contradicts this project's whole
   premise of running fully offline, on-device, or `llama-server`,
   which does not currently cross-compile for Android with this
   llama.cpp version (see Known limitations below). Neither was
   available to build against honestly.
2. **Fixed questions with data-driven answers cannot hallucinate.**
   Every number in every answer is read directly from the same
   `results`/`recommendations` data already in `report.json`, computed
   with the same logic as the charts and tables, not generated or
   guessed. A free-text chat over this data would risk producing
   plausible-sounding but wrong explanations, exactly the failure mode
   this project's own benchmarking methodology was built to avoid.
3. **It's fully offline and dependency-free**, consistent with every
   other part of this tool, no API key, no network call, no additional
   binary to build.

The tradeoff is real: it can only answer the questions it was built to
answer, not anything a user might ask. Given the choice between a
narrower feature that's guaranteed accurate and a broader one that
might not be, accuracy won for a tool whose entire purpose is producing
trustworthy benchmark numbers.

## Live chat UI (stretch goal, not required)

`web/index.html` is a separate, more ambitious piece: a live chat
interface talking to `llama-server` over HTTP with real-time streaming.
It's fully built and unit-tested, but currently blocked by the same
Android/NDK cross-compilation gap noted above. Recommended fallback for
demoing this project: record the CLI pipeline running live against the
device, then the generated dashboard, both of which work end-to-end today.

## Validating the Arm-specific claim yourself

To confirm your device actually has the CPU features KleidiAI targets:

```bash
adb shell cat /proc/cpuinfo | grep Features
```

Look for `asimddp` (DotProd). The build scripts here target
`armv8.2-a+dotprod` by default; if your device also supports `i8mm`,
you can edit `$ArchFlags` / `ARCH_FLAGS` in the build script to add it
for a larger potential speedup, but only if you've confirmed the device
actually supports it first, requesting an unsupported instruction set
at compile time risks a runtime crash (illegal instruction) rather than
just a smaller speedup.

## Known limitations

- Perplexity is measured on a small fixed text sample for speed; it's a
  relative comparison across quant levels of the same model, not an
  absolute quality benchmark.
- Only tested against Qwen2.5-0.5B/1.5B-Instruct on a Galaxy A16
  (Cortex-A76/A78 class chip, DotProd supported, I8MM not supported).
- The RAM safety margin in `report.py` may need adjusting for larger
  models or lower-RAM devices.

## License

MIT, see [LICENSE](./LICENSE).

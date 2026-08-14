# ArmFit Tuner Report (example output — illustrative, not measured)

- **Model:** Qwen/Qwen2.5-1.5B-Instruct
- **Device:** R58M-EXAMPLE (Samsung Galaxy A16)
- **Free RAM at test time:** ~3800 MB

## Results

| Quant | Build | Size (MB) | Prefill tok/s | Decode tok/s | TTFT (ms) | Perplexity | Δ vs F16 | Fits RAM |
|---|---|---|---|---|---|---|---|---|
| F16 | baseline | 3100 | 12.4 | 4.1 | 850 | 6.21 | 0.0 | no |
| F16 | kleidiai | 3100 | 12.9 | 4.3 | 810 | 6.21 | 0.0 | no |
| Q8_0 | baseline | 1650 | 19.8 | 7.9 | 520 | 6.27 | 0.06 | yes |
| Q8_0 | kleidiai | 1650 | 26.4 | 10.2 | 410 | 6.27 | 0.06 | yes |
| Q4_0 | baseline | 950 | 27.1 | 12.6 | 340 | 6.58 | 0.37 | yes |
| Q4_0 | kleidiai | 950 | 41.3 | 19.8 | 230 | 6.58 | 0.37 | yes |

## Recommendation

**Q4_0 quantization, kleidiai build** — 19.8 tok/s decode, 950 MB on
disk, perplexity delta 0.37 vs F16.

This combination gave the highest decode speed while fitting safely
within the device's free RAM at test time.

---
*Note: the numbers above are illustrative placeholders showing the
report format, not real measurements. Run the pipeline on your own
device to generate actual results.*

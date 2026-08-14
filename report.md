# ArmFit Tuner Report

- **Model(s):** Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B-Instruct
- **Device:** RF8XB0WQZ3J
- **Free RAM at test time:** ~4325 MB

## Results

| Model | Quant | Build | Size (MB) | Prefill tok/s | Decode tok/s | TTFT (ms) | Fits RAM |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | F16 | baseline | 948.1 | 38.06 | 7.56 | 1681.6 | yes |
| Qwen/Qwen2.5-0.5B-Instruct | F16 | kleidiai | 948.1 | 37.58 | 7.47 | 1703.0 | yes |
| Qwen/Qwen2.5-0.5B-Instruct | Q4_0 | baseline | 335.8 | 21.95 | 12.35 | 2915.7 | yes |
| Qwen/Qwen2.5-0.5B-Instruct | Q4_0 | kleidiai | 335.8 | 87.47 | 19.3 | 731.7 | yes |
| Qwen/Qwen2.5-0.5B-Instruct | Q8_0 | baseline | 506.5 | 33.18 | 11.47 | 1928.9 | yes |
| Qwen/Qwen2.5-0.5B-Instruct | Q8_0 | kleidiai | 506.5 | 155.18 | 17.32 | 412.4 | yes |
| Qwen/Qwen2.5-1.5B-Instruct | F16 | baseline | 2950.4 | 8.81 | 2.57 | 7264.5 | yes |
| Qwen/Qwen2.5-1.5B-Instruct | F16 | kleidiai | 2950.4 | 11.02 | 2.74 | 5807.6 | yes |
| Qwen/Qwen2.5-1.5B-Instruct | Q4_0 | baseline | 891.6 | 7.42 | 4.98 | 8625.3 | yes |
| Qwen/Qwen2.5-1.5B-Instruct | Q4_0 | kleidiai | 891.6 | 33.88 | 9.27 | 1889.0 | yes |
| Qwen/Qwen2.5-1.5B-Instruct | Q8_0 | baseline | 1570.3 | 9.37 | 5.23 | 6830.3 | yes |
| Qwen/Qwen2.5-1.5B-Instruct | Q8_0 | kleidiai | 1570.3 | 50.1 | 5.94 | 1277.4 | yes |

## Recommendation per model

- **Qwen/Qwen2.5-0.5B-Instruct**: Q4_0 quantization, kleidiai build — 19.3 tok/s decode, 335.8 MB on disk.
- **Qwen/Qwen2.5-1.5B-Instruct**: Q4_0 quantization, kleidiai build — 9.27 tok/s decode, 891.6 MB on disk.

## Cross-model pattern check

- **Qwen/Qwen2.5-0.5B-Instruct**: largest KleidiAI speedup at Q4_0 — 1.56x decode tok/s vs the same quant level without KleidiAI
- **Qwen/Qwen2.5-1.5B-Instruct**: largest KleidiAI speedup at Q4_0 — 1.86x decode tok/s vs the same quant level without KleidiAI

If these ratios are in the same ballpark across models, that's evidence the KleidiAI speedup is a real, repeatable property of the hardware/optimization, not a one-off result from a single model.
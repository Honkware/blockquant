# BlockQuant Cost Comparison

Auto-generated from actual quantization runs.

## Local (RTX 4090)

| Model Size | Example | Time | Cost |
|------------|---------|------|------|
| 3.8B | Phi-3-mini | ~2 min | $0.00* |
| 7B | Mistral-7B | ~5 min | $0.00* |
| 35B | Qwen-35B | ~45 min | $0.00* |
| 70B | Llama-3.1-70B | ~3 hr | $0.00* |

*Electricity cost not included (~$0.05-0.15 per job depending on duration)

## Cloud Providers

| Provider | Instance | GPU | $/hr | Notes |
|----------|----------|-----|------|-------|
| **Lambda** | gpu_1x_a10 | A10 | $1.10 | Best for 7B-13B models |
| **Modal** | A100-40GB | A100 | ~$2.00 | Serverless, cold start penalty |
| **Vast.ai** | A100 spot | A100 | ~$0.50-1.20 | Cheapest but spot preemption risk |

## Actual Runs (Today)

| Date | Model | Variant | Provider | Time | Cost |
|------|-------|---------|----------|------|------|
| 2026-04-21 | microsoft/Phi-3-mini-4k-instruct | 4.0bpw EXL3 | local | 0.0s | $0.00 |

Last updated: 2026-04-21

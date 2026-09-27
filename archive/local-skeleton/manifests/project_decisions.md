# Project Decisions

Last updated: 2026-09-14

| Item | Decision |
|---|---|
| Base model | `Qwen/Qwen3.5-9B`, native multimodal model |
| Post-training framework | Alibaba ModelScope `ms-swift`, pinned source commit `9d3d03dd35c8a60e519e3165962586837cea67ee` |
| Primary teacher candidate | DashScope `deepseek-v4.1-flash`, subject to blind teacher-selection review |
| Audit/secondary teacher | DashScope `qwen3.5-397b-a17b` |
| Data | Public WM811K; company source files are unavailable |
| Main outputs | Chinese caption, English caption, classification, and structured JSON |
| Initial tuning | QLoRA/LoRA with frozen ViT and aligner |
| Benchmark | Test-lot-only `wafer_bench_v1`, two-person approval required before official freeze |
| Retrieval | Evaluated separately; generative SFT is not claimed to optimize embeddings |

## Pending decisions before expensive runs

- Exact GPU type/count/VRAM after the AutoDL instance is switched on.
- Whether mentor permits QLoRA as the primary limited-memory baseline.
- Whether merged full weights are a required deliverable.
- Names of the two benchmark reviewers and the review record format approved by the team.
- Final priority ordering among caption, classification, structured understanding, and retrieval.

# Run comparison

```
| metric | Base | SFT | GRPO_lr1e5 | GRPO_lr5e5 | GRPO_lr1e5_s2 | GRPO_lr1e5_s3 | GSPO_lr5e5 | GSPO_lr1e5 | GSPO_G32 | GSPO_G32_lr1e5 | GSPO_G4_lr5e5 | GSPO_G4_lr1e5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification accuracy | 0.2143 | 0.6230 | 0.6230 | 0.5675 | 0.6071 | 0.6429 | 0.5397 | 0.6151 | 0.5675 | 0.6190 | 0.5357 | 0.6270 |
| classification macro-F1 | 0.1347 | 0.6114 | 0.6197 | 0.5535 | 0.5934 | 0.6211 | 0.5014 | 0.6089 | 0.5403 | 0.6143 | 0.4910 | 0.6255 |
| macro-F1 95% CI | [0.0955, 0.1706] | [0.5496, 0.6602] | [0.5554, 0.6704] | [0.4956, 0.6110] | [0.5310, 0.6444] | [0.5614, 0.6717] | [0.4367, 0.5537] | [0.5472, 0.6613] | [0.4807, 0.5965] | [0.5492, 0.6699] | [0.4324, 0.5483] | [0.5643, 0.6746] |
| off-vocabulary answers | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| structured defect_type acc | 0.2103 | 0.5840 | 0.6032 | 0.5516 | 0.5800 | 0.5873 | 0.5635 | 0.5595 | 0.5595 | 0.5714 | 0.5437 | 0.5794 |
| structured radial_zone acc | 0.0794 | 0.2640 | 0.2698 | 0.3452 | 0.2760 | 0.2738 | 0.3056 | 0.2460 | 0.4603 | 0.2698 | 0.3254 | 0.2619 |
| clock circular MAE (sectors) | 5.0000 | 1.6847 | 1.7121 | 1.9730 | 1.7574 | 1.7143 | 1.9519 | 1.7098 | 1.6860 | 1.7026 | 1.6761 | 1.7463 |
| size MAE (R) | 1.2139 | 0.4999 | 0.5695 | 0.9296 | 0.7211 | 0.6134 | 0.6927 | 0.6203 | 1.6509 | 0.7085 | 0.4025 | 0.8384 |
| caption must-hit rate | 0.6270 | 0.7540 | 0.7222 | 0.5516 | 0.7024 | 0.6310 | 0.6627 | 0.7143 | 0.5714 | 0.7063 | 0.6349 | 0.7381 |
| caption empty rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| root-cause hallucination rate | 0.2183 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| robustness accuracy | 0.1905 | 0.5622 | 0.5661 | 0.5251 | 0.5489 | 0.5820 | 0.5146 | 0.5688 | 0.5556 | 0.5688 | 0.5093 | 0.5807 |
| flip rate vs clean | 0.1389 | 0.2474 | 0.2672 | 0.2619 | 0.2619 | 0.2500 | 0.2354 | 0.2487 | 0.2540 | 0.2606 | 0.2553 | 0.2632 |
| retrieval mAP@10 | 0.3236 | 0.3873 | 0.3742 | 0.4260 | 0.3887 | 0.3943 | 0.3607 | 0.3782 | 0.4337 | 0.3702 | 0.4138 | 0.3748 |
| retrieval nDCG@10 | 0.2978 | 0.3563 | 0.3497 | 0.3917 | 0.3566 | 0.3605 | 0.3253 | 0.3437 | 0.3906 | 0.3431 | 0.3804 | 0.3481 |
| retrieval Recall@10 | 0.1564 | 0.1836 | 0.1794 | 0.2022 | 0.1817 | 0.1845 | 0.1788 | 0.1796 | 0.2008 | 0.1773 | 0.1928 | 0.1780 |
```

Metrics reported as `not run` were never measured; they are not zeros.
- Metrics shown as not run were never measured; they are not zeros.
- The run set was enumerated from the reports on disk by tools/run_set.py, not from a hand-written list; the enumeration is appended to this file as provenance and any report on disk that is in neither its table nor its exclusions would have failed that step.

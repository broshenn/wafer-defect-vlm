# Trivial baselines (floors)

Benchmark core: 252 samples. Majority class: `Edge_Loc` (30/252 = 0.1190).

- always-`Edge_Loc`: accuracy 0.1190, macro-F1 0.0236 (95% CI 0.0163–0.0298)
- uniform random over 9 labels: accuracy 0.1230 ± 0.0144, macro-F1 0.1211 (over seeds [3407, 1, 2, 3, 4])
- random ranking: mAP@10 0.0330, nDCG@10 0.0596, Recall@10 0.0394

Bootstrap CIs elsewhere in this project use the same resampling code as the model scores.

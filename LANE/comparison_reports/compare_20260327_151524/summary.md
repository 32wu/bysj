# LANE Baseline vs Optimized Comparison

## Experiment Mapping
- baseline: `ppo_gymip_rwtaspk_h8-8-40_none_rmsprop_0.001000_5.00_0.97000_5_0.2000_rep11_best`
- optimized: `ppo_gymip_rwtaspk_h8-8-40_none_rmsprop_0.000500_2.00_0.99500_4_0.1500_ro512_mb128_lam0.97_rs1.00_gc0.50_adaptive_seed06_best`

## Training Log Summary
| Model | Log Type | Best Val Episode | Best Val Score | Best Val Length | Best Collision | Final Train Episode | Final Train Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | baseline | 199 | 38.800 | - | - | 349 | 9.000 |
| optimized | improved | 124 | 206.990 | 150.000 | 0.000 | 170 | -0.722 |

## Clean Evaluation
| Scenario | Traffic | Model | Return | Length | Collision | Success | Lane Changes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| highway | standard | baseline | -25.661 | 11.000 | 1.000 | 0.000 | 11.000 |
| highway | standard | optimized | 198.569 | 150.000 | 0.000 | 1.000 | 0.000 |

## Action Failure Robustness
| Failure Rate | Model | Return | Length | Collision | Success |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.000 | baseline | -26.962 | 13.000 | 1.000 | 0.000 |
| 0.000 | optimized | 209.095 | 150.000 | 0.000 | 1.000 |

## Input Noise Robustness
| Noise Std | Model | Return | Length | Collision | Success |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.000 | baseline | -33.760 | 24.000 | 1.000 | 0.000 |
| 0.000 | optimized | 193.306 | 150.000 | 0.000 | 1.000 |

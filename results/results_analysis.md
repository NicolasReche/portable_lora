# Global Evaluation Report: Sentiment Control

This document consolidates the results of all configurations (Identity, Zero-Shot, Few-Step) for the different LLaMA models, and narrates the technical evolution of the RL algorithm (V1 to V4.2).

---

## 1. The Reinforcement Learning Saga: Evolution of Algorithms

The construction of the algorithm followed an iterative evolution to correct successive learning flaws. Here is the history of these algorithms (analyzed on LLaMA 3.1):

* **V1 (The Baseline)**: Initial approach (Raw control). Maximizes sentiment (PPLM 89.29%) but totally sacrifices diversity (`Dist-n` drops to 0.472). The model cheats by repeating words.
* **V2 (Mode Collapse)**: Strengthens the control penalty. Achieves excellent scores but causes a "Mode Collapse" (loops on a single syntax, diversity at 0.443).
* **V2.1 (The Transition)**: Normalizes the control score (from V2) to soften the penalty, but does not really solve the loss of diversity.
* **V3 (The "Sweet Spot" - without dict)**: Introduces the `Distinct-n` penalty with normalized control. Masterfully restores diversity (0.606). (NB: SLOR used a uniform probability by mistake).
* **V3.1 (Fluid Balancing - without dict)**: Identical to V3 but with V2's raw control. Improves natural fluency. (NB: SLOR still uniform).
* **V3.2 & V3.3 (The True SLOR)**: Respective re-runs of V3 and V3.1, but integrating the true empirical probability dictionary (`llama3_unigram_log_probs`) for linguistically correct learning.
* **V4 (The Mathematical Error)**: Replaces `Distinct-n` with Shannon Entropy on *unigrams*. Forces the vocabulary to be uniform, breaking Zipf's law and destroying fluency (SLOR at 7.85).
* **V4.1 & V4.2 (Entropy Repair)**: Fixes V4 by applying Entropy on *trigrams and 4-grams*. V4.1 uses normalized control (like V3.2), and V4.2 uses raw control (like V3.3).

---

## 2. "Identity" Performances (Without Transfer)

These tables show the scores when the module is applied to the model it was trained with.

### Target: LLaMA 3.1 8B → LLaMA 3.1 8B
| Version | Avg All | PPLM (Control) | Dist-n (Diversity) | SLOR (Fluency) |
| --- | --- | --- | --- | --- |
| **Base (no module)** | 60.90 | 65.71 | 0.647 | 9.73 |
| **SFT (Baseline)** | 87.63 | 92.14 | 0.638 | 9.82 |
| **RL V1** | 83.43 | 89.29 | 0.472 | 8.77 |
| **RL V2** | 90.61 | 93.57 | 0.443 | 8.39 |
| **RL V2.1** | 80.64 | 86.43 | 0.476 | 8.23 |
| **RL V3** | 91.87 | 97.86 | 0.606 | 8.28 |
| **RL V3.1** | 96.02 | 98.57 | 0.601 | 9.35 |
| **RL V4** | 82.54 | 97.86 | 0.619 | 7.85 |

### Target: LLaMA 3.2 3B → LLaMA 3.2 3B

| Version | Avg All | PPLM (Control) | Dist-n (Diversity) | SLOR (Fluency) |
| --- | --- | --- | --- | --- |
| **Base (no module)** | 64.07 | 70.71 | 0.647 | 9.81 |
| **SFT (Baseline)** | 80.89 | 81.43 | 0.601 | 10.11 |
| **RL V1** | 72.36 | 78.57 | 0.425 | 8.93 |
| **RL V2** | 96.39 | 96.43 | 0.617 | 9.90 |
| **RL V2.1** | 83.54 | 82.86 | 0.294 | 8.82 |
| **RL V3** | 92.49 | 95.71 | 0.575 | 9.15 |
| **RL V3.1** | 86.81 | 91.43 | 0.546 | 9.37 |
| **RL V4** | 85.23 | 91.43 | 0.578 | 8.83 |

---

## 3. The Failure of "Zero-Shot" (Raw Transfer)

Direct injection of a module from one architecture to another yields no conclusive results: the performances align with the host model's base behavior.

### Transfer: LLaMA 3.1 8B → LLaMA 3.2 3B (Zero-Shot)
| Version | Avg All | PPLM (Control) | Dist-n (Diversity) | SLOR (Fluency) |
| --- | --- | --- | --- | --- |
| **Zero-Shot SFT** | 64.83 | 70.00 | 0.653 | 9.77 |
| **Zero-Shot RL V1** | 63.15 | 70.71 | 0.634 | 9.82 |
| **Zero-Shot RL V2** | 64.06 | 71.43 | 0.652 | 9.80 |
| **Zero-Shot RL V2.1** | 64.85 | 69.29 | 0.638 | 9.83 |
| **Zero-Shot RL V3** | 64.14 | 71.43 | 0.649 | 9.79 |
| **Zero-Shot RL V3.1** | 64.10 | 69.29 | 0.638 | 9.77 |
| **Zero-Shot RL V4** | 64.39 | 71.43 | 0.650 | 9.78 |

### Transfer: LLaMA 3.2 3B → LLaMA 3.1 8B (Zero-Shot)
| Version | Avg All | PPLM (Control) | Dist-n (Diversity) | SLOR (Fluency) |
| --- | --- | --- | --- | --- |
| **Zero-Shot RL V1** | 57.90 | 60.71 | 0.647 | 9.73 |
| **Zero-Shot RL V2** | 60.17 | 65.00 | 0.659 | 9.78 |
| **Zero-Shot RL V2.1** | 59.67 | 65.00 | 0.632 | 9.76 |
| **Zero-Shot RL V3** | 59.58 | 65.00 | 0.644 | 9.78 |
| **Zero-Shot RL V3.1** | 58.93 | 64.29 | 0.627 | 9.77 |
| **Zero-Shot RL V4** | 59.79 | 62.86 | 0.623 | 9.78 |

---

## 4. The Rescue by "Few-Step Adaptation"

*(Results re-evaluated with strict calibration to a 10% data budget)*

### Transfer: LLaMA 3.1 8B → LLaMA 3.2 3B (Few-Step)
| Version | Avg All | PPLM (Control) | Dist-n (Diversity) | SLOR (Fluency) |
| --- | --- | --- | --- | --- |
| **Few-Step SFT** | 77.34 | 74.29 | 0.542 | 9.66 |
| **Few-Step RL V2** | 82.10 | 83.57 | 0.546 | 9.64 |
| **Few-Step RL V2.1** | 77.94 | 73.57 | 0.547 | 9.69 |
| **Few-Step RL V3** | 79.43 | 84.29 | 0.544 | 9.69 |
| **Few-Step RL V4** | 78.87 | 82.86 | 0.543 | 9.67 |

### Transfer: LLaMA 3.2 3B → LLaMA 3.1 8B (Few-Step)
| Version | Avg All | PPLM (Control) | Dist-n (Diversity) | SLOR (Fluency) |
| --- | --- | --- | --- | --- |
| **Few-Step SFT** | 84.46 | 87.14 | 0.572 | 9.83 |
| **Few-Step RL V1** | 82.71 | 87.14 | 0.566 | 9.81 |
| **Few-Step RL V2** | 82.51 | 84.29 | 0.572 | 9.81 |
| **Few-Step RL V3** | 81.32 | 85.71 | 0.568 | 9.80 |
| **Few-Step RL V3.1** | 78.46 | 82.14 | 0.568 | 9.82 |
| **Few-Step RL V4** | 84.22 | 86.43 | 0.574 | 9.82 |

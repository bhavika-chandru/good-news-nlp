# Project notes and decision log

## Data
- Upworthy Research Archive, exploratory file only. Confirmatory file kept untouched for the final evaluation.
- The 03.12.2020 release has no `problem` column, so tests created between 25 Jun 2013 and 10 Jan 2014 (possibly broken randomisation) are dropped by date: 4,278 packages, 1,050 tests.
- Kept only tests where every package used the same image, so CTR differences come from the headline.
- Dropped repeated headlines within a test (image-only variants): 7,586 rows.
- `excerpt` is mostly a site-wide tagline, so not used. `lede` (HTML stripped) is kept as the story text for the faithfulness check. It was not shown to readers during the tests.
- Pairs kept only if the two-proportion z-test gives p < 0.05: 4,588 pairs in 1,289 tests.

## Leakage and reproducibility
- Leak found: Upworthy re-tested some stories in new tests, so splitting by test let 67 validation pairs share a story with training. Fixed by linking tests that share a headline or story text and splitting by story group. Now zero overlap.
- Non-determinism found: tied values were ordered differently in pandas 3.0.5 and 3.1.1, which changed the split. Fixed with stable sorts and tie-breaks. A unit test checks that shuffling the input rows gives the same split.
- Final split (significant pairs): train 3,212, val 648, test 728.
- Data files (CSV, embedding cache) are kept out of git: `.gitignore` excludes everything under `data/raw` and `data/processed`.

## Baselines
- Negativity rule (VADER): ties (equal negativity) in about 37% of pairs, broken by a seeded coin flip. Val accuracy 0.511, non-tied 0.524. On train+val (3,860 pairs, 2,556 non-tied), non-tied accuracy 0.549, 95% CI [0.522, 0.578] (bootstrap over story groups): a small but real negativity effect.
- TF-IDF + logistic regression on the difference vector, no intercept (swapping A and B flips the prediction). Best C = 1, val accuracy 0.634.
- Words linked to winning: she, her, response, seen, tell, did. Linked to losing: your, help, save, rights, we, family, love. Hypothesis: advocacy and appeal language loses, narrative and curiosity language wins.

## Model A: frozen embeddings
- bge-small embeddings + logistic regression on the difference vector: best C = 0.01 (peak inside the tested range), val accuracy 0.713.

## Model B: DeBERTa-v3-base cross-encoder
- Reads both headlines together. Swap augmentation in training, symmetric prediction (both orders averaged).
- First single run: val accuracy 0.736 at epoch 2, overfits afterwards (train loss 0.02 by epoch 4).
- Bug: transformers 5 loaded DeBERTa in float16 by default, giving NaN loss and constant predictions. Fixed by loading with dtype=torch.float32.
- Colab disconnected several times (free GPU limits). Notebooks now save each run as it finishes and skip finished runs on restart; long runs moved to Kaggle (Save & Run All runs in the background).

### Multi-seed runs (6 runs, 3 epochs each, best epoch kept)

| Learning rate | Seed 42 | Seed 43 | Seed 44 | Mean ± std |
|---|---|---|---|---|
| 2e-5 | 0.755 (ep 3) | 0.744 (ep 2) | 0.742 (ep 2) | 0.747 ± 0.007 |
| 1e-5 | 0.756 (ep 3) | 0.755 (ep 3) | 0.741 (ep 3) | 0.751 ± 0.008 |

- Chose lr 1e-5: same accuracy as 2e-5 but overfits less (train loss 0.26 to 0.31 at epoch 3, against 0.06 to 0.11), and the best epoch is always the last.
- The first single run (0.736) was at the low end, which is why multiple seeds were needed.

### Significance on validation: DeBERTa (lr 1e-5) vs Model A (0.713)

| Seed | Accuracy | Gap | McNemar p | Bootstrap 95% CI (by story) |
|---|---|---|---|---|
| 42 | 0.756 | +4.3 | 0.036 | [-1.2, +9.5] |
| 43 | 0.755 | +4.2 | 0.036 | [-1.1, +9.2] |
| 44 | 0.741 | +2.8 | 0.187 | [-2.5, +8.1] |

- McNemar assumes pairs are independent, but pairs from the same story are correlated. The story-level bootstrap accounts for this, and its CI includes zero for every seed.
- Conclusion: DeBERTa is consistently 3 to 4 points better, but validation has too few stories to show this is significant. This led to cross-validation.

## Model C: Qwen2.5-1.5B-Instruct
- Prompt: two headlines, "which got more clicks? Answer A or B". Prediction read from the scores of the tokens "A" and "B"; both orders averaged to remove position bias.
- Zero-shot (no training): 0.574 on all 3,860 pairs.
- LoRA (r 16, attention layers, 1 epoch, lr 1e-4, float32), same 5 folds as below: 0.753.
- Kaggle fix: removed the preinstalled torchao (version 0.10), which crashed peft.

## 5-fold cross-validation by story (train+val, 3,860 pairs, 1,000 stories)
- Folds assigned by story group, balanced by number of pairs; test split never touched.
- Hyperparameters fixed in advance (chosen on the validation split); DeBERTa and LoRA use the last epoch, so nothing is tuned on the scored folds.

| Model | CV accuracy |
|---|---|
| Negativity rule | 0.531 |
| LLM zero-shot (Qwen2.5-1.5B) | 0.574 |
| TF-IDF + LR | 0.641 |
| Embeddings + LR (Model A) | 0.699 |
| DeBERTa cross-encoder (Model B) | 0.752 |
| LLM + LoRA (Model C) | 0.753 |
| Ensemble B + C (average of probabilities) | 0.774 |

- DeBERTa per fold: 0.765, 0.754, 0.733, 0.760, 0.747.
- DeBERTa vs Model A: +5.2 points, CI [+3.2, +7.3], McNemar 597 vs 395, p < 0.0001. Significant.
- LLM LoRA vs zero-shot: +18.0 points, CI [+15.6, +20.4]. Significant. Fine-tuning on A/B data matters far more than LLM size.
- LLM LoRA vs DeBERTa: -0.1 points, CI [-1.9, +1.6]. Equivalent within about 2 points; DeBERTa is 8x smaller and much faster at prediction time.
- The two disagree on 800 of 3,860 pairs (21%).
- Ensemble (equal-weight average, nothing tuned) vs DeBERTa: +2.3 points, CI [+1.3, +3.2], McNemar 177 vs 90. Significant.

## To do
- Error analysis: 100+ ensemble errors, categorised.
- Part B: headline rewriter (engaging, non-negative, faithful).
- Final evaluation on the held-out test split and the confirmatory file.
- Demo, Docker, report, video.

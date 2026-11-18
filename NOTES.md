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

## Baselines
- Negativity rule (VADER): ties (equal negativity) in about 37% of pairs, broken by a seeded coin flip. Val accuracy 0.511, non-tied 0.524. On train+val (3,860 pairs, 2,556 non-tied), non-tied accuracy 0.549, 95% CI [0.522, 0.578] (bootstrap over story groups): a small but real negativity effect.
- TF-IDF + logistic regression on the difference vector, no intercept (swapping A and B flips the prediction). Best C = 1, val accuracy 0.634.
- Words linked to winning: she, her, response, seen, tell, did. Linked to losing: your, help, save, rights, we, family, love. Hypothesis: advocacy and appeal language loses, narrative and curiosity language wins.

## Models
- Model A, frozen bge-small embeddings + logistic regression: best C = 0.01 (peak inside the tested range), val accuracy 0.713.
- Model B, DeBERTa-v3-base cross-encoder, swap augmentation, symmetric prediction: val accuracy 0.736 at epoch 2, overfits afterwards (train loss 0.02 by epoch 4). Single seed, so the gap over Model A (2.3 points) is not yet shown to be significant.
- Bug: transformers 5 loaded DeBERTa in float16 by default, giving NaN loss and constant predictions. Fixed by loading with dtype=torch.float32.

## To do
- DeBERTa: 3 seeds, try lr 1e-5 with 3 epochs, significance test against Model A.
# Making Good News Win

Predicting headline engagement from real A/B tests and generating faithful, non-negative alternatives.

> Work in progress. Sections are filled in as the project develops.

## Overview
TODO

## Installation
```bash
git clone <your-repo>
cd good-news-nlp
pip install -r requirements.txt
```

## Dataset
Upworthy Research Archive (Matias et al., 2021). See `data/scripts/README.md` for download steps.

Data check:
```bash
python -m src.preprocessing --config configs/config.yaml
```

## Model Architecture
TODO

## Training
TODO

## Evaluation
TODO

## Demo
TODO

## Docker
TODO

## Project Structure
```
good-news-nlp/
├── data/{raw,processed,scripts}/
├── models/
├── src/
├── notebooks/
├── configs/config.yaml
├── tests/
├── requirements.txt
└── README.md
```

## Future Work
TODO

## References
- Matias, J. N., Munger, K., Aubin Le Quere, M., & Ebersole, C. (2021). The Upworthy Research Archive, a time series of 32,487 experiments in U.S. media. *Scientific Data*, 8, 195.
- Robertson, C. E., et al. (2023). Negativity drives online news consumption. *Nature Human Behaviour*, 7, 812–822.

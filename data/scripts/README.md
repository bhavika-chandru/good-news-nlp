# Getting the data

1. Go to https://osf.io/jd64p/ and open `upworthy-archive-datasets/`.
2. Download `upworthy-archive-exploratory-packages-03.12.2020.csv` into `data/raw/`.
3. Download the confirmatory file too, but do not open or analyse it until the final evaluation.
4. Run the data check: `python -m src.preprocessing --config configs/config.yaml`

The archive is published by Cornell University under CC BY 4.0. Cite Matias et al. (2021), *Scientific Data* 8:195.

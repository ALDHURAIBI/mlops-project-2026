# Network intrusion detection with drift monitoring

MLOps project 2026 (TSM-MachLeData). A binary IDS (benign / attack) trained on the NetFlow records of
NF-UNSW-NB15-v3, then replayed day by day on NF-CICIDS2018-v3 to measure drift and test retraining.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
dvc pull                      # data and models from the GCS bucket
```

## Pipeline

`dvc repro` runs the stages of `dvc.yaml`, with the settings of `params.yaml`:

- `prepare`: sample of NF-UNSW-NB15-v3, 38 flow features, train / val / test split
- `train@<model>`, `evaluate@<model>`: one stage per model of `src/models/` (dummy, logistic_regression,
  random_forest, autoencoder); results in `evaluation/<model>/`
- `drift@<model>`: NF-CICIDS2018-v3 replayed day by day, model frozen vs retrained every night
  (random_forest and autoencoder, about 5 min each)
- `package`: exports the deployed model (`deploy.model`) for BentoML

## Results

```bash
dvc metrics show              # all models
dvc metrics diff main         # what changed compared with main
dvc plots show                # graphs in dvc_plots/index.html
```

Notebooks 02 and 03 plot the drift replay of the Random Forest and the autoencoder.

## API and CI

```bash
cd src && bentoml serve serve:NetworkIntrusionService    # POST /predict on port 3000
```

Work on a branch, open a pull request to `dev`, then from `dev` to `main`. GitHub Actions checks every pull
request; each merge into `main` builds and publishes the Docker image, and creates a release when the pull request
has a `release:major`, `release:minor` or `release:patch` label.

Journal Title: IEEE Transactions on Information Forensics & Security

Title of the **Manuscript**: The Illusion of Forgetting: Unmasking and Resolving Deceptive Federated Unlearning

This repository contains the reference code structure for the Deceptive Federated Unlearning experiments.

## Recommended structure

```text
.
├── deceptive\_fu/
│   ├── \_\_init\_\_.py
│   ├── attacks.py
│   ├── config.py
│   ├── data.py
│   ├── experiment.py
│   ├── federated.py
│   ├── game.py
│   ├── models.py
│   ├── pipelines.py
│   ├── results.py
│   ├── utils.py
│   └── verifiers.py
├── run\_experiments.py
├── requirements.txt
├── README.md
└── .gitignore
```

## What each file does

* `config.py`: default hyperparameters and dataset presets.
* `data.py`: dataset loading, preprocessing, Dirichlet split, and client dataset wrapper.
* `models.py`: backbones and feature extraction.
* `federated.py`: local client training, FedAvg aggregation, testing, and retraining without a client.
* `utils.py`: random seed control and parameter vector utilities.
* `attacks.py`: historical-gradient aggregation and HG-PGA attack.
* `verifiers.py`: conformal and representation-based verification logic.
* `pipelines.py`: lightweight server-side FU baselines.
* `game.py`: game simulation.
* `experiment.py`: one complete experimental run.
* `results.py`: CSV saving and grouped summaries.
* `run\_experiments.py`: experiment grid entry point.

## Run

```bash
pip install -r requirements.txt
python run\_experiments.py
```



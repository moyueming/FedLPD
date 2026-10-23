# FedLPD

We present **FedLPD**, a federated reinforcement learning method for recommendation with heterogeneous client representations. We use behavioral information to align local latent spaces and personalized policy distillation to transfer knowledge across clients. In this repository, we provide the code for preparing the environments, training FedLPD and the comparison methods, and exporting numerical results.

## 1. Experimental Setup

We evaluate FedLPD in two environments: a KuaiSim environment built from KuaiRand-1K interaction logs and a RecSim environment generated locally. We compare FedLPD with Local RL, FedAvg, FedRAG, GPFedRec, FedHPD, FedSlate, and Random. We adapt the baselines to the shared recommendation environment and use the same task and evaluation protocol within each environment.

| Setting | Configuration |
| --- | --- |
| Clients | 5 per environment |
| Random seeds | 2026, 2027, 2028 |
| KuaiSim training budget | 80,000 requests per client |
| RecSim training budget | 40,000 requests per client |
| Comparisons | FedLPD, six learned baselines, Random, and seven ablations |

The complete comparison matrix contains 90 runs across both environments and all three seeds. We also run two independent FedAvg calibration experiments to set validation return targets before the comparisons. The configuration is stored in `configs/experiment.json`.

## 2. Prepare the Python Environment

We use Python 3.11 or 3.12. The commands below assume a Linux shell and should be run from the repository root. You can use an existing environment or create one:

```bash
python -m venv .venv
source .venv/bin/activate
```

For GPU experiments, first install a PyTorch build compatible with your CUDA setup in the active environment. Then install this project and check that PyTorch and RecSim are available:

```bash
python -m pip install -e .
python -m pip install --no-deps recsim==0.2.4
python -m fedlpd doctor --device cuda:0 --strict
```

We install RecSim without its unused TensorFlow/Dopamine dependencies. For a CPU environment check, replace `--device cuda:0` with `--device cpu`.

## 3. Prepare KuaiRand Data and Fit the Response Model

We use the public [KuaiRand-1K dataset](https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz) to prepare the KuaiSim environment. Download and extract it under `data/`. The archive MD5 is `6b0b9c8222d67fcd4c676218edca3f1f`.

Prepare the interaction logs and fit the response model with the following commands. The first command constructs the prepared dataset; the second saves the fitted model to `data/response.pt`:

```bash
python -m fedlpd prepare --config configs/experiment.json --logs data/KuaiRand-1K/data/log_standard_4_08_to_4_21_1k.csv data/KuaiRand-1K/data/log_standard_4_22_to_5_08_1k.csv --output data/prepared
python -m fedlpd fit-response --config configs/experiment.json --data-dir data/prepared --output data/response.pt --device cuda:0
```

We reuse these prepared assets across the comparison methods. If you store them elsewhere, update `--data-dir` and `--response` in the experiment command below. RecSim does not require an external dataset.

## 4. Check the Workflow with Synthetic Data

We provide a small synthetic run to check calibration, training, and result export before launching the full experiments. It runs on CPU without downloading KuaiRand or fitting a response model:

```bash
python scripts/run_experiment.py --smoke --devices cpu --output outputs/smoke
```

This check writes its outputs to `outputs/smoke/`. Its synthetic results are for software validation and must not be used as paper results.

## 5. Run the Complete Experiments on Four GPUs

We run our experiments on four GPUs in parallel, so the provided launch command also uses four GPUs. Each GPU handles one experiment run at a time, and the scheduler assigns the next run when that GPU becomes available:

```bash
python scripts/run_experiment.py --data-dir data/prepared --response data/response.pt --devices cuda:0,cuda:1,cuda:2,cuda:3 --output outputs/full
```

The launcher first calibrates the validation return targets and checks the environments. It then freezes the experiment configuration, runs all 90 comparisons, and exports the results. Training continues to the configured budget even when a run reaches its return target early.

To resume recoverable runs after an interruption, rerun the same command with the same output directory. Use a new output directory if you change the code, configuration, input paths, or worker allocation. For fewer GPUs, adjust `--devices` to list the available devices. Optional `--cpu-affinities` assigns one comma-separated CPU list per GPU; separate the lists with semicolons and ensure that they do not overlap.

## 6. Read the Exported Results

We save aggregated results under `outputs/full/results/finish/report/` and retain the underlying run records for further analysis:

| Location | Contents |
| --- | --- |
| `report/paper_tables/` | Six CSV tables covering performance, ablations, probe activity, alignment, and training costs |
| `report/data/` | Test trajectories, teacher weights, paired client gains, and alignment diagnostics in CSV/JSON/NPZ format |
| `report/costs_by_seed.csv` | Training time, communication volume, and rounds to the fixed validation return target for each run |
| `outputs/full/runs/` | Run configurations, evaluation records, communication logs, and checkpoints |

In this table, `report/` refers to `outputs/full/results/finish/report/`. We record discounted return, reward per request, and session length, together with the diagnostic and cost measurements. Targets that are not reached within the training budget remain marked `NR`; communication rounds are marked `N/A` for Local RL and Random.

We export numerical data only and do not include plotting code. Readers can use the exported data to create their own figures. This repository does not distribute datasets, trained models, or experimental results.

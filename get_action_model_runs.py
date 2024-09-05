from dataclasses import dataclass
from pathlib import Path
import tyro
import pandas as pd

import wandb
from wandb.wandb_run import Run
from wandb.sdk.wandb_config import Config


@dataclass(frozen=True)
class Args:
    result_dir: Path
    """Directory in which results are stored"""
    wandb_runs: tuple[str, ...]
    """a wandb run is specified by <entity>/<project>/<run_id> (can be obtained from wandb run overview)"""
    num_samples: int = 12000
    """Number of samples/ data points to retrieve from wandb per run"""


def parse_metric_df(run_config: dict, run_history: pd.DataFrame, metric: str, metric_name: str | None = None) -> pd.DataFrame:
    metric_df = pd.DataFrame(run_history[[metric, "global_step"]])
    metric_df = metric_df[metric_df[metric].notna()].set_index("global_step")
    metric_df["seed"] = run_config["seed"]

    if metric_name:
        metric_df.rename({metric: metric_name}, axis=1, inplace=True)
        metric_df.rename_axis("Step", inplace=True)

    return metric_df


if __name__ == "__main__":
    args = tyro.cli(Args, description="Script for downloading data/ runs from wandb server of action model training runs (from action_model/train.py) and saving it in pandas/seaborn readable csv files (1 per metric).")

    if args.result_dir.exists():
        raise Warning("The result_dir already exists!")
    else:
        args.result_dir.mkdir(parents=True)

    api = wandb.Api()

    total_losses_train = []
    total_losses_val = []
    reconstruction_losses_train = []
    reconstruction_losses_val = []
    contrastive_losses_train = []
    contrastive_losses_val = []

    pd.Series(args.wandb_runs).to_csv(args.result_dir / "wandb_run-ids.csv", header=["run_id"])

    for run in args.wandb_runs:
        run: Run = api.run(run)

        run_history: pd.DataFrame = run.history(samples=args.num_samples)
        run_config: Config = run.config

        total_losses_train.append(parse_metric_df(run_config, run_history, metric="losses/total_train", metric_name="Total Loss (Training)"))
        total_losses_val.append(parse_metric_df(run_config, run_history, metric="losses/total_val", metric_name="Total Loss (Validation)"))
        reconstruction_losses_train.append(parse_metric_df(run_config, run_history, metric="losses/reconstruction_train", metric_name="Reconstruction Loss (Training)"))
        reconstruction_losses_val.append(parse_metric_df(run_config, run_history, metric="losses/reconstruction_val", metric_name="Reconstruction Loss (Validation)"))
        contrastive_losses_train.append(parse_metric_df(run_config, run_history, metric="losses/contrastive_train", metric_name="Contrastive Loss (Training)"))
        contrastive_losses_val.append(parse_metric_df(run_config, run_history, metric="losses/contrastive_val", metric_name="Contrastive Loss (Validation)"))

    pd.concat(total_losses_train).to_csv(args.result_dir / "total_losses_train.csv")
    pd.concat(total_losses_val).to_csv(args.result_dir / "total_losses_val.csv")
    pd.concat(reconstruction_losses_train).to_csv(args.result_dir / "reconstruction_losses_train.csv")
    pd.concat(reconstruction_losses_val).to_csv(args.result_dir / "reconstruction_losses_val.csv")
    pd.concat(contrastive_losses_train).to_csv(args.result_dir / "contrastive_losses_train.csv")
    pd.concat(contrastive_losses_val).to_csv(args.result_dir / "contrastive_losses_val.csv")

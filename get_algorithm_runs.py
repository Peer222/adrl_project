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
    num_samples: int = 10000
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
    args = tyro.cli(Args, description="Script for downloading data/ runs from wandb server of algorithm runs (from algorithms/) and saving it in pandas/seaborn readable csv files (1 per metric).")

    if args.result_dir.exists():
        raise Warning("The result_dir already exists!")
    else:
        args.result_dir.mkdir(parents=True)

    api = wandb.Api()

    episodic_returns = []
    alphas = []
    actor_losses = []
    alpha_losses = []
    qf1_losses = []
    qf2_losses = []
    qf1_values = []
    qf2_values = []

    pd.Series(args.wandb_runs).to_csv(args.result_dir / "wandb_run-ids.csv", header=["run_id"])

    for run in args.wandb_runs:
        run: Run = api.run(run)

        run_history: pd.DataFrame = run.history(samples=args.num_samples)
        run_config: Config = run.config

        episodic_returns.append(parse_metric_df(run_config, run_history, metric="charts/episodic_return", metric_name="Episodic Return"))
        alphas.append(parse_metric_df(run_config, run_history, metric="losses/alpha", metric_name="Alpha"))
        actor_losses.append(parse_metric_df(run_config, run_history, metric="losses/actor_loss", metric_name="Actor Loss"))
        alpha_losses.append(parse_metric_df(run_config, run_history, metric="losses/alpha_loss", metric_name="Alpha Loss"))
        qf1_losses.append(parse_metric_df(run_config, run_history, metric="losses/qf1_loss", metric_name="Qf1 Loss"))
        qf2_losses.append(parse_metric_df(run_config, run_history, metric="losses/qf2_loss", metric_name="Qf2 Loss"))
        qf1_values.append(parse_metric_df(run_config, run_history, metric="losses/qf1_values", metric_name="Qf1 Values"))
        qf2_values.append(parse_metric_df(run_config, run_history, metric="losses/qf2_values", metric_name="Qf2 Values"))

    pd.concat(episodic_returns).to_csv(args.result_dir / "episodic_returns.csv")
    pd.concat(alphas).to_csv(args.result_dir / "alphas.csv")
    pd.concat(actor_losses).to_csv(args.result_dir / "actor_losses.csv")
    pd.concat(alpha_losses).to_csv(args.result_dir / "alpha_losses.csv")
    pd.concat(qf1_losses).to_csv(args.result_dir / "qf1_losses.csv")
    pd.concat(qf2_losses).to_csv(args.result_dir / "qf2_losses.csv")
    pd.concat(qf1_values).to_csv(args.result_dir / "qf1_values.csv")
    pd.concat(qf2_values).to_csv(args.result_dir / "qf2_values.csv")

from dataclasses import dataclass
from pathlib import Path
import tyro
import pandas as pd

import wandb
from wandb.wandb_run import Run
from wandb.sdk.wandb_config import Config

@dataclass(frozen=True)
class Args:
    wandb_runs: tuple[str, ...]
    """a wandb run is specified by <entity>/<project>/<run_id>"""
    result_dir: Path
    """Directory in which results are stored (Required)"""

def parse_metric_df(run_config: dict, run_history: pd.DataFrame, metric: str, metric_name: str | None = None) -> pd.DataFrame:
    metric_df = pd.DataFrame(run_history[[metric, "global_step"]])
    metric_df = metric_df[metric_df[metric].notna()].set_index("global_step")
    metric_df["seed"] = run_config["seed"]

    if metric_name:
        metric_df.rename({metric: metric_name}, inplace=True)

    return metric_df

if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.result_dir.exists():
        raise Warning("The result_dir already exists!")
    else:
        args.result_dir.mkdir(parents=True)

    api = wandb.Api()

    episodic_returns = []

    for run in args.wandb_runs:
        run: Run = api.run(run)

        run_history: pd.DataFrame = run.history()
        run_config: Config = run.config

        episodic_returns.append(parse_metric_df(run_config, run_history, metric="charts/episodic_return", metric_name="Episodic Return"))

    episodic_returns_df = pd.concat(episodic_returns)

    episodic_returns_df.to_csv(args.result_dir / "episodic_returns.csv")

from enum import Enum
from pathlib import Path
from dataclasses import dataclass
import tyro
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Color(Enum):
    BLACK = (0, 0, 0)
    ORANGE = (217 / 256, 130 / 256, 30 / 256)
    YELLOW = (227 / 256, 193 / 256, 0)
    BLUE = (55 / 256, 88 / 256, 136 / 256)
    GREY = (180 / 256, 180 / 256, 180 / 256)
    RED = (161 / 256, 34 / 256, 0)
    GREEN = (0, 124 / 256, 6 / 256)
    LIGHT_GREY = (240 / 256, 240 / 256, 240 / 256)

def prepare_dataframe(df, x: str, y: str, smoothing_window: int, step_resolution: int):
    if smoothing_window:
        step_size = round(df[x].max() / (len(df[x].unique()) - 1))
        df[y] = df.rolling(max(smoothing_window // step_size, 1), min_periods=0, on=x).mean()[y]

    if step_resolution:
        # somehow the data is saved every ...00 step or ...99
        df = df[(df[x] % step_resolution == 0) | ((df[x] + 1) % step_resolution == 0)]
    return df


def plot(df: pd.DataFrame, x: str, y: str, hue: str | None, filepath: Path, run_label: str | None = None):
    plt.figure(figsize=(8, 5))
    sns.lineplot(df, x=x, y=y, hue=hue, err_kws={"alpha": .25})

    title = df.columns[1]
    if run_label:
        title += f" ({run_label})"
    plt.title(title)
    plt.ylabel("Value")
    plt.xlabel(x)
    plt.legend()
    
    ax = plt.gca()
    ax.grid(True, color=Color.LIGHT_GREY.value)
    sns.despine(left=True, bottom=True, right=True, top=True)

    plt.savefig(filepath, dpi=300, format="png", bbox_inches="tight")
    plt.close()


@dataclass(frozen=True)
class Args:
    file_paths: tuple[Path, ...]
    """File paths to csv files from which plots should be created"""
    result_dir: Path
    """Path to result directory in which the plots are saved"""
    run_labels: tuple[str, ...] = ()
    """Labels that are shown in the legend for each experiment. Should have the same length as file_paths."""
    smoothing_window: int = 10000
    """Apply moving average with the passed window size (in steps). 0 for no averaging"""
    step_resolution: int = 10000
    """Frequency of steps that should be plotted"""


if __name__ == "__main__":
    args = tyro.cli(Args, description="Plotting script for training progress of multiple action_model or sac runs")

    if len(args.run_labels):
        assert len(args.file_paths) == len(args.run_labels), "Number of run labels and number of filepaths has to be the same"
        run_labels = args.run_labels
    else:
        run_labels = [fp.stem for fp in args.file_paths]

    if args.result_dir.exists():
        pass
        #raise Warning("The result_dir already exists!")
    else:
        args.result_dir.mkdir(parents=True)

    dfs = []

    for run_label, path in zip(run_labels, args.file_paths):

        df = pd.read_csv(path)
        df["Experiment"] = run_label.replace("_", " ")
        df = prepare_dataframe(df, x="Step", y=df.columns[1], smoothing_window=args.smoothing_window, step_resolution=args.step_resolution)
        dfs.append(df)
    
    plot(pd.concat(dfs), x="Step", y=df.columns[1], hue="Experiment", filepath=args.result_dir / f"multi_{args.file_paths[0].stem}.png")

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


def plot(df: pd.DataFrame, x: str, y: str, hue: str | None, filepath: Path, smoothing_window: int = 0, step_resolution: int = 0, run_label: str | None = None):
    if smoothing_window:
        step_size = round(df[x].max() / (len(df[x].unique()) - 1))
        df[y] = df.rolling(max(smoothing_window // step_size, 1), min_periods=0, on=x).mean()[y]

    if step_resolution:
        # somehow the data is saved every ...00 step or ...99
        df = df[(df[x] % step_resolution == 0) | ((df[x] + 1) % step_resolution == 0)]

    plt.figure(figsize=(8, 5))
    sns.lineplot(df, x=x, y=y, hue=hue)

    title = df.columns[1]
    if run_label:
        run_label = run_label.replace("_", " ")
        title += f" ({run_label})"
    plt.title(title)
    plt.ylabel("Value")
    plt.xlabel(x)
    
    ax = plt.gca()
    ax.grid(True, color=Color.LIGHT_GREY.value)
    sns.despine(left=True, bottom=True, right=True, top=True)

    plt.savefig(filepath, dpi=300, format="png", bbox_inches="tight")
    plt.close()


@dataclass(frozen=True)
class Args:
    file_paths: tuple[Path, ...]
    """File paths to csv files from which plots should be created or directory paths that contain the csv files that should be processed"""
    result_dir: Path
    """Path to result directory in which the plots are saved"""
    run_labels: tuple[str, ...] = ()
    """Labels that should be used in title along metric. Should have the same length as file_paths."""
    smoothing_window: int = 10000
    """Apply moving average with the passed window size (in steps). 0 for no averaging"""
    step_resolution: int = 10000
    """Frequency of steps that should be plotted"""


if __name__ == "__main__":
    args = tyro.cli(Args, description="Plotting script for training progress of action_model or sac runs")

    if len(args.run_labels):
        assert len(args.file_paths) == len(args.run_labels), "Number of run labels and number of filepaths has to be the same"
        run_labels = args.run_labels
    else:
        run_labels = [None for _ in range(len(args.file_paths))]

    if args.result_dir.exists():
        pass
        #raise Warning("The result_dir already exists!")
    else:
        args.result_dir.mkdir(parents=True)

    for run_label, path in zip(run_labels, args.file_paths):
        paths = [path]
        if path.is_dir():
            paths = path.glob("*.csv")

        for path in paths:
            if "wandb" in path.name:
                continue
            df = pd.read_csv(path)
            plot(df, x="Step", y=df.columns[1], hue=None, filepath=(args.result_dir / path.stem).with_suffix(".png"), smoothing_window=args.smoothing_window, step_resolution=args.step_resolution, run_label=run_label)

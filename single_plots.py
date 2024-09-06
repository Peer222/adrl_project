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


def plot(df: pd.DataFrame, x: str, y: str, hue: str | None, filepath: Path, smoothing_window: int = 0, step_resolution: int = 0):
    if smoothing_window:
        df[y] = df[y].rolling(smoothing_window).mean()

    if step_resolution:
        df = df[(df[x] % step_resolution == 0) | ((df[x] + 1) % step_resolution == 0)]

    plt.figure(figsize=(8, 5))
    sns.lineplot(df, x=x, y=y, hue=hue)
    plt.title(df.columns[1])
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


if __name__ == "__main__":
    args = tyro.cli(Args, description="Plotting script for training progress of action_model or sac runs")

    if args.result_dir.exists():
        raise Warning("The result_dir already exists!")
    else:
        args.result_dir.mkdir(parents=True)

    for path in args.file_paths:
        paths = [path]
        if path.is_dir():
            paths = path.glob("*.csv")

        for path in paths:
            if "wandb" in path.name:
                continue
            df = pd.read_csv(path)
            plot(df, x="Step", y=df.columns[1], hue=None, filepath=(args.result_dir / path.stem).with_suffix(".png"))

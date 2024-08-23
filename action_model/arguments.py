from typing import Literal
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class Args:
    exp_name: str = "action_model"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    track_frequency: int = 10
    """After n steps add losses... to log"""
    validation_frequency: int = 10000
    """After n steps perform validation run"""
    wandb_project_name: str = "adrl_project"
    """the wandb's project name"""
    wandb_entity: str = "peer222-luh"
    """the entity (team) of wandb's project"""

    dataset_id: str = "door-expert-v2"
    """id of minari dataset (has to be pre-downloaded)"""
    env_id: str = "AdroitHandDoor-v1"
    """the environment id of the task"""
    train_split: float = 0.9
    """portion of samples from dataset used for training"""

    num_layers: int = 3 # tuning required
    """number of layers for encoder and decoder (each)"""
    latent_features: int = 20 # tuning required
    """number of latent action features"""
    activation_fn: Literal["relu", "tanh", "sigmoid"] = "tanh"
    """latent activation functions that are used in encoder and decoder (final activation of decoder is always tanh to map to action space)"""

    iterations: int = 100000
    """number of iterations/ parameter updates"""
    window_size: int = 11 # tuning required
    """number of subsequent actions (1/2 before and 1/2 after) that should be considered similar in a batch"""
    num_negative_examples: int = 11
    """number of non-matching actions that should be considered dissimilar from current action"""
    negative_example_source: Literal["random", "sampling"] = "sampling"
    """Source of negative examples: Sampling from dataset or random initialization"""
    lr: float = 0.0001
    """learning rate"""
    weight_decay: float = 0
    """weight decay"""
    optimizer: Literal["adam", "sgd"] = "adam"
    """optimizer for training"""
    contrastive_loss: Literal["cosine_embedding", "triplet_margin"] = "cosine_embedding"
    """Loss that is used for comparing latent representations"""
    contrastive_loss_margin: float = 0
    """Margin used for loss computations. For cosine_embedding: [-1, 1] with default 0 (->0.5), For triplet_margin: default 1"""
    contrastive_loss_factor : float = 1
    """Factor that is multiplied to contrastive loss"""
    reconstruction_loss: Literal["mse"] = "mse"
    """Loss used for reconstruction"""
    reconstruction_loss_factor: float = 1
    """Factor that is multiplied to reconstruction loss"""

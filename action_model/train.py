from typing import TypeAlias, Iterable
import warnings
from pathlib import Path
import minari.dataset
import minari.dataset.episode_data
import tyro
import datetime
import time

import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import minari
import model
import arguments


EpisodeData: TypeAlias = minari.dataset.episode_data.EpisodeData


def create_random_actions(action_space: gym.Space, num_actions: int) -> np.ndarray:
    """Gets random actions sampled from action__space

    Args:
        action_space (gym.Space): Action space to sample from
        num_actions (int): Number of samples/ actions

    Returns:
        np.ndarray: random actions
    """
    actions = [action_space.sample() for _ in range(num_actions)]
    return np.stack(actions)

def get_negative_actions(episodes: Iterable[EpisodeData]) -> np.ndarray:
    """Gets actions sampled from episodes
    One action is sampled per episode

    Args:
        episodes (np.ndarray): Episodes from which to sample

    Returns:
        np.ndarray: sampled actions
    """
    actions = [episode.actions[np.random.randint(0, len(episode.actions))] for episode in episodes]
    return np.stack(actions)


def get_nearby_actions(episode: EpisodeData, action_index: int, window_size: int) -> np.ndarray:
    """Gets actions surrounding current action
    If window is out of bounds zero actions are appended

    Args:
        episode (EpisodeData): Episode that contains current action
        action_index (int): Index in episode of current action
        window_size (int): Number of actions that are added (1/2 before and 1/2 after current action. (odd number expected))

    Returns:
        np.ndarray: action sequence
    """
    assert window_size % 2 == 1, "Odd number for window_size expected"
    actions = []
    for index in range(action_index - window_size // 2, action_index + window_size // 2 + 1, 1):
        if index < 0 or index >= len(episode.actions):
            actions.append(np.zeros_like(episode.actions[action_index]))
        elif index != action_index:
            actions.append(episode.actions[index])
    
    return np.stack(actions)


def create_input_batch(args: arguments.Args, dataset: minari.MinariDataset, episode: EpisodeData | None = None) -> torch.Tensor:
    """Creates input batch for encoder according to args

    Args:
        args (arguments.Args): arguments
        dataset (minari.MinariDataset): dataset to sample from
        episode (EpisodeData | None): episode to sample current action... from (used for validation)

    Returns:
        torch.Tensor: input batch
    """
    if not episode:
        episode: EpisodeData = dataset.sample_episodes(1)[0]

    # batch creation
    action_index = np.random.randint(0, len(episode.actions))
    current_action = torch.from_numpy(episode.actions[action_index]).unsqueeze(0)

    nearby_actions = torch.from_numpy(get_nearby_actions(episode, action_index, args.window_size))

    if args.negative_example_source == "sampling":
        episodes = dataset.sample_episodes(args.num_negative_examples)
        random_actions_np = get_negative_actions(episodes)
    elif args.negative_example_source == "random":
        random_actions_np = create_random_actions(env.action_space, args.num_negative_examples)
    random_actions = torch.from_numpy(random_actions_np)

    return torch.cat([current_action, nearby_actions, random_actions]).to(device)


def compute_losses(args: arguments.Args, reconstruction_loss_fn: nn.Module, contrastive_loss_fn: nn.Module, input_batch: torch.Tensor, latent_batch: torch.Tensor, output_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes losses for given inputs

    Args:
        args (arguments.Args): arguments
        reconstruction_loss_fn (nn.Module): Loss function for reconstruction
        contrastive_loss_fn (nn.Module): Loss function for latent space similarities
        input_batch (torch.Tensor): Inputs for encoder  [current_action, nearby_actions, random_actions]
        latent_batch (torch.Tensor): Inputs for decoder/ Outputs of encoder  [current_action, nearby_actions, random_actions]
        output_batch (torch.Tensor): Outputs of decoder  [current_action, nearby_actions, random_actions]

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: total loss, contrastive loss, reconstruction loss
    """
    if args.contrastive_loss == "cosine_embedding":
        # target 1 for nearby_actions and -1 for random_actions
        num_actions = args.window_size - 1 + args.num_negative_examples
        target = torch.ones(num_actions).to(device)
        target[-args.num_negative_examples:] = -1

        # compare current action to the others
        cur_latent_action = latent_batch[0].unsqueeze(0).repeat(num_actions, 1)
        contrastive_loss = contrastive_loss_fn(cur_latent_action, latent_batch[1:], target)
    elif args.contrastive_loss == "triplet_margin":
        # compare current action to the others
        cur_latent_action = latent_batch[0].unsqueeze(0).repeat(args.num_negative_examples, 1)
        contrastive_loss = contrastive_loss_fn(cur_latent_action, latent_batch[1 : args.window_size], latent_batch[args.window_size:])

    reconstruction_loss = reconstruction_loss_fn(input_batch, output_batch)

    loss = args.contrastive_loss_factor * contrastive_loss + args.reconstruction_loss_factor * reconstruction_loss

    return loss, contrastive_loss, reconstruction_loss


if __name__ == "__main__":
    args = tyro.cli(arguments.Args)

    print(args, flush=True)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{datetime.datetime.now()}"
    run_dir = Path("action_runs") / run_name

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args) | {"run_name": run_name},
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(str(run_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # SEEDING
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    dataset = minari.load_dataset(args.dataset_id)
    env  = dataset.recover_environment()
    train_dataset, val_dataset = minari.split_dataset(dataset, [int(args.train_split * len(dataset)), int((1 - args.train_split) * len(dataset))], seed=args.seed)

    if args.env_id != env.spec.id:
        warnings.warn(f"Environment id passed as argument does not match environment id of dataset: {args.env_id} VS {env.spec.id}")

    if args.activation_fn == "relu":
        activation_fn = nn.ReLU()
    elif args.activation_fn == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif args.activation_fn == "tanh":
        activation_fn = nn.Tanh()

    encoder = model.ActionEncoder(env.action_space, args.latent_features, args.num_layers, activation_fn).to(device)
    decoder = model.ActionDecoder(env.action_space, args.latent_features, args.num_layers, activation_fn).to(device)


    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    if args.contrastive_loss == "cosine_embedding":
        contrastive_loss_fn = nn.CosineEmbeddingLoss(margin=args.contrastive_loss_margin)
    elif args.contrastive_loss == "triplet_margin":
        assert args.num_negative_examples == args.window_size - 1, "Triplet loss needs same amount of positive and negative examples"
        contrastive_loss_fn = nn.TripletMarginLoss(margin=args.contrastive_loss_margin)
    reconstruction_loss_fn = nn.MSELoss()

    start_time = time.time()

    for iter in range(args.iterations):
        encoder.train()
        decoder.train()

        input_batch = create_input_batch(args, train_dataset).to(device)
        latent_batch: torch.Tensor = encoder(input_batch)
        output_batch: torch.Tensor = decoder(latent_batch)

        loss, contrastive_loss, reconstruction_loss = compute_losses(args, reconstruction_loss_fn, contrastive_loss_fn, input_batch, latent_batch, output_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter == 0 or (iter + 1) % args.track_frequency == 0:
            writer.add_scalar("losses/total_train", loss.item(), iter)
            writer.add_scalar("losses/contrastive_train", contrastive_loss.item(), iter)
            writer.add_scalar("losses/reconstruction_train", reconstruction_loss.item(), iter)
            writer.add_scalar("charts/SPS", int(iter / (time.time() - start_time)), iter)

        if iter == 0 or (iter + 1) % args.validation_frequency == 0:
            val_total_losses = []
            val_contrastive_losses = []
            val_reconstruction_losses = []

            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                for episode in val_dataset:
                    input_batch = create_input_batch(args, val_dataset, episode).to(device)
                    latent_batch: torch.Tensor = encoder(input_batch)
                    output_batch: torch.Tensor = decoder(latent_batch)

                    loss, contrastive_loss, reconstruction_loss = compute_losses(args, reconstruction_loss_fn, contrastive_loss_fn, input_batch, latent_batch, output_batch)
                    val_total_losses.append(loss.item())
                    val_contrastive_losses.append(contrastive_loss.item())
                    val_reconstruction_losses.append(reconstruction_loss.item())
            
            print(f"Val loss after {iter} iterations: {np.mean(val_total_losses)}")
            writer.add_scalar("losses/total_val", np.mean(val_total_losses), iter)
            writer.add_scalar("losses/contrastive_val", np.mean(val_contrastive_losses), iter)
            writer.add_scalar("losses/reconstruction_val", np.mean(val_reconstruction_losses), iter)

    writer.close()

    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True)
    torch.save(encoder.state_dict(), model_dir / "encoder.pth")
    torch.save(decoder.state_dict(), model_dir / "decoder.pth")
    torch.save(optimizer.state_dict(), model_dir / "optimizer.pth")

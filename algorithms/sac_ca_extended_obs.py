import sys
sys.path.append('./')

import os
from typing import Literal
from pathlib import Path
from dataclasses import dataclass
import json
import random
import time
import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from utils import ReplayBuffer
from action_model import model as action_model


@dataclass
class Args:
    action_model_dir: Path
    """Directory in which the trained action autoencoder is saved (encoder.pth, decoder.pth)"""
    observation_input: Literal["concat_actions", "replace_actions"] = "concat_actions"
    """Wether the action embedding of the previous action should be concatenated to the observation or should replace action related observation parameters"""

    exp_name: str = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    track_frequency: int = 100
    """After n steps add losses... to log"""
    wandb_project_name: str = "adrl_project"
    """the wandb's project name"""
    wandb_entity: str = "peer222-luh"
    """the entity (team) of wandb's project"""
    capture_video: int = 50000
    """Frequency (global_step) of capturing videos. Set to 0 for no video capture  (check out `videos` folder)"""

    load_models_from: Path | None = None
    """load model parameters from model_dir"""

    # Algorithm specific arguments
    env_id: str = "AdroitHandDoor-v1"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    max_episode_length: int = 200
    """maximal length of an episode"""


def make_env(env_id, seed, idx, capture_video, run_name, max_episode_length):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=max_episode_length)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", step_trigger=lambda s: s % capture_video == 0, video_length=max_episode_length * 3)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, observation_size: int, action_size: int, action_embedding_size: int, mode: Literal["concat_actions", "replace_actions", "obs_only"]):
        super().__init__()
        if mode == "concat_actions":
            self.fc1 = nn.Linear(observation_size + action_embedding_size + action_size, 256)
        elif mode == "replace_actions":
            raise NotImplementedError("Replacing observation parameters by action embeddings is currently not supported. Use 'concat' instead")
        elif mode == "obs_only":
            self.fc1 = nn.Linear(observation_size, 256)
        
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, observation_size: int, action_size: int, action_embedding_size: int, mode: Literal["concat_actions", "replace_actions", "obs_only"], highest_action_value: float, lowest_action_value: float):
        super().__init__()
        if mode == "concat_actions":
            self.fc1 = nn.Linear(observation_size + action_embedding_size, 256)
        elif mode == "replace_actions":
            raise NotImplementedError("Replacing observation parameters by action embeddings is currently not supported. Use 'concat' instead")
        elif mode == "obs_only":
            self.fc1 = nn.Linear(observation_size, 256)
        self.mode = mode
        
        
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_size)
        self.fc_logstd = nn.Linear(256, action_size)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((highest_action_value - lowest_action_value) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((highest_action_value + lowest_action_value) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    print(args, flush=True)
    if not args.exp_name:
        run_name = f"{args.env_id}_{os.path.basename(__file__)[: -len('.py')]}_s{args.seed}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    else:
        run_name = args.exp_name
    run_dir = Path("runs") / run_name
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args) | {"run_name": run_name},
            name=run_name,
            id=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(str(run_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name, args.max_episode_length)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    with open(args.action_model_dir / "config.json", "r") as config_file:
        action_model_config = json.load(config_file)
    action_encoder = action_model.ActionEncoder(envs.single_action_space, action_model_config["latent_features"], action_model_config["num_layers"], action_model_config["activation_fn"]).to(device)
    action_encoder.load_state_dict(torch.load(args.action_model_dir / "encoder.pth", weights_only=True))
    action_encoder.eval()

    obs_size = np.array(envs.single_observation_space.shape).prod()
    action_size = np.prod(envs.single_action_space.shape)

    actor = Actor(obs_size, action_size, action_model_config["latent_features"], args.observation_input, envs.action_space.high, envs.action_space.low).to(device)
    qf1 = SoftQNetwork(obs_size, action_size, action_model_config["latent_features"], args.observation_input).to(device)
    qf2 = SoftQNetwork(obs_size, action_size, action_model_config["latent_features"], args.observation_input).to(device)
    qf1_target = SoftQNetwork(obs_size, action_size, action_model_config["latent_features"], args.observation_input).to(device)
    qf2_target = SoftQNetwork(obs_size, action_size, action_model_config["latent_features"], args.observation_input).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    # TODO check
    if args.load_models_from:
        actor.load_state_dict(torch.load(args.load_models_from / "actor.pth", map_location=device))
        qf1.load_state_dict(torch.load(args.load_models_from / "qf1.pth", map_location=device))
        qf1_target.load_state_dict(torch.load(args.load_models_from / "qf1_target.pth", map_location=device))
        qf2.load_state_dict(torch.load(args.load_models_from / "qf2.pth", map_location=device))
        qf2_target.load_state_dict(torch.load(args.load_models_from / "qf2_target.pth", map_location=device))

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        action_model_config["latent_features"],
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    prev_latent_actions = np.zeros((envs.num_envs, action_model_config["latent_features"]))

    # so that 3 full episodes are recorded at end of training
    for global_step in range(args.total_timesteps + args.max_episode_length * 3):
        # ALGO LOGIC: put action logic here
        if False and global_step < args.learning_starts: # random sampling cannot be combined with previous action embeddings in a reasonable way
            actions_np = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            actions = torch.Tensor(actions_np).to(device)
        else:
            if args.observation_input == "concat_actions":
                extended_obs = torch.Tensor(np.concatenate([obs, prev_latent_actions], 1)).to(device)
            elif args.observation_input == "replace_actions":
                raise NotImplementedError("replace_actions is not implemented, Use 'concat_actions' instead")
            actions, _, _ = actor.get_action(extended_obs)
            actions_np = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions_np)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step + 1)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step + 1)
                break

        with torch.no_grad():
            latent_actions = action_encoder(actions).detach().cpu().numpy()

        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions_np, rewards, terminations, infos, latent_actions, prev_latent_actions)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        prev_latent_actions = latent_actions
        for idx, trunc in enumerate(truncations):
            if trunc:
                # new episode is started so starting with zero action as previous action
                prev_latent_actions[idx] = 0

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                if args.observation_input == "concat_actions":
                    extended_next_obs = torch.cat([data.next_observations, data.latent_actions], 1)
                next_state_actions, next_state_log_pi, _ = actor.get_action(extended_next_obs)
                qf1_next_target = qf1_target(extended_next_obs, next_state_actions)
                qf2_next_target = qf2_target(extended_next_obs, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            if args.observation_input == "concat_actions":
                extended_obs = torch.cat([data.observations, data.previous_latent_actions], 1)
            qf1_a_values = qf1(extended_obs, data.actions).view(-1)
            qf2_a_values = qf2(extended_obs, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(extended_obs)
                    qf1_pi = qf1(extended_obs, pi)
                    qf2_pi = qf2(extended_obs, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(extended_obs)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step == 0 or (global_step + 1) % args.track_frequency == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()

    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True)
    torch.save(actor.state_dict(), model_dir / "actor.pth")
    torch.save(qf1.state_dict(), model_dir / "qf1.pth")
    torch.save(qf1_target.state_dict(), model_dir / "qf1_target.pth")
    torch.save(qf2.state_dict(), model_dir / "qf2.pth")
    torch.save(qf2_target.state_dict(), model_dir / "qf2_target.pth")

import warnings
from pathlib import Path
import tyro
import datetime
import time

import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import minari
import model
import arguments


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
        optimizer = torch.optim.adam.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.sgd.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    if args.contrastive_loss == "cosine_embedding":
        contrastive_loss_fn = nn.CosineEmbeddingLoss(margin=args.contrastive_loss_margin)
    elif args.contrastive_loss == "triplet_margin":
        contrastive_loss_fn = nn.TripletMarginLoss(margin=args.contrastive_loss_margin)
    reconstruction_loss_fn = nn.MSELoss()

    start_time = time.time()

    for iter in range(args.iterations):
        negative_samples = []
        if args.negative_example_source == "sampling":
            episodes = train_dataset.sample_episodes(args.num_negative_examples + 1)
            actions = episodes[0].actions
            print(actions, type(actions))

        elif args.negative_example_source == "random":
            pass

        if args.contrastive_loss == "cosine_embedding":
            contrastive_loss = contrastive_loss_fn()
        elif args.contrastive_loss == "triplet_margin":
            contrastive_loss = contrastive_loss_fn()

        reconstruction_loss = torch.Tensor([0])

        if iter == 0 or (iter + 1) % args.track_frequency == 0:
            writer.add_scalar("losses/contrastive_train", contrastive_loss.item(), iter)
            writer.add_scalar("losses/reconstruction_train", reconstruction_loss.item(), iter)
            writer.add_scalar("charts/SPS", int(iter / (time.time() - start_time)), iter)

        if iter == 0 or (iter + 1) % args.validation_frequency == 0:
            pass

        break



    writer.close()

    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True)
    torch.save(encoder.state_dict(), model_dir / "encoder.pth")
    torch.save(decoder.state_dict(), model_dir / "decoder.pth")
    torch.save(optimizer.state_dict(), model_dir / "optimizer.pth")

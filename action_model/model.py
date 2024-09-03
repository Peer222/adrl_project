import gymnasium as gym

import torch
import torch.nn as nn
import numpy as np

class ActionEncoder(nn.Module):
    def __init__(self, action_space: gym.Space, latent_features: int, num_layers: int = 3, activation_fn: nn.Module | str = nn.Tanh()) -> None:
        super().__init__()

        in_features = np.prod(action_space.shape)
        out_features = latent_features

        layers = []
        for i in range(num_layers):
            in_ = int(in_features + (out_features - in_features) * i / num_layers)
            out_ = int(in_features + (out_features - in_features) * (i + 1) / num_layers)
            layers.append(nn.Linear(in_features=in_, out_features=out_))
        self.layers = nn.ModuleList(layers)

        if isinstance(activation_fn, str):
            if activation_fn == "relu":
                self.activation_fn = nn.ReLU()
            elif activation_fn == "sigmoid":
                self.activation_fn = nn.Sigmoid()
            elif activation_fn == "tanh":
                self.activation_fn = nn.Tanh()
            else:
                raise ValueError(f"The identifier {activation_fn} is not supported for activation_fn.")
        else:
            self.activation_fn = activation_fn


    def forward(self, action: torch.Tensor) -> torch.Tensor:

        for layer in self.layers:
            action = layer(action)
            action = self.activation_fn(action)

        return action


class ActionDecoder(nn.Module):
    def __init__(self, action_space: gym.Space, latent_features: int, num_layers: int = 3, activation_fn: nn.Module | str = nn.Tanh()) -> None:
        super().__init__()

        in_features = latent_features
        out_features = np.prod(action_space.shape)


        layers = []
        for i in range(num_layers):
            in_ = int(in_features + (out_features - in_features) * i / num_layers)
            out_ = int(in_features + (out_features - in_features) * (i + 1) / num_layers)
            layers.append(nn.Linear(in_features=in_, out_features=out_))
        self.layers = nn.ModuleList(layers)

        if isinstance(activation_fn, str):
            if activation_fn == "relu":
                self.activation_fn = nn.ReLU()
            elif activation_fn == "sigmoid":
                self.activation_fn = nn.Sigmoid()
            elif activation_fn == "tanh":
                self.activation_fn = nn.Tanh()
            else:
                raise ValueError(f"The identifier {activation_fn} is not supported for activation_fn.")
        else:
            self.activation_fn = activation_fn

        # Adroit action space normalizes all actions between -1 and 1
        self.final_activation_fn = nn.Tanh()


    def forward(self, action: torch.Tensor) -> torch.Tensor:

        for i, layer in enumerate(self.layers):
            action = layer(action)
            if i < len(self.layers) - 1:
                action = self.activation_fn(action)

        action = self.final_activation_fn(action)
        return action
    

if __name__ == "__main__":
    #from torchviz import make_dot
    import minari
    import tyro

    import arguments

    args = tyro.cli(arguments.Args)

    dataset = minari.load_dataset(args.dataset_id)
    env  = dataset.recover_environment()

    if args.activation_fn == "relu":
        activation_fn = nn.ReLU()
    elif args.activation_fn == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif args.activation_fn == "tanh":
        activation_fn = nn.Tanh()

    input_action = torch.randn(env.action_space.shape, requires_grad=False)

    encoder = ActionEncoder(env.action_space, args.latent_features, args.num_layers, activation_fn)
    print(encoder)

    latent_action = encoder(input_action)

    decoder = ActionDecoder(env.action_space, args.latent_features, args.num_layers, activation_fn)
    print(decoder)

    restored_action = decoder(latent_action)

    # make_dot(restored_action, decoder.state_dict() | encoder.state_dict()).render(filename="network_graph", directory="action_model", format="png", cleanup=True)
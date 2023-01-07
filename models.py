from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from simple_vit import SimpleViT
from tianshou.utils.net.discrete import NoisyLinear


def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        device: Union[str, int, torch.device] = "cpu",
        features_only: bool = False,
        output_dim: Optional[int] = None,
        layer_init: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> None:
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True), nn.Flatten()
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.model(torch.zeros(1, c, h, w)).shape[1:])
        if not features_only:
            self.model = nn.Sequential(
                self.model, layer_init(nn.Linear(self.output_dim, 512)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(512, np.prod(action_shape)))
            )
            self.output_dim = np.prod(action_shape)
        elif output_dim is not None:
            self.model = nn.Sequential(
                self.model, layer_init(nn.Linear(self.output_dim, output_dim)),
                nn.ReLU(inplace=True)
            )
            self.output_dim = output_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.model(obs), state


class AtariViT(torch.nn.Module):
    def __init__(self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        device: Union[str, int, torch.device] = "cpu" ) -> None:
        super().__init__()
        # 84 observation size is because of FrameWrap in retro_wrappers. 
        # It follows the deep-mind style of capturing Atari observations 
        assert h == w == 84 

        # Below is a set of (not rigorously tested) hyper-parameters for testing the model. 
        # Note that 84 should be fixed and c (number of channels) == args.frame_stack in 
        # retro_main.py for compatibability. 
        # NOTE: dim, mlp_dim, dim_head, depth, have all been scaled down by a factor of two
        # compared to the original papers setup:  
        ### Better plain ViT baselines for ImageNet-1k: https://arxiv.org/abs/2205.01580


        self.model = SimpleViT(
            image_size = 84,
            patch_size = 7,
            num_classes = action_shape,
            dim = 512,
            depth = 3,
            heads = 16,
            mlp_dim = 1024,
            channels=c, 
            dim_head=32
        )
        self.device = device

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.model(obs), state


class DNN(nn.Module):
    def __init__(self, 
    state_shape: Sequence[int], 
    action_shape: Sequence[int],
    hidden_dims: Sequence[int] = [128,128],
    device: Union[str, int, torch.device] = "cpu") -> None:
        super().__init__()
        self.model = nn.Sequential(nn.Linear(np.prod(state_shape),hidden_dims[0]))
        for i in range(len(hidden_dims) - 1): 
            self.model = nn.Sequential(
                self.model,
                nn.Linear(hidden_dims[i],hidden_dims[i+1]), 
                nn.ReLU(inplace=True)
            )
        self.model = nn.Sequential(self.model, nn.Linear(hidden_dims[-1],action_shape))
        self.device = device

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""

        batch = obs.shape[0]
        obs = obs.reshape((batch,-1))
        print(self.model)
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.model(obs), state
import torch as th
from typing import Tuple, Callable
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ScanFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super().__init__(observation_space, features_dim)

        self.scan_net = nn.Sequential(
            nn.Linear(36, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim // 2),
            nn.LayerNorm(features_dim // 2),
            nn.ReLU(),
            nn.Linear(features_dim // 2, features_dim // 2),
        )

        self.dynamic_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, features_dim // 2),
            nn.LayerNorm(features_dim // 2),
            nn.ReLU(),
            nn.Linear(features_dim // 2, features_dim //2),
        )

        self.feature_net = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim // 2),
            nn.LayerNorm( features_dim // 2),
            nn.ReLU(),
            nn.Linear(features_dim // 2, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, observations) -> th.Tensor:
        scans, dynamic = th.split(observations, [36, 4], dim=-1)
        scan_feature = self.scan_net(scans)
        dynamic_feature = self.dynamic_net(dynamic)
        features = self.feature_net(th.cat([scan_feature, dynamic_feature], dim=-1))
        return features


class ScanNetWork(nn.Module):
    def __init__(
        self, feature_dim: int, last_layer_dim_pi: int = 64, last_layer_dim_vf: int = 64
    ):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf


        self.p_net0 = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        self.p_net1 = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.LayerNorm(feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        self.p_net_a1 = nn.ReLU()
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, self.latent_dim_pi),
            nn.LayerNorm(self.latent_dim_pi),
            nn.ReLU()
        )



        self.c_net0 = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        self.c_net1 = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.LayerNorm(feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        self.c_net_a1 = nn.ReLU()
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, self.latent_dim_pi),
            nn.LayerNorm(self.latent_dim_pi),
            nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        feature = self.p_net0(features)
        feature_inner = self.p_net1(feature)
        feature_inner = self.p_net_a1(feature_inner + feature)
        output = self.policy(feature_inner)
        return output
    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        feature = self.c_net0(features)
        feature_inner = self.c_net1(feature)
        feature_inner = self.c_net_a1(feature_inner + feature)
        output = self.critic(feature_inner)
        return output


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ScanNetWork(128)

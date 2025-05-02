from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tyro

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    # torch_deterministic: bool = True
    # """if toggled, `torch.backends.cudnn.deterministic=False`"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Augmentations runs"
    """the wandb's project name"""
    wandb_entity: Optional[str] = "pranavmalpure-uc-san-diego-health"
    """the entity (team) of wandb's project"""
    wandb_group: str = "SAC"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    wandb_video_freq: int = 10000
    save_trajectory: bool = False
    """whether to save trajectory data into the `videos` folder"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    log_freq: int = 1_000
    """logging frequency in terms of environment steps"""

    # Environment specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    obs_mode: str = "pointcloud"
    """the observation mode to use"""
    include_state: bool = True
    """whether to include the state in the observation"""
    env_vectorization: str = "gpu"
    """the type of environment vectorization to use"""
    num_envs: int = 2
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    render_mode: str = "all"
    """the environment rendering mode"""
    robot_uids: str = "panda"
    """the robot uids to use. If none it will use the default the environment specifies."""
    reward_mode: str = "dense"
    """the reward mode to use"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""

class DP3Encoder(nn.Module):
    def __init__(self, channels=3):
        # We only use xyz (channels=3) in this work
        # while our encoder also works for xyzrgb (channels=6) in our experiments
        self.mlp = nn.Sequential(
        nn.Linear(channels, 64), nn.LayerNorm(64), nn.ReLU(),
        nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
        nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU())
        self.projection = nn.Sequential(nn.Linear(256, 64), nn.LayerNorm(64))
    
    def forward(self, x):
        # x: B, N, 3
        x = self.mlp(x) # B, N, 256
        x = torch.max(x, 1)[0] # B, 256
        x = self.projection(x) # B, 64
        return x



def train():





if __name__ == "__main__":
    
    train()
    
    args = tyro.cli(Args)
    env = gym.make(args.env_id,
               num_envs=args.num_envs,
               obs_mode=args.obs_mode,
               reward_mode=args.reward_mode,
               control_mode=args.control_mode,
               robot_uids=args.robot_uids,
               enable_shadow=True # this makes the default lighting cast shadows
               )
    
    obs, _ = env.reset()
    dp3encoder = DP3Encoder(3)

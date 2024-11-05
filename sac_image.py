
from collections import defaultdict
from dataclasses import dataclass
import os
import random
import time
from typing import Optional

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro

import mani_skill.envs
from torch.distributions.normal import Normal



# env_id = "PickCube-v1"
# obs_mode = "rgb+depth"
# control_mode = "pd_joint_delta_pos"
# reward_mode = "sparse"
# robot_uids = "panda"


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: str = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save the model checkpoints"""

    # Env specific arguments
    env_id: str = "PushCube-v1"
    """the environment id of the task"""
    num_envs: int = 16
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    num_eval_steps: int = 50
    """the number of steps to take in evaluation environments"""
    reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    log_freq: int = 1_000
    """logging frequency in terms of environment steps"""
    eval_freq: int = 100_000
    """evaluation frequency in terms of environment steps"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of environment steps"""

    # Algorithm specific arguments
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    buffer_size: int = 1
    """the replay memory buffer size"""
    buffer_device: str = "cpu"
    """where the replay buffer is stored. Can be 'cpu' or 'cuda' for GPU"""
    gamma: float = 0.8
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 1024
    """the batch size of sample from the replay memory"""
    learning_starts: int = 4_000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    training_freq: int = 64
    """training frequency (in steps)"""
    utd: float = 0.5
    """update to data ratio"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    bootstrap_at_done: str = "always"
    """the bootstrap method to use when a done signal is received. Can be 'always' or 'never'"""

    # to be filled in runtime
    grad_steps_per_iteration: int = 0
    """the number of gradient updates per iteration"""
    steps_per_env: int = 0
    """the number of steps each parallel env takes per iteration"""


    # Misc
    obs_mode: str = "state"
    """the observation mode of the environment, can be 'state' or 'rgb+depth' or 'rgb+depth+segmentation'"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode of the environment, refer https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html for more details"""
    render_mode: str = "rgb_array"
    """the render mode of the environment, refer https://maniskill.readthedocs.io/en/latest/user_guide/reference/mani_skill.envs.sapien_env.html#mani_skill.envs.sapien_env.BaseEnv.SUPPORTED_RENDER_MODES for more details"""
    sim_backend: str = "gpu"
    """the simulation backend, refer https://maniskill.readthedocs.io/en/latest/user_guide/reference/mani_skill.envs.sapien_env.html for more details"""
    reward_mode: str = "dense"
    """the reward mode of the environment, can be 'dense' or 'sparse'"""
    robot_uids: str = "panda"
    """the uid of the robot to be used"""
    enable_shadow: bool = True
    """whether to enable shadow, can be True or False"""


@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
class ReplayBuffer:
    def __init__(self, env, num_envs: int, buffer_size: int, storage_device: torch.device, sample_device: torch.device):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        print("line 153")
        # self.obs = torch.zeros((buffer_size, num_envs) + env.single_observation_space.shape).to(storage_device)
        print("buffer_size",buffer_size)
        print("num_envs",num_envs)
        print("addition", (buffer_size, num_envs) + env.single_observation_space['sensor_data']['base_camera']['rgb'].shape)
        self.obs = torch.zeros((buffer_size, num_envs) + env.single_observation_space['sensor_data']['base_camera']['rgb'].shape).to(storage_device)
        print("line 155")
        self.next_obs = torch.zeros((buffer_size, num_envs) + env.single_observation_space.shape).to(storage_device)
        print("line 158")
        self.actions = torch.zeros((buffer_size, num_envs) + env.single_action_space.shape).to(storage_device)
        print("line 160")
        self.logprobs = torch.zeros((buffer_size, num_envs)).to(storage_device)
        print("line 162")
        self.rewards = torch.zeros((buffer_size, num_envs)).to(storage_device)
        print("line 164")
        self.dones = torch.zeros((buffer_size, num_envs)).to(storage_device)
        print("line 166")
        self.values = torch.zeros((buffer_size, num_envs)).to(storage_device)
        print("line 168")
    def add(self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor):
        if self.storage_device == torch.device("cpu"):
            obs = obs.cpu()
            next_obs = next_obs.cpu()
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs

        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    def sample(self, batch_size: int):
        if self.full:
            batch_inds = torch.randint(0, self.buffer_size, size=(batch_size, ))
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size, ))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size, ))
        return ReplayBufferSample(
            obs=self.obs[batch_inds, env_inds].to(self.sample_device),
            next_obs=self.next_obs[batch_inds, env_inds].to(self.sample_device),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device)
        )



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# class NatureCNN(nn.Module):
#     def __init__(self, sample_obs):
#         super().__init__()

#         extractors = {}

#         self.out_features = 0
#         feature_size = 256
#         in_channels=sample_obs["rgb"].shape[-1]
#         image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])


#         # here we use a NatureCNN architecture to process images, but any architecture is permissble here
#         cnn = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=in_channels,
#                 out_channels=32,
#                 kernel_size=8,
#                 stride=4,
#                 padding=0,
#             ),
#             nn.ReLU(),
#             nn.Conv2d(
#                 in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
#             ),
#             nn.ReLU(),
#             nn.Conv2d(
#                 in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
#             ),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         # to easily figure out the dimensions after flattening, we pass a test tensor
#         with torch.no_grad():
#             n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
#             fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
#         extractors["rgb"] = nn.Sequential(cnn, fc)
#         self.out_features += feature_size

#         if "state" in sample_obs:
#             # for state data we simply pass it through a single linear layer
#             state_size = sample_obs["state"].shape[-1]
#             extractors["state"] = nn.Linear(state_size, 256)
#             self.out_features += 256

#         self.extractors = nn.ModuleDict(extractors)

#     def forward(self, observations) -> torch.Tensor:
#         encoded_tensor_list = []
#         # self.extractors contain nn.Modules that do all the processing.
#         for key, extractor in self.extractors.items():
#             obs = observations[key]
#             if key == "rgb":
#                 obs = obs.float().permute(0,3,1,2)
#                 obs = obs / 255
#             encoded_tensor_list.append(extractor(obs))
#         return torch.cat(encoded_tensor_list, dim=1)


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        extractors = {}

        feature_size = 256
        in_channels=env.single_observation_space['sensor_data']['base_camera']['rgb'].shape[-1]
        # cnn = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=in_channels,
        #         out_channels=32,
        #         kernel_size=8,
        #         stride=4,
        #         padding=0,
        #     ),
        #     nn.ReLU(),
        #     nn.Conv2d(
        #         in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
        #     ),
        #     nn.ReLU(),
        #     nn.Conv2d(
        #         in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
        #     ),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )

        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
            # nn.Linear(9216, 256),
            # nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Linear(256, 1),
        )
        sample_obs, _ = env.reset()

        with torch.no_grad():
            n_flatten = cnn(sample_obs['sensor_data']['base_camera']["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
            self.fc = nn.Sequential(nn.Linear(n_flatten + np.prod(env.single_action_space.shape), feature_size), nn.ReLU(),
                               nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 1))
        extractors["rgb"] = nn.Sequential(cnn, self.fc)
        # self.out_features += feature_size

        self.extractors = nn.ModuleDict(extractors)


    def forward(self, observations, actions) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0,3,1,2)
                obs = obs / 255
            encoded_tensor_list.append(extractor(obs))

            encoded_tensor = torch.cat(encoded_tensor_list, dim=1)
            x = torch.cat([encoded_tensor, actions], dim=1)
        return self.fc(x)
    
    # def forward(self, x, a):
    #     encoded_tensor_list = []        
    #     x = torch.cat([x, a], 1)
    #     return self.net(x)
        

    #     # to easily figure out the dimensions after flattening, we pass a test tensor
    #     with torch.no_grad():
    #         n_flatten = cnn(env.single_observation_space['sensor_data']['base_camera']["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
    #         fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
    #     extractors["rgb"] = nn.Sequential(cnn, fc)
    #     self.out_features += feature_size

    #     if "state" in env.single_observation_space['sensor_data']:
    #         # for state data we simply pass it through a single linear layer
    #         state_size = env.single_observation_space['sensor_data']['base_camera']["state"].shape[-1]
    #         extractors["state"] = nn.Linear(state_size, 256)
    #         self.out_features += 256

    #     self.extractors = nn.ModuleDict(extractors)

    # def forward(self, observations) -> torch.Tensor:
    #     encoded_tensor_list = []
    #     # self.extractors contain nn.Modules that do all the processing.
    #     for key, extractor in self.extractors.items():
    #         obs = observations[key]
    #         if key == "rgb":
    #             obs = obs.float().permute(0,3,1,2)
    #             obs = obs / 255
    #         encoded_tensor_list.append(extractor(obs))
    #     return torch.cat(encoded_tensor_list, dim=1)


        


LOG_STD_MAX = 2
LOG_STD_MIN = -5

# ALGO LOGIC: initialize agent here:
class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()



        obs_space = env.single_observation_space['sensor_data']['base_camera']['rgb']
        obs_shape = obs_space.shape
        in_channels=env.single_observation_space['sensor_data']['base_camera']['rgb'].shape[-1]

        print("obs_shape", obs_shape)
        print("obs_shape", obs_shape)
        if obs_shape == (128, 128, 3):
            self.cnn = nn.Sequential(
                    nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=32,
                    kernel_size=8,
                    stride=4,
                    padding=0,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.Flatten(),

            )
            sample_obs, _ = env.reset()
            with torch.no_grad():
                n_flatten = self.cnn(sample_obs['sensor_data']['base_camera']["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
                self.backbone = nn.Sequential(nn.Linear(n_flatten, 256), nn.ReLU(),
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                    )
        else:
            raise ValueError("Expected image shape (128, 128, 3) for RGB data")



        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))
        # will be saved in the state_dict

    def forward(self, x):
        x = self.backbone(self.cnn(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_eval_action(self, x):
        x = self.backbone(self.cnn(x))
        mean = self.fc_mean(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

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

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


class Logger:
    
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        import wandb
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()


class SAC(Args):
    def __init__(self, Args):

        self.args = Args # instance variable 
        if self.args.exp_name is None:
            self.args.exp_name = os.path.basename(__file__)[: -len(".py")]
            self.run_name = f"{self.args.env_id}__{self.args.exp_name}__{self.args.seed}__{int(time.time())}"
        else:
            self.run_name = self.args.exp_name

        # TRY NOT TO MODIFY: seeding
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
        
        self.env_kwargs = dict(obs_mode=self.args.obs_mode, control_mode=self.args.control_mode, render_mode = self.args.render_mode, sim_backend=self.args.sim_backend)
        self.envs = gym.make(self.args.env_id, num_envs=self.args.num_envs if not self.args.evaluate else 1, **self.env_kwargs)
        self.eval_envs = gym.make(self.args.env_id, num_envs=self.args.num_eval_envs, reconfiguration_freq=self.args.reconfiguration_freq, **self.env_kwargs)
        if isinstance(self.envs.action_space, gym.spaces.Dict):
            self.envs= FlattenActionSpaceWrapper(self.envs)
            self.eval_envs = FlattenActionSpaceWrapper(self.eval_envs)
        if self.args.capture_video:
            self.eval_output_dir = f"runs/{self.run_name}/videos"
            if self.args.evaluate:
                self.eval_output_dir = f"{os.path.dirname(self.args.checkpoint)}/test_videos"
            print(f"Saving eval videos to {self.eval_output_dir}")
            if self.args.save_train_video_freq is not None:
                save_video_trigger = lambda x : (x // self.args.num_steps) % self.args.save_train_video_freq == 0
                self.envs = RecordEpisode(self.envs, output_dir=f"runs/{self.run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=self.args.num_steps, video_fps=30)
            self.eval_envs = RecordEpisode(self.eval_envs, output_dir=self.eval_output_dir, save_trajectory=self.args.evaluate, trajectory_name="trajectory", max_steps_per_video=self.args.num_eval_steps, video_fps=30)
        self.envs = ManiSkillVectorEnv(self.envs, self.args.num_envs, ignore_terminations=not self.args.partial_reset, record_metrics=True)
        self.eval_envs = ManiSkillVectorEnv(self.eval_envs, self.args.num_eval_envs, ignore_terminations=True, record_metrics=True)
        assert isinstance(self.envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        self.max_episode_steps = gym_utils.find_max_episode_steps_value(self.envs._env)
        self.logger = None
        if not self.args.evaluate:
            print("Running training")
            if self.args.track:
                import wandb
                
                print("Using wandb")

                config = vars(self.args)
                config["env_cfg"] = dict(**self.env_kwargs, num_envs=self.args.num_envs, env_id=self.args.env_id, reward_mode="normalized_dense", env_horizon=self.max_episode_steps, partial_reset=self.args.partial_reset)
                config["eval_env_cfg"] = dict(**self.env_kwargs, num_envs=self.args.num_eval_envs, env_id=self.args.env_id, reward_mode="normalized_dense", env_horizon=self.max_episode_steps, partial_reset=False)
                wandb.init(
                    project=self.args.wandb_project_name,
                    entity=self.args.wandb_entity,
                    sync_tensorboard=False,
                    config=config,
                    name=self.run_name,
                    save_code=True,
                    group="PPO",
                    tags=["ppo", "walltime_efficient"]
                )
            self.writer = SummaryWriter(f"runs/{self.run_name}")
            self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
            )
            self.logger = Logger(log_wandb=self.args.track, tensorboard=self.writer)
        else:
            print("Running evaluation")


        print("line 558 now")
        self.max_action = float(self.envs.single_action_space.high[0])

        self.actor = Actor(self.envs).to(self.device)
        self.qf1 = SoftQNetwork(self.envs).to(self.device)
        self.qf2 = SoftQNetwork(self.envs).to(self.device)
        self.qf1_target = SoftQNetwork(self.envs).to(self.device)
        self.qf2_target = SoftQNetwork(self.envs).to(self.device)
        if self.args.checkpoint is not None:
            ckpt = torch.load(self.args.checkpoint)
            self.actor.load_state_dict(ckpt['actor'])
            self.qf1.load_state_dict(ckpt['qf1'])
            self.qf2.load_state_dict(ckpt['qf2'])
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.args.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.args.policy_lr)

        print("line 576 now")

        # Automatic entropy tuning
        if self.args.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.envs.single_action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.args.q_lr)
        else:
            self.alpha = self.args.alpha

        self.envs.single_observation_space.dtype = np.float32
        print("line 588 now")
        self.rb = ReplayBuffer(
            env=self.envs,
            num_envs=self.args.num_envs,
            buffer_size=self.args.buffer_size,
            storage_device=torch.device(self.args.buffer_device),
            sample_device=self.device
        )

        print("line 596 now")
    

    def start_game(self):
        # TRY NOT TO MODIFY: start the game
        obs, info = self.envs.reset(seed=self.args.seed) # in Gymnasium, seed is given to reset() instead of seed()
        eval_obs, _ = self.eval_envs.reset(seed=self.args.seed)
        global_step = 0
        global_update = 0
        learning_has_started = False

        global_steps_per_iteration = self.args.num_envs * (self.args.steps_per_env)

        print("yo yo")
        while global_step < self.args.total_timesteps:
            print("yo yo 2")

            if self.args.eval_freq > 0 and (global_step - self.args.training_freq) // self.args.eval_freq < global_step // self.args.eval_freq:
                # evaluate
                self.actor.eval()
                print("Evaluating")
                eval_obs, _ = self.eval_envs.reset()
                eval_metrics = defaultdict(list)
                num_episodes = 0
                for _ in range(self.args.num_eval_steps):
                    with torch.no_grad():
                        eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = self.eval_envs.step(self.actor.get_eval_action(eval_obs))
                        if "final_info" in eval_infos:
                            mask = eval_infos["_final_info"]
                            num_episodes += mask.sum()
                            for k, v in eval_infos["final_info"]["episode"].items():
                                eval_metrics[k].append(v)
                print(f"Evaluated {self.args.num_eval_steps * self.args.num_eval_envs} steps resulting in {num_episodes} episodes")
                for k, v in eval_metrics.items():
                    mean = torch.stack(v).float().mean()
                    if self.logger is not None:
                        self.logger.add_scalar(f"eval/{k}", mean, global_step)
                    print(f"eval_{k}_mean={mean}")
                if self.args.evaluate:
                    break
                self.actor.train()

                if self.args.save_model:
                    model_path = f"runs/{self.run_name}/ckpt_{global_step}.pt"
                    torch.save({
                        'actor': self.actor.state_dict(),
                        'qf1': self.qf1_target.state_dict(),
                        'qf2': self.qf2_target.state_dict(),
                        'log_alpha': self.log_alpha,
                    }, model_path)
                    print(f"model saved to {model_path}")

            # Collect samples from environemnts
            rollout_time = time.time()
            for local_step in range(self.args.steps_per_env):
                global_step += 1 * self.args.num_envs

                # ALGO LOGIC: put action logic here
                if not learning_has_started:
                    actions = torch.tensor(self.envs.action_space.sample(), dtype=torch.float32, device=self.device)
                else:
                    actions, _, _ = self.actor.get_action(obs)
                    actions = actions.detach()

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)
                real_next_obs = next_obs.clone()
                if self.args.bootstrap_at_done == 'always':
                    next_done = torch.zeros_like(terminations).to(torch.float32)
                else:
                    next_done = (terminations | truncations).to(torch.float32)
                if "final_info" in infos:
                    final_info = infos["final_info"]
                    done_mask = infos["_final_info"]
                    real_next_obs[done_mask] = infos["final_observation"][done_mask]
                    for k, v in final_info["episode"].items():
                        self.logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

                self.rb.add(obs, real_next_obs, actions, rewards, next_done)

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_obs
            rollout_time = time.time() - rollout_time

            # ALGO LOGIC: training.
            if global_step < self.args.learning_starts:
                continue

            update_time = time.time()
            learning_has_started = True
            print("learning has started")
            for local_update in range(self.args.grad_steps_per_iteration):
                global_update += 1
                data = self.rb.sample(self.args.batch_size)

                # update the value networks
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_obs)
                    qf1_next_target = self.qf1_target(data.next_obs, next_state_actions)
                    qf2_next_target = self.qf2_target(data.next_obs, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.args.gamma * (min_qf_next_target).view(-1)
                    # data.dones is "stop_bootstrap", which is computed earlier according to self.args.bootstrap_at_done

                qf1_a_values = self.qf1(data.obs, data.actions).view(-1)
                qf2_a_values = self.qf2(data.obs, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                self.q_optimizer.zero_grad()
                qf_loss.backward()
                self.q_optimizer.step()

                # update the policy network
                if global_update % self.args.policy_frequency == 0:  # TD 3 Delayed update support
                    pi, log_pi, _ = self.actor.get_action(data.obs)
                    qf1_pi = self.qf1(data.obs, pi)
                    qf2_pi = self.qf2(data.obs, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    if self.args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = self.actor.get_action(data.obs)
                        # if self.args.correct_alpha:
                        alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()
                        # else:
                        #     alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()
                        # log_alpha has a legacy reason: https://github.com/rail-berkeley/softlearning/issues/136#issuecomment-619535356

                        self.a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.a_optimizer.step()
                        self.alpha = self.log_alpha.exp().item()

                # update the target networks
                if global_update % self.args.target_network_frequency == 0:
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
                    for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                        target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
            update_time = time.time() - update_time

            # Log training-related data
            if (global_step - self.args.training_freq) // self.args.log_freq < global_step // self.args.log_freq:
                print(f"Global Step: {global_step}")
                self.logger.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                self.logger.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                self.logger.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                self.logger.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                self.logger.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                self.logger.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                self.logger.add_scalar("losses/alpha", self.alpha, global_step)
                self.logger.add_scalar("charts/update_time", update_time, global_step)
                self.logger.add_scalar("charts/rollout_time", rollout_time, global_step)
                self.logger.add_scalar("charts/rollout_fps", global_steps_per_iteration / rollout_time, global_step)
                if self.args.autotune:
                    self.logger.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        if not self.args.evaluate and self.args.save_model:
            model_path = f"runs/{self.run_name}/final_ckpt.pt"
            torch.save({
                'actor': self.actor.state_dict(),
                'qf1': self.qf1_target.state_dict(),
                'qf2': self.qf2_target.state_dict(),
                'log_alpha': self.log_alpha,
            }, model_path)
            print(f"model saved to {model_path}")
            self.writer.close()
        self.envs.close()




"""
if __name__ == "__main__":

    args = tyro.cli(Args)
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(obs_mode="state", control_mode="pd_joint_delta_pos", render_mode="rgb_array", sim_backend="gpu")
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=True, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=False)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group="PPO",
                tags=["ppo", "walltime_efficient"]
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        actor.load_state_dict(ckpt['actor'])
        qf1.load_state_dict(ckpt['qf1'])
        qf2.load_state_dict(ckpt['qf2'])
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    self.actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

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
        env=envs,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=device
    )


    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(seed=args.seed) # in Gymnasium, seed is given to reset() instead of seed()
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    global_step = 0
    global_update = 0
    learning_has_started = False

    global_steps_per_iteration = args.num_envs * (args.steps_per_env)

    while global_step < args.total_timesteps:
        if args.eval_freq > 0 and (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            # evaluate
            actor.eval()
            print("Evaluating")
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(actor.get_eval_action(eval_obs))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")
            if args.evaluate:
                break
            actor.train()

            if args.save_model:
                model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
                torch.save({
                    'actor': actor.state_dict(),
                    'qf1': qf1_target.state_dict(),
                    'qf2': qf2_target.state_dict(),
                    'log_alpha': log_alpha,
                }, model_path)
                print(f"model saved to {model_path}")

        # Collect samples from environemnts
        rollout_time = time.time()
        for local_step in range(args.steps_per_env):
            global_step += 1 * args.num_envs

            # ALGO LOGIC: put action logic here
            if not learning_has_started:
                actions = torch.tensor(envs.action_space.sample(), dtype=torch.float32, device=device)
            else:
                actions, _, _ = actor.get_action(obs)
                actions = actions.detach()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            real_next_obs = next_obs.clone()
            if args.bootstrap_at_done == 'always':
                next_done = torch.zeros_like(terminations).to(torch.float32)
            else:
                next_done = (terminations | truncations).to(torch.float32)
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                real_next_obs[done_mask] = infos["final_observation"][done_mask]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

            rb.add(obs, real_next_obs, actions, rewards, next_done)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
        rollout_time = time.time() - rollout_time

        # ALGO LOGIC: training.
        if global_step < args.learning_starts:
            continue

        update_time = time.time()
        learning_has_started = True
        for local_update in range(args.grad_steps_per_iteration):
            global_update += 1
            data = rb.sample(args.batch_size)

            # update the value networks
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_obs)
                qf1_next_target = qf1_target(data.next_obs, next_state_actions)
                qf2_next_target = qf2_target(data.next_obs, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                # data.dones is "stop_bootstrap", which is computed earlier according to args.bootstrap_at_done

            qf1_a_values = qf1(data.obs, data.actions).view(-1)
            qf2_a_values = qf2(data.obs, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # update the policy network
            if global_update % args.policy_frequency == 0:  # TD 3 Delayed update support
                pi, log_pi, _ = actor.get_action(data.obs)
                qf1_pi = qf1(data.obs, pi)
                qf2_pi = qf2(data.obs, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.obs)
                    # if args.correct_alpha:
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    # else:
                    #     alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()
                    # log_alpha has a legacy reason: https://github.com/rail-berkeley/softlearning/issues/136#issuecomment-619535356

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_update % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        update_time = time.time() - update_time

        # Log training-related data
        if (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            print(f"Global Step: {global_step}")
            logger.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            logger.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            logger.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            logger.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            logger.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            logger.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            logger.add_scalar("losses/alpha", alpha, global_step)
            logger.add_scalar("charts/update_time", update_time, global_step)
            logger.add_scalar("charts/rollout_time", rollout_time, global_step)
            logger.add_scalar("charts/rollout_fps", global_steps_per_iteration / rollout_time, global_step)
            if args.autotune:
                logger.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if not args.evaluate and args.save_model:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save({
            'actor': actor.state_dict(),
            'qf1': qf1_target.state_dict(),
            'qf2': qf2_target.state_dict(),
            'log_alpha': log_alpha,
        }, model_path)
        print(f"model saved to {model_path}")
        writer.close()
    envs.close()
"""
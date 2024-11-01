"""
import subprocess

# This script will be executed before the notebook is run, to set up the environment
# Setup Vulkan
subprocess.run(["mkdir", "-p", "/usr/share/vulkan/icd.d"], check=True)
subprocess.run(["wget", "-q", "https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/nvidia_icd.json"], check=True)
subprocess.run(["wget", "-q", "https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/10_nvidia.json"], check=True)
subprocess.run(["mv", "nvidia_icd.json", "/usr/share/vulkan/icd.d"], check=True)
subprocess.run(["mv", "10_nvidia.json", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"], check=True)
subprocess.run(["apt-get", "install", "-y", "--no-install-recommends", "libvulkan-dev"], check=True)

# Install Python dependencies
subprocess.run(["pip", "install", "--upgrade", "mani_skill", "tyro"], check=True)
"""

### Make sure to restart the notebook if you already ran a CPU sim!! ###
# Import required packages
import numpy as np
import gymnasium as gym
from tqdm.notebook import tqdm
import mani_skill.envs
import torch
import matplotlib.pyplot as plt
import time
from sac_image import Args, SAC
# from sac_joints import Args, SAC

env_id = "PickCube-v1"
obs_mode = "rgb"
control_mode = "pd_joint_delta_pos"
reward_mode = "sparse"
robot_uids = "panda"
num_envs = 1
num_eval_envs = 1


try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    import site
    site.main() # run this so local pip installs are recognized


def main():


    args = Args(env_id=env_id, num_envs=num_envs, num_eval_envs=num_eval_envs, obs_mode=obs_mode, control_mode=control_mode, 
                reward_mode=reward_mode, robot_uids=robot_uids, enable_shadow=True)

    print("hey not even till here??")
    sac_agent = SAC(args)
    try:
        sac_agent.start_game()
    except KeyboardInterrupt:
        print("Training interrupted manually.")
    except Exception as e:
        print(e)

    # env.unwrapped.print_sim_details()
    # obs, info_reset = env.reset()
    # print("Observation Space:", obs)
    # print("Info:", info_reset)
    # print("Action Space:", env.action_space)
    # done = False
    # start_time = time.time()
    # total_rew = 0
    # obs, rew, terminated, truncated, info = env.step(torch.from_numpy(env.action_space.sample()))
    # # while not done:
    # #     # note that env.action_space is now a batched action space
    # #     obs, rew, terminated, truncated, info = env.step(torch.from_numpy(env.action_space.sample()))
    # #     done = (terminated | truncated).any() # stop if any environment terminates/truncates
    # N = num_envs * info["elapsed_steps"][0].item()
    # dt = time.time() - start_time
    # FPS = N / (dt)
    # print(f"Frames Per Second = {N} / {dt} = {FPS}")


if __name__ == "__main__":
    main()
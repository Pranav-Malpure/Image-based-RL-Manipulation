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
import gymnasium as gym
import mani_skill.envs as maniskill
import torch
import time

env_id = "StackCube-v1"
obs_mode = "rgb+depth+segmentation"
control_mode = "pd_joint_delta_pos"
reward_mode = "sparse"
robot_uids = "Panda"



num_envs = 2048 # you can go up to 4096 on better GPUs


try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    import site
    site.main() # run this so local pip installs are recognized


def main():
    env = gym.make(env_id, num_envs=num_envs, obs_mode=obs_mode, control_mode=control_mode, 
                   reward_mode=reward_mode, robot_uids=robot_uids, enable_shadows=True)
    env.unwrapped.print_sim_details()
    obs, _ = env.reset()
    print("Action Space:", env.action_space)
    done = False
    start_time = time.time()
    total_rew = 0
    while not done:
        # note that env.action_space is now a batched action space
        obs, rew, terminated, truncated, info = env.step(torch.from_numpy(env.action_space.sample()))
        done = (terminated | truncated).any() # stop if any environment terminates/truncates
    N = num_envs * info["elapsed_steps"][0].item()
    dt = time.time() - start_time
    FPS = N / (dt)
    print(f"Frames Per Second = {N} / {dt} = {FPS}")

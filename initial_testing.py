import subprocess

# Setup Vulkan
subprocess.run(["mkdir", "-p", "/usr/share/vulkan/icd.d"], check=True)
subprocess.run(["wget", "-q", "https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/nvidia_icd.json"], check=True)
subprocess.run(["wget", "-q", "https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/10_nvidia.json"], check=True)
subprocess.run(["mv", "nvidia_icd.json", "/usr/share/vulkan/icd.d"], check=True)
subprocess.run(["mv", "10_nvidia.json", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"], check=True)
subprocess.run(["apt-get", "install", "-y", "--no-install-recommends", "libvulkan-dev"], check=True)

# Install Python dependencies
subprocess.run(["pip", "install", "--upgrade", "mani_skill", "tyro"], check=True)


### Make sure to restart the notebook if you already ran a CPU sim!! ###
# Import required packages
import gymnasium as gym
import mani_skill.envs as maniskill
import torch
import time
num_envs = 2048 # you can go up to 4096 on better GPUs
env = gym.make("PickCube-v1", num_envs=num_envs)
env.unwrapped.print_sim_details()
obs, _ = env.reset(seed=0)
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

print("hello")

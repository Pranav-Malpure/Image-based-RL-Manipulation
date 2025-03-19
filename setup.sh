#!/bin/bash

# Remove the existing mani_skill directory
rm -rf mani_skill/

# Clone the repository with submodules
git clone --recurse-submodules https://github.com/Pranav-Malpure/Image-based-RL-Manipulation

# Navigate into the Image-based-RL-Manipulation/ManiSkill directory
cd Image-based-RL-Manipulation/ManiSkill

# Checkout the xarm_allegro branch
git checkout xarm_allegro

# Move the mani_skill directory to the correct location
mv mani_skill/ /workspace/ManiSkill/

# Go back to the previous directory
cd ..

# Run the Python script with the provided arguments
python sac_rgbd_aug.py --env_id="PickCube-v1" --obs_mode="rgb" --num_envs=32 --utd=0.5 --buffer_size=300_000 --control-mode="pd_ee_delta_pos" --camera_width=64 --camera_height=64 --total_timesteps=500_000 --eval_freq=10_000
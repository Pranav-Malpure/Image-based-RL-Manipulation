# Image-based-RL-Manipulation

Based on Maniskill. The files above are the iterations underway to try out different techniques.

## Implemented so far...
SADA on maniskill: `sac_rgbd_sada.py`

DrQ v2 replay buffer: `efficient_replay_buffer.py`

Allegro hand cube grabbing: _Mix of files from env and robot agents_, but specifically these two files: [env code](https://github.com/Pranav-Malpure/ManiSkill/blob/xarm_allegro/mani_skill/envs/tasks/tabletop/pick_cube.py), [robot code](https://github.com/Pranav-Malpure/ManiSkill/blob/xarm_allegro/mani_skill/agents/robots/xarm6/xarm6_allegro.py)

DP3 policy in Maniskill(underway), in [diffusion_policy_3d](https://github.com/Pranav-Malpure/ManiSkill/tree/c2d3d9ae4f0ae466b825583335770a08bd16ea74/examples/baselines/diffusion_policy_3d)

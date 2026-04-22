# MuJoCo Robot Simulation and Data Collection

Simulates a Franka Emika Panda robot in MuJoCo, allowing keyboard control for data collection. Trajectories are recorded and saved as a LeRobot dataset.

## Dependencies

mujoco
pygame
datasets
numpy
matplotlib

## Usage

1. Run `python main.py` to start the simulation and collect data.
2. Use keyboard controls to move the robot and collect episodes. Make sure to focus on the pygame control tab.
3. Dataset is saved to `./lerobot_demo_dataset`.
4. Run `python verification.py` to visualize the saved dataset

## Controls

- W/A/S/D/Up/Down: Move gripper
- left/right arrow: Rotate gripper
- Space: Toggle gripper
- R: Reset episode
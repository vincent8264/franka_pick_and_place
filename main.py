import mujoco
import mujoco.viewer
import numpy as np
import time
import pygame
from datasets import Dataset
from utils import *

# Loading and initialization
model = mujoco.MjModel.from_xml_path("./franka_emika_panda/scene.xml")
data = mujoco.MjData(model)

gripper_site_id = model.site('hand_tcp').id
arm_dof = 7

initial_angles = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.57, 0.785])
initial_gripper_ori = np.array([0, 1.0, 0, 0])  # Facing downwards
data.qpos[:arm_dof] = initial_angles
data.ctrl[:arm_dof] = initial_angles
mujoco.mj_forward(model, data)

# Initialize Pygame for keyboard input
pygame.init()
pygame.display.set_caption("Robot Controls")
screen = pygame.display.set_mode((300, 100))

# Data collection setup
recording_fps = 30
record_interval = 1.0 / recording_fps
last_record_time = 0.0

episode_idx = 0
frame_idx = 0

trajectory = { #required trajectory
    "timestamp": [],
    "observation.state": [],            # Robot joint states + gripper state
    "observation.environment_state": [],# Object positions and velocities
    "action": [],                       # Commands sent to the robot
    "episode_index": [],
    "frame_index": []
}

metadata = {
    "env_name": "mujoco_pick_and_place",
    "demo_id": f"demo_{episode_idx}",
    "robot": "franka_emika_panda"
}

# Main simulation loop
target_pos = data.site_xpos[gripper_site_id].copy()
target_quat = initial_gripper_ori.copy()
move_speed = 0.2
rot_speed = 2
gripper_open = True

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # 1. Get Inputs
        v, w, gripper_open, reset = get_keyboard_commands(viewer, gripper_open)

        # Reset trigger
        if reset:
            print(f"Ending episode {episode_idx}. Resetting environment...")
            
            episode_idx += 1
            frame_idx = 0
            
            # Clear all positions, velocities, and forces
            mujoco.mj_resetData(model, data) 
            data.qpos[:arm_dof] = initial_angles
            data.ctrl[:arm_dof] = initial_angles

            mujoco.mj_forward(model, data) 
            
            # Reset Target Variables
            target_pos = data.site_xpos[gripper_site_id].copy()
            target_quat = initial_gripper_ori.copy()
            
            # Skip the rest of the loop for this frame to avoid recording the transition
            continue

        # 2. Update Target Position
        target_pos += v * move_speed * model.opt.timestep

        # if target position drifts too far from current position (hit an obstacle), reset it to current position
        if np.linalg.norm(target_pos - data.site_xpos[gripper_site_id]) > 0.05:
            target_pos = data.site_xpos[gripper_site_id].copy()

        # 3. Update Target Orientation
        if w != 0.0:
            # Create a quaternion representing the angle change around the Z-axis [0, 0, 1]
            angle = w * rot_speed * model.opt.timestep
            axis_of_rotation = np.array([0.0, 0.0, 1.0])
            delta_quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(delta_quat, axis_of_rotation, angle)

            # Multiply target_quat by delta_quat
            new_target_quat = np.zeros(4)
            mujoco.mju_mulQuat(new_target_quat, delta_quat, target_quat)
            target_quat = new_target_quat / np.linalg.norm(new_target_quat) # Normalize to prevent drift

        # 4. Calculate inverse kinematics and apply controls
        dq = ik_step(model, data, gripper_site_id, target_pos, target_quat)
        data.ctrl[:arm_dof] = data.qpos[:arm_dof] + dq 
        data.ctrl[7] = 255 if gripper_open else 0.0 

        # 5. Step Simulation
        mujoco.mj_step(model, data)
        viewer.sync()

        # 6. Record Data in simulation time frame intervals
        current_time = data.time
        if current_time - last_record_time >= record_interval:
            # Timestamp
            trajectory["timestamp"].append(current_time)
            trajectory["episode_index"].append(episode_idx)
            trajectory["frame_index"].append(frame_idx)

            # Robot joint states + gripper state
            gripper_val = 1.0 if gripper_open else 0.0 
            robot_state = np.concatenate([data.qpos[:arm_dof], [gripper_val]])
            trajectory["observation.state"].append(robot_state)

            # Object positions and velocities
            env_state = []
            for cube_name in ["cube1", "cube2", "cube3"]:
                pos = data.body(cube_name).xpos.copy()
                # data.cvel last 3 for linear velocity
                vel = data.cvel[model.body(cube_name).id][3:].copy() 
                env_state.extend(pos)
                env_state.extend(vel)
            trajectory["observation.environment_state"].append(np.array(env_state))

            # Action
            action = data.ctrl.copy() 
            trajectory["action"].append(action)
            
            last_record_time = current_time
            frame_idx += 1

        # 7. Sync UI with Real Time for better input handling
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

pygame.quit()

# Save Dataset
# Convert to float32 numpy arrays for LeRobot compatibility
for key in trajectory:
    if key not in ["episode_index", "frame_index", "timestamp"]:
        trajectory[key] = np.array(trajectory[key], dtype=np.float32)

# Create Hugging Face Dataset
hf_dataset = Dataset.from_dict(trajectory)
hf_dataset.info.description = str(metadata)

# Save to disk
dataset_path = "./lerobot_demo_dataset"
hf_dataset.save_to_disk(dataset_path)

print(f"Dataset saved to {dataset_path}")
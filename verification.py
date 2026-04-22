from datasets import load_from_disk
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
dataset = load_from_disk("./lerobot_demo_dataset")

print(f"Total Frames: {len(dataset)}")
print(f"Features: {dataset.column_names}")

# Sample check
state_sample = np.array(dataset[0]["observation.state"])
print(f"Robot State Shape: {state_sample.shape}")

env_sample = np.array(dataset[0]["observation.environment_state"])
print(f"Env State Shape: {env_sample.shape}")

action_sample = np.array(dataset[0]["action"])
print(f"Action Shape: {action_sample.shape}")

# Visualize joint angles and gripper state over time, only the first episode
plt.figure(figsize=(12, 6))
time = dataset["timestamp"][:] 
joint_angles = np.array(dataset["observation.state"][:])[:, :7]
gripper_state = np.array(dataset["observation.state"][:])[:, 7]
plt.plot(time, gripper_state, label='Gripper State', linestyle='--')
for i in range(7):
    plt.plot(time, joint_angles[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time (s)')  
    plt.ylabel('Value')
    plt.title('Robot Joint Angles and Gripper State Over Time')
    plt.legend()
    plt.grid()
plt.show()
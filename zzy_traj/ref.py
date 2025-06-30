import numpy as np
import torch
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

position = np.load("./position_ned.npy")
velocity = np.load("./velocity_ned.npy")
position_tensor = torch.tensor(position, dtype=torch.float32)
velocity_tensor = torch.tensor(velocity, dtype=torch.float32)
for _ in range(6):
    position_tensor = torch.roll(position_tensor, shifts=-1, dims=0)
    velocity_tensor = torch.roll(velocity_tensor, shifts=-1, dims=0)
position_tensor = position_tensor.reshape(-1, 7, 3)
velocity_tensor = velocity_tensor.reshape(-1, 7, 3)
print(position_tensor)


def plot_trajectory(positions, velocities):
    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    
    # Plot positions
    axs[0, 0].plot(positions[:, 0], label='X Position')
    axs[0, 0].plot(positions[:, 1], label='Y Position')
    axs[0, 0].plot(positions[:, 2], label='Z Position')
    axs[0, 0].set_title('Position Trajectory')
    axs[0, 0].set_xlabel('Time Step')
    axs[0, 0].set_ylabel('Position (m)')
    axs[0, 0].legend()

    # Plot velocity norm as a colormap (hot) along the trajectory
    velocity_norm = np.linalg.norm(velocities, axis=1)
    points = positions[:, :2].reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='hot', norm=plt.Normalize(velocity_norm.min(), velocity_norm.max()))
    lc.set_array(velocity_norm[:-1])
    lc.set_linewidth(2)
    line = axs[2, 1].add_collection(lc)
    axs[2, 1].set_xlim(positions[:, 0].min(), positions[:, 0].max())
    axs[2, 1].set_ylim(positions[:, 1].min(), positions[:, 1].max())
    axs[2, 1].set_title('Trajectory (X-Y) colored by Velocity Magnitude')
    axs[2, 1].set_xlabel('X Position (m)')
    axs[2, 1].set_ylabel('Y Position (m)')
    cbar = plt.colorbar(line, ax=axs[2, 1])
    cbar.set_label('Velocity Magnitude (m/s)')
    
    # Plot velocities
    axs[1, 0].plot(velocities[:, 0], label='X Velocity')
    axs[1, 0].plot(velocities[:, 1], label='Y Velocity')
    axs[1, 0].plot(velocities[:, 2], label='Z Velocity')
    axs[1, 0].set_title('Velocity Trajectory')
    axs[1, 0].set_xlabel('Time Step')
    axs[1, 0].set_ylabel('Velocity (m/s)')
    axs[1, 0].legend()
    
    # Plot X vs Y position
    axs[0, 1].plot(positions[:, 0], positions[:, 1])
    axs[0, 1].set_title('X vs Y Position')
    axs[0, 1].set_xlabel('X Position (m)')
    axs[0, 1].set_ylabel('Y Position (m)')
    
    # Plot X vs Z position
    axs[1, 1].plot(positions[:, 0], positions[:, 2])
    axs[1, 1].set_title('X vs Z Position')
    axs[1, 1].set_xlabel('X Position (m)')
    axs[1, 1].set_ylabel('Z Position (m)')
    
    # Plot Y vs Z position
    axs[2, 0].plot(positions[:, 1], positions[:, 2])
    axs[2, 0].set_title('Y vs Z Position')
    axs[2, 0].set_xlabel('Y Position (m)')
    axs[2, 0].set_ylabel('Z Position (m)')
    
    # NEW subplot: XY position with velocity direction arrows
    velocity_norm = velocities / np.linalg.norm(velocities, axis=1, keepdims=True)  # Normalize velocity vectors
    axs[2, 2].quiver(
        positions[:, 0], positions[:, 1],  # positions
        velocity_norm[:, 0], velocity_norm[:, 1],  # velocity directions
        angles='xy', scale_units='xy', scale=1, width=0.003, color='blue'
    )
    axs[2, 2].set_title('X-Y Trajectory with Velocity Directions')
    axs[2, 2].set_xlabel('X Position (m)')
    axs[2, 2].set_ylabel('Y Position (m)')
    axs[2, 2].axis('equal')    

    plt.tight_layout()
    plt.savefig('trajectory_plot.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # Convert numpy arrays to torch tensors
    position_tensor = torch.tensor(position, dtype=torch.float32)
    velocity_tensor = torch.tensor(velocity, dtype=torch.float32)

    # Print shapes
    print(f"Position shape: {position_tensor.shape}")
    print(f"Velocity shape: {velocity_tensor.shape}")

    # Plot the trajectory
    plot_trajectory(position, velocity)
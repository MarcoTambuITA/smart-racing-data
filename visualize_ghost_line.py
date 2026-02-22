import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os

def load_track(track_path):
    # Load track data: x_m, y_m, w_tr_right_m, w_tr_left_m
    data = np.genfromtxt(track_path, delimiter=',', skip_header=1)
    x = data[:, 0]
    y = data[:, 1]
    w_right = data[:, 2]
    w_left = data[:, 3]
    return x, y, w_right, w_left

def load_trajectory(traj_path):
    # Load trajectory data: s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
    data = np.genfromtxt(traj_path, delimiter=';', skip_header=3)
    x = data[:, 1]
    y = data[:, 2]
    v = data[:, 5]
    a = data[:, 6]
    return x, y, v, a

def get_track_boundaries(x, y, w_right, w_left):
    # Calculate normal vectors for boundaries
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    # Normal vectors (perp to tangent)
    nx = -dy
    ny = dx
    norm = np.sqrt(nx**2 + ny**2)
    nx /= (norm + 1e-9)
    ny /= (norm + 1e-9)
    
    x_right = x + nx * w_right
    y_right = y + ny * w_right
    x_left = x - nx * w_left
    y_left = y - ny * w_left
    
    return x_right, y_right, x_left, y_left

def create_pilot_map(track_file, traj_file, output_file):
    print(f"Loading track: {track_file}")
    tx, ty, w_r, w_l = load_track(track_file)
    br_x, br_y, bl_x, bl_y = get_track_boundaries(tx, ty, w_r, w_l)
    
    print(f"Loading trajectory: {traj_file}")
    x, y, v, a = load_trajectory(traj_file)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(20, 15)) # Larger for better detail
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#1a1a1a')
    
    # Plot boundaries
    ax.plot(br_x, br_y, color='#444444', linewidth=1.5, ls='--')
    ax.plot(bl_x, bl_y, color='#444444', linewidth=1.5, ls='--')
    
    # Close the track loop
    ax.plot([br_x[-1], br_x[0]], [br_y[-1], br_y[0]], color='#444444', linewidth=1.5, ls='--')
    ax.plot([bl_x[-1], bl_x[0]], [bl_y[-1], bl_y[0]], color='#444444', linewidth=1.5, ls='--')

    # Create segments for the color line
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Normalize speed for color mapping
    norm = plt.Normalize(v.min(), v.max())
    lc = LineCollection(segments, cmap='RdYlGn', norm=norm)
    lc.set_array(v)
    lc.set_linewidth(6) # Thicker line
    line = ax.add_collection(lc)
    
    # Identify BRAKING ENTRY POINTS (where acceleration drops below -1.0 after being above)
    braking_threshold = -1.0
    braking_mask = a < braking_threshold
    
    # Detection of entry (mask current is true, mask previous is false)
    entry_points_idx = []
    for i in range(1, len(braking_mask)):
        if braking_mask[i] and not braking_mask[i-1]:
            entry_points_idx.append(i)
            
    # Highlight full braking areas with subtle shadow
    ax.scatter(x[braking_mask], y[braking_mask], color='cyan', s=20, alpha=0.3, label='Braking Zone')

    # Plot large, distinct Cyan indicators for ENTRY points
    if entry_points_idx:
        ax.scatter(x[entry_points_idx], y[entry_points_idx], color='#00ffff', s=250, 
                   edgecolor='white', linewidth=2, marker='X', label='BRAKING ENTRY', zorder=10)


    # Highlight full throttle zones (a > 0.5)
    throttle = a > 0.8
    ax.scatter(x[throttle], y[throttle], color='white', s=5, alpha=0.5, label='Full Throttle')

    # Colorbar
    cbar = fig.colorbar(line, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Target Speed (m/s)', color='white', size=14)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    ax.set_title(f'Pilot Implementation Guide: {os.path.basename(track_file)}', color='white', fontsize=24, pad=20)
    ax.set_xlabel('Coordinates East (m)', color='white')
    ax.set_ylabel('Coordinates North (m)', color='white')
    ax.tick_params(colors='white')
    ax.axis('equal')
    ax.grid(color='#333333', linestyle=':', alpha=0.5)
    
    # Legend
    ax.legend(loc='upper right', facecolor='#222222', edgecolor='white', labelcolor='white', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Detailed map saved to {output_file}")

if __name__ == "__main__":
    track = "global_racetrajectory_optimization-master/inputs/tracks/berlin_2018.csv"
    traj = "global_racetrajectory_optimization-master/outputs/traj_race_cl.csv"
    output = "pilot_ghost_line.png"
    
    if os.path.exists(track) and os.path.exists(traj):
        create_pilot_map(track, traj, output)
    else:
        print("Required files not found. Check paths.")

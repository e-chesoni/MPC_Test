from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from matplotlib.patches import Rectangle


def skid_steer_MPC_animation(x, x_d, tf, n_frames=60):
    # Sample desired trajectory
    n_samples = 1000
    t_samples = np.linspace(0.0, tf, n_samples)
    x_des = np.zeros((n_samples, 3))

    for i in range(t_samples.shape[0]):
        x_des[i] = x_d

    from matplotlib import rc
    rc('animation', html='jshtml')

    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes()

    x_max = max(np.max(x_des[:, 0]), np.max(x[:, 0]))
    x_min = min(np.min(x_des[:, 0]), np.min(x[:, 0]))
    y_max = max(np.max(x_des[:, 1]), np.max(x[:, 1]))
    y_min = min(np.min(x_des[:, 1]), np.min(x[:, 1]))

    frame_idx = [round(x) for x in np.linspace(0, x.shape[0] - 1, n_frames).tolist()]
    x_anim = np.zeros((n_frames, 3))
    for i in range(n_frames):
        x_anim[i, :] = x[frame_idx[i], :]

    a = 0.25
    y = x_anim[:, 0]
    z = x_anim[:, 1]
    theta = x_anim[:, 2]

    x_padding = 0.25 * (x_max - x_min)
    y_padding = 0.25 * (y_max - y_min)

    def frame(i):
        ax.clear()

        ax.plot(x_des[0, 0], x_des[0, 1], 'b*', label='desired position')
        ax.plot(x_anim[:i + 1, 0], x_anim[:i + 1, 1], '--', label='actual trajectory')

        # THIS IS TO MAKE THE BOX
        # Calculate the center of the box

        center_y = y[i]
        center_z = z[i]

        # Define the width and height of the box
        box_width = a * 0.025
        box_height = a * 0.05

        # Calculate the orientation angle (in degrees)
        angle_deg = degrees(theta[i])

        # Create a rectangle
        # The anchor point (bottom left corner of the rectangle) is calculated by subtracting half the width and height from the center coordinates
        rect = Rectangle((center_y - box_width / 16, center_z - box_height / 16), box_width, box_height, color='green',
                         angle=angle_deg)

        # Calculate the coordinates for the front part of the box
        front_height = box_height / 4

        # Create a rectangle for the front part with a different color (e.g., red)
        front_rect = Rectangle((center_y - box_width / 16, center_z - front_height / 16), box_width, front_height,
                        color='red', angle=angle_deg)

        ax.add_patch(rect)
        ax.add_patch(front_rect)
        plot = [rect]  # plot needs to be a list of artists
        # END ADDED BOX CODE

        if (np.abs((x_max - x_min) - (y_max - y_min)) < 5):
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
        ax.set_xlabel('y (m)')
        ax.set_ylabel('z (m)')
        ax.set_aspect('equal')
        ax.legend(loc='upper left')

        return plot

    return animation.FuncAnimation(fig, frame, frames=n_frames, blit=False, repeat=False), fig


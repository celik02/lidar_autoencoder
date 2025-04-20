import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from dataloader import LidarDataset
import numpy as np
from scipy.spatial.transform import Rotation as R

plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['figure.dpi'] = 100
data_dir = 'lidardata2/'


def parse_timestamp(fname):
    # expects: lidar_data_YYYYMMDD-HHMMSS.csv
    ts = os.path.basename(fname).replace('lidar_data_', '').replace('.csv', '')
    return datetime.datetime.strptime(ts, '%Y%m%d-%H%M%S')


def plot_scan_points(beam_idx):
    ds = LidarDataset(csv_path_list=data_dir)
    times, ranges7, angles7, xs7, ys7 = [], [], [], [], []
    for df in ds.scans:
        times.append(parse_timestamp(df.loc[0, 'filename']))
        # drop infinites
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        r = df.loc[beam_idx, 'range']         # raw range at beam 7
        a = df.loc[beam_idx, 'angle']         # angle at beam 7
        # convert to Cartesian
        x = r * np.cos(a)
        y = r * np.sin(a)

        ranges7.append(r)
        angles7.append(a)
        xs7.append(x)
        ys7.append(y)

    # Example: plot x-coordinate over time (smoother)
    df_plot = pd.DataFrame({
        'time': times,
        'x7': xs7,
        'y7': ys7,
        'range7': ranges7,
        'angle7': angles7
    }).set_index('time')

    plt.figure(figsize=(8, 4))
    df_plot['x7'].plot(label='x (m)')
    df_plot['y7'].plot(label='y (m)')
    df_plot['range7'].plot(label='range (m)')
    df_plot['angle7'].plot(label='angle (rad)')
    plt.xlabel('Scan time')
    plt.ylabel(f'Position of {beam_idx}th beam')
    plt.title('LiDAR 7th-point Cartesian over time')
    plt.legend()
    plt.grid(True)
    plt.show()


def load_quaternion(fullpath):
    # read only the orientation line
    with open(fullpath, 'r') as f:
        f.readline()  # skip position
        line = f.readline().strip()
    parts = line.split(',')
    # parts = ['orientation', qx, qy, qz, qw]
    qx, qy, qz, qw = map(float, parts[1:])
    return np.array([qx, qy, qz, qw])


# I want to plot all the points in one scan, highlighting the 7th beam
def plot_scan_points_all(beam_idx=7):
    ds = LidarDataset(csv_path_list=data_dir)
    csv_dir = data_dir
    for df in ds.scans:
        # extract angles & ranges
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        angles = df['angle'].values
        ranges = df['range'].values

        # convert to Cartesian
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        pts = np.stack([xs, ys, np.zeros_like(xs)], axis=1)  # [N,3]

        fname0 = df.loc[0, 'filename']
        full0 = os.path.join(csv_dir, fname0)
        # 2) get quaternion + rotate into world frame
        q = load_quaternion(full0)
        rot = R.from_quat(q)    # note: [qx,qy,qz,qw]
        world = rot.apply(pts)  # [N,3]

        # plot
        plt.figure(figsize=(6, 6))
        plt.scatter(world[:, 0], world[:, 1], s=10)
        plt.scatter(world[beam_idx, 0], world[beam_idx, 1])
        plt.axis('equal')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title(f"LiDAR scan: {df.loc[0, 'filename']}")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    plot_scan_points(500)
    plot_scan_points_all()

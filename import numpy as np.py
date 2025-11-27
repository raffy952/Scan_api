import numpy as np
import matplotlib.pyplot as plt

import scansegmentapi.compact as CompactApi
from scansegmentapi.udp_handler import UDPHandler

UDP_IP = "172.16.35.58"
UDP_PORT = 2122

transport = UDPHandler(UDP_IP, UDP_PORT, 65535)
receiver = CompactApi.Receiver(transport)

plt.ion()
fig, ax = plt.subplots(figsize=(7,7))

def plot_frame(r_all, a_all):
    if len(r_all) == 0:
        return

    r = np.concatenate(r_all)
    a = np.concatenate(a_all)

    mask = (r > 0) & np.isfinite(r)
    r = r[mask]
    a = a[mask]

    r_m = r / 1000.0
    a_rad = np.deg2rad(a)

    x = r_m * np.cos(a_rad)
    y = r_m * np.sin(a_rad)

    ax.cla()
    ax.scatter(x, y, s=2)
    ax.plot(0, 0, 'ro')
    ax.set_aspect('equal')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Scan 2D ricostruita")
    plt.pause(0.01)

current_frame = None
accum_r = []
accum_a = []

try:
    while True:
        segments, frame_numbers, segment_counters = receiver.receive_segments(1)

        for i, seg in enumerate(segments):
            frame_n = frame_numbers[i]

            if current_frame is None:
                current_frame = frame_n

            if frame_n != current_frame:
                plot_frame(accum_r, accum_a)
                accum_r = []
                accum_a = []
                current_frame = frame_n

            segdata = seg["Modules"][0]["SegmentData"][0]

            distance_list = segdata["Distance"]
            theta_list = segdata["ChannelTheta"]

            num_channels = len(distance_list)

            for ch in range(num_channels):
                pts = np.array(distance_list[ch]).reshape(-1)
                theta = float(theta_list[ch])  # singolo valore
                angles = np.full_like(pts, theta, dtype=float)

                accum_r.append(pts)
                accum_a.append(angles)

except KeyboardInterrupt:
    print("Interrotto")

finally:
    receiver.close_connection()
    plt.ioff()
    print("Connessione chiusa.")


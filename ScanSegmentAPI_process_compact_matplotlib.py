#
# Copyright (c) 2023-2024 SICK AG
# SPDX-License-Identifier: MIT
#
# This program receives scan segments in Compact format.
# The values are converted to Cartesian coordinates and plotted.
# The standard deviation of the x values is calculated and displayed in the histogram.

# make sure to install the required packages (add them to pyproject.toml)

import matplotlib.pyplot as plt
import numpy as np
import scansegmentapi.compact as CompactApi
from scansegmentapi.compact_stream_extractor import CompactStreamExtractor
from scansegmentapi.udp_handler import UDPHandler

# Port used for data streaming. Enter the port configured in your device.
PORT = 2122

# IP of the receiver.
IP = "172.16.35.58"

if __name__ == "__main__":
    transportLayer = UDPHandler(IP, PORT, 65535)
    receiver = CompactApi.Receiver(transportLayer)
    # receive 1 segment, adjust picoScan scan range filter if necessary
    (segments, frameNumbers, segmentCounters) = receiver.receive_segments(1) 
    receiver.close_connection()

    # Store distance values (radial) and theta values
    r=segments[0]["Modules"][0]["SegmentData"][0]["Distance"][0]
    theta=segments[0]["Modules"][0]["SegmentData"][0]["ChannelTheta"]

    # Convert to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    print(f'Distances : {r.shape}')

    # Remove zero values only for plotting
    non_zero_indices = r != 0
    x_non_zero = x[non_zero_indices]
    y_non_zero = y[non_zero_indices]

    # Calculate the mean value and the standard deviation of x values
    mean_x = np.mean(x_non_zero)
    std_dev_x = np.std(x_non_zero)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot the Cartesian coordinates on the first subplot as a line graph with dots
    ax1.plot(x_non_zero, y_non_zero, marker='o', linestyle='-')
    ax1.set_xlabel('Distances [mm]')
    ax1.set_ylabel('Distances [mm]')
    ax1.set_title('Cartesian plot')
    ax1.xaxis.set_major_locator(plt.MultipleLocator(1))  # Set major ticks to every 1 mm

    # Plot a horizontal line at the mean value of x
    ax1.axvline(mean_x, color='r', linestyle='--', linewidth=1, label='Mean distance values [mm]: {:.2f}'.format(mean_x))
    ax1.legend()

    # Plot the histogram of x values on the second subplot
    ax2.hist(x_non_zero, bins=100, edgecolor='black')
    ax2.set_xlabel('Distances [mm]')
    ax2.set_title('Histogram of distance values')
    ax2.xaxis.set_major_locator(plt.MultipleLocator(1))  # Set major ticks to every 1 mm

    # Add the standard deviation value as text in the top left of the histogram
    ax2.text(0.05, 0.95, 'Std Dev [mm]: {:.2f}'.format(std_dev_x), transform=ax2.transAxes, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))

    # Adjust layout to prevent overlap and display the plots
    plt.tight_layout() 
    plt.show()
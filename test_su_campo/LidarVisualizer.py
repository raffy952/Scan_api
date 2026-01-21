import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

class LidarVisualizer:
    def __init__(self, ax):
        self.ax = ax

    def plot_frame(self, all_points, classified_clusters, valid_ids, prediction):
        # cla() pulisce i dati ma è più veloce di clear() per cicli continui
        self.ax.cla()
        
        # Sfondo (punti grigi)
        self.ax.scatter(all_points[:, 0], -all_points[:, 1], color='lightgray', s=1, alpha=0.3)
        
        # Cluster validi
        colors = {1: 'blue', 2: 'green'}
        for lab_id in valid_ids:
            if lab_id in classified_clusters:
                cluster = classified_clusters[lab_id]
                cls = cluster['classification']
                self.ax.scatter(cluster['points'][:, 0], -cluster['points'][:, 1], 
                               s=8, color=colors.get(cls, 'orange'), label=f"Cls {cls}")

        # Pose Estimation
        if prediction is not None:
            self.ax.scatter(prediction[0, 0], -prediction[0, 1], color='red', marker='X', s=80)
            
            legend_elements = [
                Line2D([0], [0], linestyle='None', label=f"X: {prediction[0,0]:.2f}"),
                Line2D([0], [0], linestyle='None', label=f"Y: {-prediction[0,1]:.2f}"),
                Line2D([0], [0], linestyle='None', label=f"Theta: {prediction[0,2]:.2f}")
            ]
            self.ax.legend(handles=legend_elements, loc="upper right", fontsize='small')

        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title("LiDAR Scan & Tracking")
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_xlim(-3, 7)
        self.ax.set_ylim(-3, 7)
        plt.pause(0.001)

from urllib.parse import ParseResultBytes
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import scansegmentapi.compact as CompactApi
from scansegmentapi.udp_handler import UDPHandler
import keyboard
import math
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
from feature_extractor import calculate_cluster_features
import warnings
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ===============================
# CONFIGURAZIONE
# ===============================
X_lim = 3.0  # metri
y_lim = 0.2 # metri
delta_x = 0.05 
delta_y = 0.05


# Carica modello
model_path = 'best_params_180.pkl'
model = joblib.load(model_path)
scaler = joblib.load('scaler_y.pkl')
model_reg = load_model('my_model_10000.h5')
labels_to_keep = [1, 2]
dist_path = 'scans_reali/distances3.npy'
angle_path = 'scans_reali/thetas3.npy'

# Ordine delle feature utilizzate dal modello
FEATURE_ORDER = [
    'num_points',
    'pca_length',
    'pca_width',
    'convex_area',
    'convex_perimeter',
    'aspect_ratio',
    'mean_curvature'
]



def pad_center(arr, X):
    n = len(arr)
    if n >= X:
        return arr[:X]   # opzionale: tronca se troppo lungo

    total_pad = X - n
    left = total_pad // 2
    right = total_pad - left   # garantisce correttezza anche per X dispari

    return np.pad(arr, (left, right), mode='constant', constant_values=60.0)

def check_regression_boundaries(centroid_1, centroid_2):

    xm = (centroid_1[0] + centroid_2[0]) / 2
    ym = (centroid_1[1] + centroid_2[1]) / 2

    if (abs(xm) <= (X_lim + delta_x)) and (abs(ym) <= (y_lim + delta_y)):
        return True
    else:
        return False


    


# ===============================
# LOOP PRINCIPALE
# ===============================
def main():
    scans = np.load(dist_path, allow_pickle=True)
    #print(scans)
    angles = np.load(angle_path, allow_pickle=True)
    print(np.rad2deg(angles))
    for i in range(len(scans)):
        
        scan = scans[i]
        angle = angles[i]
        # Conversione in coordinate cartesiane
        x = scan * np.cos(angle + np.pi/2) / 1000
        y = scan * np.sin(angle + np.pi/2) / 1000
        points = np.column_stack((x.flatten(), y.flatten()))
        # Clustering con DBSCAN
        db = DBSCAN(eps=0.25, min_samples=4).fit(points)
        labels = db.labels_
        unique_labels = set(labels)
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, c='lightgray', s=5)
        for label in unique_labels:
            if label == -1:
                continue  # Rumore
            cluster_points = points[labels == label]
            features = calculate_cluster_features(cluster_points)
            feature_vector = [features[feat] for feat in FEATURE_ORDER]
            #feature_vector_scaled = scaler.transform([feature_vector])
            predicted_label = model.predict([feature_vector])[0]
            if predicted_label in labels_to_keep:
                centroid = np.mean(cluster_points, axis=0)
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=20)
                plt.text(centroid[0], centroid[1], f'ID: {predicted_label}', fontsize=9, ha='center')
        plt.xlim(-X_lim - 0.5, X_lim + 0.5)
        plt.ylim(-y_lim - 0.5, y_lim + 0.5)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Lidar Scan with DBSCAN Clustering and Classification')
        plt.grid()
        plt.show()

    

if __name__ == "__main__":
    main()

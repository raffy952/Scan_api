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
import time

LIDAR_TYPE = 'sick'  # 'sick' or 'r2000'
IP = '172.16.35.58'
PORT = 2122
EPS = 0.25  # DBSCAN eps parameter
MIN_SAMPLES = 4  # DBSCAN min_samples parameter
CLASSIFICATION_MODEL = joblib.load('best_params_180.pkl')
SCALER = joblib.load('scaler_y.pkl')
REGRESSION_MODEL = load_model('my_model_10000.h5')
LABELS_TO_KEEP = [1, 2]
WIDTH_RECTANGLE = 1.69  # meters
HEIGHT_RECTANGLE = 1.80  # meters
TOLERANCE = 0.1  # meters


FEATURE_ORDER = [
    'num_points',
    'pca_length',
    'pca_width',
    'convex_area',
    'convex_perimeter',
    'aspect_ratio',
    'mean_curvature'
]

class CaptureLidarData:
    def __init__(self, ip="172.16.35.58", port=2122, lidar_type='sick'):
        self.ip = ip
        self.port = port
        self.lidar_type = lidar_type

    def lidar_protocol(self):
        if self.lidar_type == 'sick':
            self.transport = UDPHandler(self.ip, self.port, 65535)
            self.receive = CompactApi.Receiver(self.transport)
        elif self.lidar_type == 'rplidar':
            pass
            #properties = CompactApi.RPLidarProperties()
        else:
            raise ValueError("Unsupported LIDAR type")
    
    def get_scan(self):
        if self.lidar_type == 'sick':
            segment, frame_number, segment_counter = self.receive.receive_segment(1)
            return segment, frame_number, segment_counter
            
        elif self.lidar_type == 'rplidar':
            pass
        else:
            raise ValueError("Unsupported LIDAR type")



class LidarScanProcessor:
    def __init__(self):
        self.dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
        self.classification_model = CLASSIFICATION_MODEL
        self.regression_model = REGRESSION_MODEL
        self.scaler = SCALER

    
    def get_clusters(self, x, y):
        points = np.column_stack((x, y))
        labels = self.dbscan.fit_predict(points)
        return labels

    def classify_clusters(self, x, y, labels):
        unique_labels = set(labels)
        classified_clusters = {}
        pts = np.column_stack((x.flatten(), y.flatten()))

        for lab in unique_labels:
            cluster_pts = pts[labels == lab]
            if lab == -1:
                continue  # Skip noise

            # Cluster troppo piccoli
            if len(cluster_pts) < 3 or np.all(cluster_pts == 0.0) or np.all(cluster_pts < 0.05):
                print(f"Cluster {lab}: troppo piccolo ({len(cluster_pts)} punti). Saltato.")
                continue

            # FEATURE EXTRACTION
            # ===============================
            features = calculate_cluster_features(cluster_pts)
            feature_vector = [features[f] for f in FEATURE_ORDER]
            
            # ===============================
            # CLASSIFICAZIONE
            # ===============================
            classification = self.classification_model.predict([feature_vector])[0]
            prob = self.classification_model.predict_proba([feature_vector])[0][np.argmax(self.classification_model.predict_proba([feature_vector]))]
            classified_clusters[lab] = {
                'points': cluster_pts,
                'centroid': np.mean(cluster_pts, axis=0),
                'classification': classification,
                'probability': prob
            }

        return classified_clusters

    def filter_label2_pairs(self, clusters_info, max_dist, tolerance):

        label2_clusters = [lab for lab, info in clusters_info.items() 
                    if info['classification'] == 2]

        if len(label2_clusters) < 2:
            return 

        # Trova la coppia che rispetta il limite di distanza
        valid_pair = None
        for i, lab1 in enumerate(label2_clusters):
            for lab2 in label2_clusters[i+1:]:
                centroid1 = clusters_info[lab1]['centroid']
                centroid2 = clusters_info[lab2]['centroid']
                dist = np.linalg.norm(centroid1 - centroid2)
                if ((dist <= max_dist + tolerance) and dist >= (max_dist - tolerance)):
                    valid_pair = (lab1, lab2)
                    break
            if valid_pair:
                break

        # Crea lista dei cluster da mantenere
        clusters_to_keep = []
        for lab in clusters_info.keys():
            if clusters_info[lab]['classification'] != 2:
                # Mantieni tutti i cluster che non sono label 2
                clusters_to_keep.append(lab)
            elif valid_pair and lab in valid_pair:
                # Mantieni solo i cluster della coppia valida
                clusters_to_keep.append(lab)

        return clusters_to_keep
    
    
    
    def pad_center(self, arr, X):
        n = len(arr)
        if n >= X:
            return arr[:X]

        total_pad = X - n
        left = total_pad // 2
        right = total_pad - left

        return np.pad(arr, (left, right), mode='constant', constant_values=60.0) 
    
    def pose_estimation(self, filtered_distances):
        filtered_distances_padded = self.pad_center(filtered_distances, 1105).reshape(1,-1)
                    
        prediction_scaled = self.regression_model.predict(filtered_distances_padded / (60.0))
        prediction = self.scaler.inverse_transform(prediction_scaled) 
        
        return prediction
        

    

class LidarVisualizer:

    def plot_settings(self,ax):
        ax.clear()
        ax.set_title(f'Scan con cluster e classificazione')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.axis('equal')
        ax.grid()

    def plot_points(self, ax, points, prediction):
        ax.scatter(points[:,0], -points[:,1], color='gray', s=5, alpha=0.5)
        ax.scatter(prediction[0,0], -prediction[0,1], color='red', alpha=1.0, marker='x')
                    
        legend1 = ax.legend(loc="upper right", fontsize=8)
        ax.add_artist(legend1)

        legend_elements = [
            Line2D([0], [0], linestyle='None',
                    label=f"x: {prediction[0,0]:.3f}"),
            Line2D([0], [0], linestyle='None',
                    label=f"y: {-prediction[0,1]:.3f}"),
            Line2D([0], [0], linestyle='None',
                    label=f"theta: {prediction[0,2]:.3f}")
        ]

        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

        plt.draw()
        plt.pause(0.05)  # Aggiorna la visualizzazione

        

    

def main():
    segments = []
    frame_numbers = []
    segment_counters = []

    lidar_capture = CaptureLidarData(ip=IP, port=PORT, lidar_type=LIDAR_TYPE)
    lidar_capture.lidar_protocol()
    lidar_processor = LidarScanProcessor()
    lidar_visualizer = LidarVisualizer()

    # Setup figura per visualizzazione continua
    plt.ion()  # Modalità interattiva
    fig, ax = plt.subplots(figsize=(8, 8))

    while True:
        segment, frame_number, segment_counter = lidar_capture.get_scan()
        segments.append(segment)
        frame_numbers.append(frame_number)
        segment_counters.append(segment_counter)

        # Controlla se abbiamo 7 segmenti dello stesso frame in ordine
        if len(frame_numbers) == 7 and len(segment_counters) == 7:
            if np.all(np.array(frame_numbers) == frame_numbers[0]):
                if np.all(np.array(segment_counters[:-1]) < np.array(segment_counters[1:])):
                    distances = np.vstack([
                        seg["Modules"][0]["SegmentData"][0]["Distance"][0] 
                        for seg in segments
                    ]).reshape(1, -1)

                    thetas = np.vstack([
                        seg["Modules"][0]["SegmentData"][0]["ChannelTheta"] 
                        for seg in segments
                    ]).reshape(1, -1)

                    mask = (thetas > (-np.pi-0.004)/2) & (thetas < (np.pi+0.004)/2)

                    thetas = thetas[mask]
                    distances = distances[mask] / 1000.0 # Converti in metri
                    
                    x = distances * np.cos(thetas) 
                    y = distances * np.sin(thetas) 

                    labels = lidar_processor.get_clusters(x, y)
                    classified_clusters = lidar_processor.classify_clusters(x, y, labels)

                    label_2_filtered = lidar_processor.filter_label2_pairs(classified_clusters, labels, WIDTH_RECTANGLE, TOLERANCE)
                    filtered_distances = np.where(np.isin(label_2_filtered, LABELS_TO_KEEP), distances, 0)

                    prediction = lidar_processor.pose_estimation(filtered_distances)

                    lidar_visualizer.plot_settings(ax, classified_clusters['points'], prediction)


                    frame_numbers = []
                    segment_counters = []
                    segments = []

            else:
                mask = np.array(frame_numbers) != frame_numbers[0]
                frame_numbers = list(np.array(frame_numbers)[mask])
                segment_counters = list(np.array(segment_counters)[mask])
                segments = list(np.array(segments)[mask])


    plt.ioff()
    plt.close()
    receiver.close_connection()
                    
            
                

if __name__ == "__main__":
    main()
                    
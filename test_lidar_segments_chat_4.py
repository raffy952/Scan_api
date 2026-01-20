import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import scansegmentapi.compact as CompactApi
from scansegmentapi.udp_handler import UDPHandler
import joblib
import os
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

from tensorflow.keras.models import load_model
from feature_extractor import calculate_cluster_features
import warnings
from matplotlib.lines import Line2D
import yaml
warnings.filterwarnings("ignore")


#warnings.filterwarnings("ignore", message="X does not have valid feature names")

# --- CONFIGURATION ---
def load_config(config_path='config.yml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Caricamento della configurazione
config = load_config()

# --- ASSEGNAZIONE PARAMETRI DA CONFIG ---
LIDAR_TYPE = config['lidar']['type']
IP = config['lidar']['ip']
PORT = config['lidar']['port']

EPS = config['dbscan']['eps']
MIN_SAMPLES = config['dbscan']['min_samples']

# Caricamento modelli usando i percorsi nel YAML
CLASSIFICATION_MODEL = joblib.load(config['models']['classification'])
SCALER = joblib.load(config['models']['scaler'])
REGRESSION_MODEL = load_model(config['models']['regression'])

WIDTH_RECTANGLE = config['dimensions']['width_rectangle']
WIDTH_PALLET_COMPLETE1 = config['dimensions']['width_pallet_1']
WIDTH_PALLET_COMPLETE2 = config['dimensions']['width_pallet_2']
TOLERANCE = config['dimensions']['tolerance']
FEATURE_ORDER = config['feature_order']['parameters']

class CaptureLidarData:
    def __init__(self, ip="172.16.35.58", port=2122, lidar_type='sick'):
        self.ip = ip
        self.port = port
        self.lidar_type = lidar_type
        self.receive = None

    def lidar_protocol(self):
        if self.lidar_type == 'sick':
            self.transport = UDPHandler(self.ip, self.port, 65535)
            self.receive = CompactApi.Receiver(self.transport)
        else:
            raise ValueError("Unsupported LIDAR type")

    def get_scan(self):
        return self.receive.receive_segments(1)

class LidarScanProcessor:
    def __init__(self):
        self.dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)

    def get_clusters(self, x, y):
        points = np.column_stack((x, y))
        return self.dbscan.fit_predict(points)

    def classify_clusters(self, x, y, labels):
        unique_labels = set(labels)
        classified_clusters = {}
        pts = np.column_stack((x, y))

        for lab in unique_labels:
            if lab == -1: continue
            cluster_pts = pts[labels == lab]
            if len(cluster_pts) < 3: continue

            features = calculate_cluster_features(cluster_pts)
            feature_vector = [features[f] for f in FEATURE_ORDER]
            
            classification = CLASSIFICATION_MODEL.predict([feature_vector])[0]
            probs = CLASSIFICATION_MODEL.predict_proba([feature_vector])[0]
            
            classified_clusters[lab] = {
                'points': cluster_pts,
                'centroid': np.mean(cluster_pts, axis=0),
                'classification': classification,
                'probability': np.max(probs)
            }
        return classified_clusters

    def filter_logic(self, clusters_info):
        label1_clusters = [lab for lab, info in clusters_info.items() if info['classification'] == 1]
        valid_l1 = None
        for lab in label1_clusters:
            width = np.ptp(clusters_info[lab]['points'][:, 0])
            if abs(width - WIDTH_PALLET_COMPLETE1) <= TOLERANCE or abs(width - WIDTH_PALLET_COMPLETE2) <= TOLERANCE:
                valid_l1 = lab
                break
        
        label2_clusters = [lab for lab, info in clusters_info.items() if info['classification'] == 2]
        valid_l2_pair = []
        for i, lab1 in enumerate(label2_clusters):
            for lab2 in label2_clusters[i+1:]:
                dist = np.linalg.norm(clusters_info[lab1]['centroid'] - clusters_info[lab2]['centroid'])
                if abs(dist - WIDTH_RECTANGLE) <= TOLERANCE:
                    valid_l2_pair = [lab1, lab2]
                    break
            if valid_l2_pair: break

        final_valid_ids = []
        if valid_l1 is not None: final_valid_ids.append(valid_l1)
        final_valid_ids.extend(valid_l2_pair)
        return final_valid_ids

    def pose_estimation(self, distances, labels, valid_ids):
        mask = np.isin(labels, valid_ids)
        filtered_dist_input = np.where(mask, distances, 60.0)
        
        n = len(filtered_dist_input)
        if n >= 1105:
            final_input = filtered_dist_input[:1105]
        else:
            pad_left = (1105 - n) // 2
            pad_right = 1105 - n - pad_left
            final_input = np.pad(filtered_dist_input, (pad_left, pad_right), constant_values=60.0)
            
        prediction_scaled = REGRESSION_MODEL.predict(final_input.reshape(1, -1) / 60.0, verbose=0)
        return SCALER.inverse_transform(prediction_scaled)

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
        plt.pause(0.001)

def main():
    lidar_capture = CaptureLidarData(ip=IP, port=PORT, lidar_type=LIDAR_TYPE)
    lidar_capture.lidar_protocol()
    processor = LidarScanProcessor()
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    visualizer = LidarVisualizer(ax)

    segments, frame_numbers, segment_counters = [], [], []

    try:
        while True:
            res = lidar_capture.get_scan()
            if not res: continue
            
            seg_data, f_num, s_cnt = res
            segments.append(seg_data[0])
            frame_numbers.append(f_num[0])
            segment_counters.append(s_cnt[0])

            # Prevenzione accumulo memoria
            if len(frame_numbers) > 14:
                segments.pop(0); frame_numbers.pop(0); segment_counters.pop(0)

            if len(frame_numbers) == 7:
                if all(fn == frame_numbers[0] for fn in frame_numbers):
                    # Ricostruzione dati
                    raw_dist = np.concatenate([s["Modules"][0]["SegmentData"][0]["Distance"][0] for s in segments])
                    raw_thetas = np.concatenate([s["Modules"][0]["SegmentData"][0]["ChannelTheta"] for s in segments])
                    
                    mask_fov = (raw_thetas > -np.pi/2) & (raw_thetas < np.pi/2)
                    thetas = raw_thetas[mask_fov]
                    distances = raw_dist[mask_fov] / 1000.0
                    
                    x, y = distances * np.cos(thetas), distances * np.sin(thetas)
                    all_points = np.column_stack((x, y))

                    labels = processor.get_clusters(x, y)
                    clusters_info = processor.classify_clusters(x, y, labels)
                    valid_ids = processor.filter_logic(clusters_info)
                    
                    prediction = None
                    if valid_ids:
                        try:
                            prediction = processor.pose_estimation(distances, labels, valid_ids)
                        except: pass
                    
                    visualizer.plot_frame(all_points, clusters_info, valid_ids, prediction)
                    
                    # Reset buffer
                    segments, frame_numbers, segment_counters = [], [], []
                else:
                    segments.pop(0); frame_numbers.pop(0); segment_counters.pop(0)

    except KeyboardInterrupt:
        print("Uscita...")
    finally:
        plt.ioff()
        plt.close('all')

if __name__ == "__main__":
    main()
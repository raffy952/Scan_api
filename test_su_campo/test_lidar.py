import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

from tensorflow.keras.models import load_model
import warnings
import yaml

from CaptureLidarData import CaptureLidarData
from LidarScanProcessor import LidarScanProcessor
from LidarVisualizer import LidarVisualizer

warnings.filterwarnings("ignore")


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



def main():
    lidar_capture = CaptureLidarData(ip=IP, port=PORT, lidar_type=LIDAR_TYPE)
    lidar_capture.lidar_protocol()
    processor = LidarScanProcessor(EPS, MIN_SAMPLES, CLASSIFICATION_MODEL, SCALER, REGRESSION_MODEL, FEATURE_ORDER,
                                   WIDTH_RECTANGLE, WIDTH_PALLET_COMPLETE1, WIDTH_PALLET_COMPLETE2, TOLERANCE)

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
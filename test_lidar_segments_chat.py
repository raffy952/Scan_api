from urllib.parse import ParseResultBytes
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import scansegmentapi.compact as CompactApi
from scansegmentapi.udp_handler import UDPHandler
import keyboard
import math
import joblib
from tensorflow.keras.models import load_model
from feature_extractor import calculate_cluster_features
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ===============================
# CONFIGURAZIONE
# ===============================
UDP_IP = "172.16.35.58"
UDP_PORT = 2122

transport = UDPHandler(UDP_IP, UDP_PORT, 65535)
receiver = CompactApi.Receiver(transport)

# Carica modello
model_path = 'best_params_180.pkl'
model = joblib.load(model_path)
scaler = joblib.load('scaler_y.pkl')
model_reg = load_model('model_final_20000.h5')
labels_to_keep = [1, 2]

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

segments = []
frame_numbers = []
segment_counters = []


def pad_center(arr, X):
    n = len(arr)
    if n >= X:
        return arr[:X]   # opzionale: tronca se troppo lungo

    total_pad = X - n
    left = total_pad // 2
    right = total_pad - left   # garantisce correttezza anche per X dispari

    return np.pad(arr, (left, right), mode='constant', constant_values=0)


# ===============================
# LOOP PRINCIPALE
# ===============================
while True:
    
    segment, frame_number, segment_counter = receiver.receive_segments(1)

    segments.append(segment[0])
    frame_numbers.append(frame_number[0])
    segment_counters.append(segment_counter[0])
    # Controlla se abbiamo 10 segmenti dello stesso frame in ordine
    if len(frame_numbers) == 6 and len(segment_counters) == 6:
        if np.all(np.array(frame_numbers) == frame_numbers[0]):
            if np.all(np.array(segment_counters[:-1]) < np.array(segment_counters[1:])):

                print("Ricevuti 10 segmenti di uno stesso frame in ordine corretto")

                # ===============================
                # RICOSTRUZIONE SCAN
                # ===============================
                distances = np.vstack([
                    seg["Modules"][0]["SegmentData"][0]["Distance"][0] 
                    for seg in segments
                ]).reshape(1, -1)

                thetas = np.vstack([
                    seg["Modules"][0]["SegmentData"][0]["ChannelTheta"] 
                    for seg in segments
                ]).reshape(1, -1)

                x = distances * np.cos(thetas + math.pi/2) / 1000.0
                y = distances * np.sin(thetas + math.pi/2) / 1000.0
                
                pts = np.column_stack((x.flatten(), y.flatten()))
                print(f"Numero punti scan ricostruito: {pts.shape[0]}")

                # ===============================
                # DBSCAN
                # ===============================
                dbscan = DBSCAN(eps=0.25, min_samples=4)
                labels = dbscan.fit_predict(pts)
                unique_labels = set(labels)

                # ===============================
                # VISUALIZZAZIONE
                # ===============================
                plt.figure(figsize=(8, 8))
                plt.title(f'Scan con cluster e classificazione - Frame {frame_numbers[0]}')
                plt.xlabel('X [m]')
                plt.ylabel('Y [m]')
                plt.axis('equal')
                plt.grid()

                colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

                for lab, col in zip(unique_labels, colors):
                    cluster_pts = pts[labels == lab]

                    # Noise
                    if lab == -1:
                        col = (0, 0, 0, 1)
                        plt.scatter(cluster_pts[:, 0], cluster_pts[:, 1], s=6, color=col, label="Noise")
                        continue

                    # Cluster troppo piccoli → salto PCA e ConvexHull
                    if len(cluster_pts) < 3 or np.all(cluster_pts == 0.0) or np.all(cluster_pts < 0.05):
                        print(f"Cluster {lab}: troppo piccolo ({len(cluster_pts)} punti). Saltato.")
                        continue

                    # ===============================
                    # FEATURE EXTRACTION
                    # ===============================
                    features = calculate_cluster_features(cluster_pts)

                    # Costruisci vettore feature nell’ordine corretto
                    feature_vector = [features[f] for f in FEATURE_ORDER]

                    # ===============================
                    # CLASSIFICAZIONE
                    # ===============================
                    classification = model.predict([feature_vector])[0]
                    prob = model.predict_proba([feature_vector])[0][np.argmax(model.predict_proba([feature_vector]))]

                    #print(f"Cluster {lab} → Class: {classification} | Feature vector: {feature_vector}")

                    

                    # ===============================
                    # DISEGNA CLUSTER
                    # ===============================
                    plt.scatter(cluster_pts[:, 0], cluster_pts[:, 1], s=8, color=col, label=f"{classification} {prob:.3f}")
                
                filtered_points = np.where(~np.isin(labels, labels_to_keep)[:, None], pts, 0)
                filtered_distances = np.linalg.norm(filtered_points, axis=1)
                filtered_distances_padded = pad_center(filtered_distances, 1105).reshape(1,-1)
                prediction = model_reg.predict(filtered_distances_padded / 60)
                prediction = scaler.inverse_transform(prediction)   
                plt.scatter(prediction[0,0], prediction[0,1], color='red', alpha=1.0, marker='x')
                plt.scatter(filtered_points[:, 1], filtered_points[:, 0], s=2, color='gray', alpha=0.3, label="Filtered points")

                plt.legend(loc="upper right", fontsize=8)
                plt.show()

                # RESET
                frame_numbers = []
                segment_counters = []
                segments = []


        else:
            mask = np.array(frame_numbers) != frame_numbers[0]
            frame_numbers = list(np.array(frame_numbers)[mask])
            segment_counters = list(np.array(segment_counters)[mask])
            segments = list(np.array(segments)[mask])

    if keyboard.is_pressed('q'):
        break

receiver.close_connection()


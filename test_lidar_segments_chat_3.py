
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
UDP_IP = "172.16.35.58"
UDP_PORT = 2122
X_lim = 3.0  # metri
y_lim = 0.2 # metri
delta_x = 0.05 
delta_y = 0.05

transport = UDPHandler(UDP_IP, UDP_PORT, 65535)
receiver = CompactApi.Receiver(transport)

# Carica modello
model_path = 'best_params_180.pkl'
model = joblib.load(model_path)
scaler = joblib.load('scaler_y.pkl')
model_reg = load_model('my_model_10000.h5')
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

def filter_label2_pairs(clusters_info, labels, max_dist, tolerance):
#     """
#     Filtra le coppie di elementi con label 2 in base alla distanza tra centroidi.
#     Restituisce gli indici dei cluster da mantenere.
#     """
    label2_clusters = [lab for lab, info in clusters_info.items() 
                    if info['classification'] == 2]

    if len(label2_clusters) < 2:
        # Se c'è 0 o 1 elemento con label 2, mantieni tutti
        return list(clusters_info.keys())

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

def pad_center(arr, X):
    n = len(arr)
    if n >= X:
        return arr[:X]

    total_pad = X - n
    left = total_pad // 2
    right = total_pad - left

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
    segments = []
    frame_numbers = []
    segment_counters = []
    clusters_info = {}

    # Setup figura per visualizzazione continua
    plt.ion()  # Modalità interattiva
    fig, ax = plt.subplots(figsize=(8, 8))

    while True:
        segment, frame_number, segment_counter = receiver.receive_segments(1)

        segments.append(segment[0])
        frame_numbers.append(frame_number[0])
        segment_counters.append(segment_counter[0])
        
        # Controlla se abbiamo 7 segmenti dello stesso frame in ordine
        if len(frame_numbers) == 7 and len(segment_counters) == 7:
            if np.all(np.array(frame_numbers) == frame_numbers[0]):
                if np.all(np.array(segment_counters[:-1]) < np.array(segment_counters[1:])):

                    print("Ricevuti 7 segmenti di uno stesso frame in ordine corretto")

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
                    
                    mask = (thetas > (-np.pi-0.004)/2) & (thetas < (np.pi+0.004)/2)

                    thetas = thetas[mask]
                    distances = distances[mask]
                    
                    x = distances * np.cos(thetas) / 1000.0
                    y = distances * np.sin(thetas) / 1000.0
                    
                    pts = np.column_stack((x.flatten(), y.flatten()))
                    print(f"Numero punti scan ricostruito: {pts.shape[0]}")

                    # ===============================
                    # DBSCAN
                    # ===============================
                    dbscan = DBSCAN(eps=0.25, min_samples=4)
                    labels = dbscan.fit_predict(pts)
                    unique_labels = set(labels)

                    # ===============================
                    # VISUALIZZAZIONE CONTINUA
                    # ===============================
                    ax.clear()
                    ax.set_title(f'Scan con cluster e classificazione - Frame {frame_numbers[0]}')
                    ax.set_xlabel('X [m]')
                    ax.set_ylabel('Y [m]')
                    ax.axis('equal')
                    ax.grid()

                    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

                    for lab, col in zip(unique_labels, colors):
                        cluster_pts = pts[labels == lab]

                        # Noise
                        if lab == -1:
                            col = (0, 0, 0, 1)
                            ax.scatter(cluster_pts[:, 0], cluster_pts[:, 1], s=6, color=col, label="Noise")
                            continue

                        # Cluster troppo piccoli
                        if len(cluster_pts) < 3 or np.all(cluster_pts == 0.0) or np.all(cluster_pts < 0.05):
                            print(f"Cluster {lab}: troppo piccolo ({len(cluster_pts)} punti). Saltato.")
                            continue

                        # ===============================
                        # FEATURE EXTRACTION
                        # ===============================
                        features = calculate_cluster_features(cluster_pts)
                        feature_vector = [features[f] for f in FEATURE_ORDER]

                        # ===============================
                        # CLASSIFICAZIONE
                        # ===============================
                        classification = model.predict([feature_vector])[0]
                        prob = model.predict_proba([feature_vector])[0][np.argmax(model.predict_proba([feature_vector]))]
                        clusters_info[lab] = { 
                            'points' : cluster_pts,
                            'centroid' : np.mean(cluster_pts, axis=0),
                            'classification' : classification,
                            'probability' : prob
                        }

                        # ===============================
                        # DISEGNA CLUSTER
                        # ===============================
                        ax.scatter(cluster_pts[:, 0], cluster_pts[:, 1], s=8, color=col, label=f"{classification} {prob:.3f}")
                    
                    # Predizione regressione
                    filtered_distances = np.where(np.isin(labels, filter_label2_pairs(clusters_info, labels, 1.69, 0.1)), distances, 0.0) / 1000.0
                    #filtered_distances = np.where(np.isin(labels, labels_to_keep), distances, 0.0) / 1000.0
                    
                    filtered_distances = np.where(filtered_distances == 0.0, 60.0, filtered_distances)
                    filtered_distances_padded = pad_center(filtered_distances, 1105).reshape(1,-1)
                    
                    prediction_scaled = model_reg.predict(filtered_distances_padded / (60.0))
                    prediction = scaler.inverse_transform(prediction_scaled)   
                    print(prediction)
                    
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

                    # RESET
                    frame_numbers = []
                    segment_counters = []
                    segments = []
                    clusters_info = {}

            else:
                mask = np.array(frame_numbers) != frame_numbers[0]
                frame_numbers = list(np.array(frame_numbers)[mask])
                segment_counters = list(np.array(segment_counters)[mask])
                segments = list(np.array(segments)[mask])

        if keyboard.is_pressed('q'):
            break

    plt.ioff()
    plt.close()
    receiver.close_connection()

if __name__ == "__main__":
    main()



# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
# import scansegmentapi.compact as CompactApi
# from scansegmentapi.udp_handler import UDPHandler
# import keyboard
# import math
# import joblib
# import pandas as pd
# from tensorflow.keras.models import load_model
# from feature_extractor import calculate_cluster_features
# import warnings
# from matplotlib.lines import Line2D
# warnings.filterwarnings("ignore", message="X does not have valid feature names")

# # ===============================
# # CONFIGURAZIONE
# # ===============================
# UDP_IP = "172.16.35.58"
# UDP_PORT = 2122
# X_lim = 3.0  # metri
# y_lim = 0.2 # metri
# delta_x = 0.05 
# delta_y = 0.05

# # Distanza massima tra centroidi per coppie label 2
# MAX_CENTROID_DISTANCE = 1.69  # metri (modifica questo valore secondo necessità)
# DISTANCE_TOLERANCE = 0.1  # tolleranza in metri

# transport = UDPHandler(UDP_IP, UDP_PORT, 65535)
# receiver = CompactApi.Receiver(transport)

# # Carica modello
# model_path = 'best_params_180.pkl'
# model = joblib.load(model_path)
# scaler = joblib.load('scaler_y.pkl')
# model_reg = load_model('my_model_10000.h5')
# labels_to_keep = [1, 2]

# # Ordine delle feature utilizzate dal modello
# FEATURE_ORDER = [
#     'num_points',
#     'pca_length',
#     'pca_width',
#     'convex_area',
#     'convex_perimeter',
#     'aspect_ratio',
#     'mean_curvature'
# ]

# def pad_center(arr, X):
#     n = len(arr)
#     if n >= X:
#         return arr[:X]

#     total_pad = X - n
#     left = total_pad // 2
#     right = total_pad - left

#     return np.pad(arr, (left, right), mode='constant', constant_values=60.0)

# def check_regression_boundaries(centroid_1, centroid_2):
#     xm = (centroid_1[0] + centroid_2[0]) / 2
#     ym = (centroid_1[1] + centroid_2[1]) / 2

#     if (abs(xm) <= (X_lim + delta_x)) and (abs(ym) <= (y_lim + delta_y)):
#         return True
#     else:
#         return False

# def filter_label2_pairs(clusters_info, labels, max_dist, tolerance):
#     """
#     Filtra le coppie di elementi con label 2 in base alla distanza tra centroidi.
#     Restituisce gli indici dei cluster da mantenere.
#     """
#     label2_clusters = [lab for lab, info in clusters_info.items() 
#                        if info['classification'] == 2]
    
#     if len(label2_clusters) < 2:
#         # Se c'è 0 o 1 elemento con label 2, mantieni tutti
#         return list(clusters_info.keys())
    
#     # Trova la coppia che rispetta il limite di distanza
#     valid_pair = None
#     for i, lab1 in enumerate(label2_clusters):
#         for lab2 in label2_clusters[i+1:]:
#             centroid1 = clusters_info[lab1]['centroid']
#             centroid2 = clusters_info[lab2]['centroid']
#             dist = np.linalg.norm(centroid1 - centroid2)
#             if ((dist <= max_dist + tolerance) and dist >= (max_dist - tolerance)):
#                 valid_pair = (lab1, lab2)
#                 break
#         if valid_pair:
#             break
    
#     # Crea lista dei cluster da mantenere
#     clusters_to_keep = []
#     for lab in clusters_info.keys():
#         if clusters_info[lab]['classification'] != 2:
#             # Mantieni tutti i cluster che non sono label 2
#             clusters_to_keep.append(lab)
#         elif valid_pair and lab in valid_pair:
#             # Mantieni solo i cluster della coppia valida
#             clusters_to_keep.append(lab)
    
#     return clusters_to_keep

# # ===============================
# # LOOP PRINCIPALE
# # ===============================
# def main():
#     segments = []
#     frame_numbers = []
#     segment_counters = []
#     clusters_info = {}

#     # Setup figura per visualizzazione continua
#     plt.ion()  # Modalità interattiva
#     fig, ax = plt.subplots(figsize=(8, 8))

#     while True:
#         segment, frame_number, segment_counter = receiver.receive_segments(1)

#         segments.append(segment[0])
#         frame_numbers.append(frame_number[0])
#         segment_counters.append(segment_counter[0])
        
#         # Controlla se abbiamo 7 segmenti dello stesso frame in ordine
#         if len(frame_numbers) == 7 and len(segment_counters) == 7:
#             if np.all(np.array(frame_numbers) == frame_numbers[0]):
#                 if np.all(np.array(segment_counters[:-1]) < np.array(segment_counters[1:])):

#                     print("Ricevuti 7 segmenti di uno stesso frame in ordine corretto")

#                     # ===============================
#                     # RICOSTRUZIONE SCAN
#                     # ===============================
#                     distances = np.vstack([
#                         seg["Modules"][0]["SegmentData"][0]["Distance"][0] 
#                         for seg in segments
#                     ]).reshape(1, -1)

#                     thetas = np.vstack([
#                         seg["Modules"][0]["SegmentData"][0]["ChannelTheta"] 
#                         for seg in segments
#                     ]).reshape(1, -1)

#                     # distances = np.flip(distances.flatten())
#                     # thetas = np.flip(thetas.flatten())


#                     mask = (thetas > (-np.pi-0.004)/2) & (thetas < (np.pi+0.004)/2)
#                     thetas = thetas[mask]
#                     distances = distances[mask]

#                     x = distances * np.cos(thetas) / 1000.0
#                     y = distances * np.sin(thetas) / 1000.0
                    
#                     pts = np.column_stack((x.flatten(), y.flatten()))
                    
#                     #ax.scatter(prediction[0,1], prediction[0,0], color='red', alpha=1.0, marker='x')
#                     print(f"Numero punti scan ricostruito: {pts.shape[0]}")

#                     # ===============================
#                     # DBSCAN
#                     # ===============================
#                     dbscan = DBSCAN(eps=0.25, min_samples=4)
#                     labels = dbscan.fit_predict(pts)
#                     unique_labels = set(labels)

#                     # ===============================
#                     # VISUALIZZAZIONE CONTINUA
#                     # ===============================
#                     ax.clear()
#                     ax.set_title(f'Scan con cluster e classificazione - Frame {frame_numbers[0]}')
#                     ax.set_xlabel('X [m]')
#                     ax.set_ylabel('Y [m]')
#                     ax.axis('equal')
#                     ax.grid()
                    
#                     #ax.scatter(pts[:, 0], pts[:, 1], s=8, color='red')
#                     #ax.scatter(x, y, s=8, color='red')
#                     colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

#                     for lab, col in zip(unique_labels, colors):
#                         cluster_pts = pts[labels == lab]

#                         # Noise
#                         if lab == -1:
#                             col = (0, 0, 0, 1)
#                             ax.scatter(cluster_pts[:, 0], cluster_pts[:, 1], s=6, color=col, label="Noise")
#                             continue

#                         # Cluster troppo piccoli
#                         if len(cluster_pts) < 3 or np.all(cluster_pts == 0.0) or np.all(cluster_pts < 0.05):
#                             print(f"Cluster {lab}: troppo piccolo ({len(cluster_pts)} punti). Saltato.")
#                             continue

#                         # ===============================
#                         # FEATURE EXTRACTION
#                         # ===============================
#                         features = calculate_cluster_features(cluster_pts)
#                         feature_vector = [features[f] for f in FEATURE_ORDER]

#                         # ===============================
#                         # CLASSIFICAZIONE
#                         # ===============================
#                         classification = model.predict([feature_vector])[0]
#                         prob = model.predict_proba([feature_vector])[0][np.argmax(model.predict_proba([feature_vector]))]
#                         clusters_info[lab] = { 
#                             'points' : cluster_pts,
#                             'centroid' : np.mean(cluster_pts, axis=0),
#                             'classification' : classification,
#                             'probability' : prob
#                         }

#                         # ===============================
#                         # DISEGNA CLUSTER
#                         # ===============================
#                         #ax.scatter(cluster_pts[:, 0], cluster_pts[:, 1], s=8, color=col, label=f"{classification} {prob:.3f}")
                    
#                     # ===============================
#                     # FILTRAGGIO COPPIE LABEL 2
#                     # ===============================
#                     clusters_to_keep = filter_label2_pairs(clusters_info, labels, MAX_CENTROID_DISTANCE, DISTANCE_TOLERANCE)
#                     #cluster_pts = np.vstack([clusters_info[lab]['points'] for lab in clusters_to_keep if lab in clusters_info])
#                     # Predizione regressione
#                     filtered_distances = distances.copy()
#                     for lab in unique_labels:
#                         if lab == -1:
#                             continue
#                         if lab not in clusters_to_keep:
#                             # Poni a zero le distanze dei cluster non validi
#                             filtered_distances[labels == lab] = 0.0
                    
#                     filtered_distances = np.where(np.isin(labels, labels_to_keep), filtered_distances, 0.0) / 1000.0
                    
#                     filtered_distances = np.where(filtered_distances == 0.0, 60.0, filtered_distances)
#                     filtered_distances_padded = pad_center(filtered_distances, 1105).reshape(1,-1)
                    
#                     prediction_scaled = model_reg.predict(filtered_distances_padded / (60.0))
#                     prediction = scaler.inverse_transform(prediction_scaled)   
#                     print(prediction)
#                     ax.scatter(cluster_pts[:, 0], cluster_pts[:, 1], s=8, color=col, label=f"{classification} {prob:.3f}")
#                     ax.scatter(prediction[0,0], prediction[0,1], color='red', alpha=1.0, marker='x')
                    
#                     legend1 = ax.legend(loc="upper right", fontsize=8)
#                     ax.add_artist(legend1)

#                     legend_elements = [
#                         Line2D([0], [0], linestyle='None',
#                                label=f"x: {prediction[0,0]:.3f}"),
#                         Line2D([0], [0], linestyle='None',
#                                label=f"y: {prediction[0,1]:.3f}"),
#                         Line2D([0], [0], linestyle='None',
#                                label=f"theta: {prediction[0,2]:.3f}")
#                     ]

#                     ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

#                     plt.draw()
#                     plt.pause(0.05)  # Aggiorna la visualizzazione

#                     # RESET
#                     frame_numbers = []
#                     segment_counters = []
#                     segments = []
#                     clusters_info = {}

#             else:
#                 mask = np.array(frame_numbers) != frame_numbers[0]
#                 frame_numbers = list(np.array(frame_numbers)[mask])
#                 segment_counters = list(np.array(segment_counters)[mask])
#                 segments = list(np.array(segments)[mask])

#         if keyboard.is_pressed('q'):
#             break

#     plt.ioff()
#     plt.close()
#     receiver.close_connection()

# if __name__ == "__main__":
#     main()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import scansegmentapi.compact as CompactApi
from scansegmentapi.udp_handler import UDPHandler
import keyboard
import math
import joblib

# Carica il modello ML
model = joblib.load('best_params.pkl')

def calculate_perimeter(points):
    """Calcola il perimetro di un poligono definito da punti ordinati"""
    perimeter = 0
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        perimeter += np.linalg.norm(p2 - p1)
    return perimeter

def calculate_mean_curvature(points):
    """Calcola la curvatura media del cluster"""
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    
    curvatures = []
    n = len(sorted_points)
    
    for i in range(n):
        p1 = sorted_points[i]
        p2 = sorted_points[(i + 1) % n]
        p3 = sorted_points[(i + 2) % n]
        
        try:
            v1 = p2 - p1
            v2 = p3 - p2
            l1 = np.linalg.norm(v1)
            l2 = np.linalg.norm(v2)
            
            if l1 > 0 and l2 > 0:
                cross = np.cross(np.append(v1, 0), np.append(v2, 0))
                curvature = abs(cross[2]) / ((l1 + l2) / 2)
                curvatures.append(curvature)
        except:
            continue
    
    return np.mean(curvatures) if curvatures else 0

def calculate_cluster_features(points):
    """Calcola tutte le features geometriche per un cluster di punti"""
    features = {}
    features['num_points'] = len(points)
    
    # 1. Bounding box
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    
    # 2. PCA per lunghezza e larghezza orientate
    pca = PCA(n_components=2)
    pca.fit(points)
    points_pca = pca.transform(points)
    pca_min_x, pca_max_x = np.min(points_pca[:, 0]), np.max(points_pca[:, 0])
    pca_min_y, pca_max_y = np.min(points_pca[:, 1]), np.max(points_pca[:, 1])
    features['pca_length'] = pca_max_x - pca_min_x
    features['pca_width'] = pca_max_y - pca_min_y
    
    # 3. Area (usando convex hull)
    try:
        hull = ConvexHull(points)
        features['convex_area'] = hull.volume
        features['convex_perimeter'] = calculate_perimeter(points[hull.vertices])
    except:
        features['convex_area'] = 0
        features['convex_perimeter'] = 0
    
    # 4. Rapporto di aspetto
    if features['pca_width'] > 0:
        features['aspect_ratio'] = features['pca_length'] / features['pca_width']
    else:
        features['aspect_ratio'] = float('inf')
    
    # 5. Curvatura media
    features['mean_curvature'] = calculate_mean_curvature(points)
    
    return features

UDP_IP = "172.16.35.58"
UDP_PORT = 2122

transport = UDPHandler(UDP_IP, UDP_PORT, 65535)
receiver = CompactApi.Receiver(transport)

temp_frame = 0
segments = []
frame_numbers = []
segment_counters = []

while True:
    segment, frame_number, segment_counter = receiver.receive_segments(1)
    
    segments.append(segment[0])
    frame_numbers.append(frame_number[0])
    segment_counters.append(segment_counter[0])
    
    if len(frame_numbers) == 10 and len(segment_counters) == 10:
        if np.all(np.array(frame_numbers) == frame_numbers[0]):
            if np.all(np.array(segment_counters[:-1]) < np.array(segment_counters[1:])):
                print("Ricevuti 10 segmenti di uno stesso frame in ordine corretto")
                print(f'Frame number: {frame_numbers}')
                print(f'Segment counters: {segment_counters}')
                
                distances = np.vstack([seg["Modules"][0]["SegmentData"][0]["Distance"][0] for seg in segments]).reshape(1, -1)
                thetas = np.vstack([seg["Modules"][0]["SegmentData"][0]["ChannelTheta"] for seg in segments]).reshape(1, -1)
                
                x = distances * np.cos(thetas + math.pi/2 ) / 1000.0
                y = distances * np.sin(thetas + math.pi/2) / 1000.0
                
                # DBSCAN clustering
                points = np.column_stack((x.flatten(), y.flatten()))
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                labels = dbscan.fit_predict(points)
                
                # Estrai features e classifica ogni cluster
                unique_labels = set(labels)
                unique_labels.discard(-1)  # Rimuovi il rumore
                
                cluster_classifications = {}
                for label in unique_labels:
                    cluster_mask = labels == label
                    cluster_points = points[cluster_mask]
                    
                    if len(cluster_points) >= 3:
                        features = calculate_cluster_features(cluster_points)
                        
                        # Crea feature vector per il modello
                        feature_vector = [
                            features['num_points'],
                            features['pca_length'],
                            features['pca_width'],
                            features['convex_area'],
                            features['convex_perimeter'],
                            features['aspect_ratio'],
                            features['mean_curvature']
                        ]
                        
                        # Classifica con il modello
                        prediction = model.predict([feature_vector])[0]
                        cluster_classifications[label] = prediction
                
                # Visualizzazione
                plt.figure(figsize=(8,8))
                plt.scatter(x, y, c=labels, s=1, cmap='viridis')
                
                # Aggiungi label ai cluster
                for label in unique_labels:
                    cluster_mask = labels == label
                    cluster_points = points[cluster_mask]
                    centroid = np.mean(cluster_points, axis=0)
                    plt.text(centroid[0], centroid[1], cluster_classifications.get(label, ""), 
                            fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                plt.title(f'Scan ricostruita con cluster - Frame {frame_numbers[0]}')
                plt.xlabel('X [m]')
                plt.ylabel('Y [m]')
                plt.axis('equal')
                plt.grid()
                plt.colorbar(label='Cluster ID')
                plt.show()
                
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

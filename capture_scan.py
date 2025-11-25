'''
import numpy as np
import pandas as pd
import os
import pickle
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import joblib
import keyboard
import scansegmentapi.compact as CompactApi
from scansegmentapi.udp_handler import UDPHandler
from sklearn.cluster import DBSCAN
import feature_extractor
import data_processor
from classifier import ClusterClassifier



class ScanCommunication():
    def __init__(self, ip:str, port:int, model_path:str, scaler_path:str, scale_factor=60000.0, buffer_size=3):
            self.ip = ip
            self.port = port
            self.scale_factor = scale_factor
            self.buffer_size = buffer_size
            self.model_path = model_path
            self.scaler_path = scaler_path
            self.classifier = None
          
    # -------------------------
    #  Funzioni operative
    # -------------------------
    def setup(self):
        """Inizializza connessione, modello e scaler."""
        transport_layer = UDPHandler(self.ip, self.port, 65535)
        receiver = CompactApi.Receiver(transport_layer)
        model = tf.keras.models.load_model(self.model_path)
        self.model = model
        scaler_y = joblib.load(self.scaler_path)
        self.scaler_y = scaler_y
        return receiver

            
    
    # -------------------------
    #  Utility per il processing
    # -------------------------
    def scan_giver_processed(self, segments):
        """Restituisce angoli (theta) e distanze normalizzate dai segmenti LiDAR."""
        try:
            theta = np.concatenate([
                seg["Modules"][0]["SegmentData"][0]["ChannelTheta"].reshape(-1, 1)
                for seg in segments[:5]
            ], axis=0)

            distances = np.concatenate([
                seg["Modules"][0]["SegmentData"][0]["Distance"][0].reshape(-1, 1)
                for seg in segments[:5]
            ], axis=0)

            distances = distances[:1105].T / self.scale_factor
            return theta, distances

        except Exception as e:
            raise RuntimeError(f"Errore durante il processamento dei segmenti: {e}")


    def process_and_predict(self, receiver):
        """Riceve i segmenti, elabora i dati e calcola la predizione."""
        segments, frame_numbers, _ = receiver.receive_segments(5)
        theta, distances = self.scan_giver_processed(segments)
        prediction = self.model.predict(distances, verbose=0).reshape(1, -1)
        prediction = self.scaler_y.inverse_transform(prediction)
        return theta, distances, prediction

    def mean_prediction(self, predictions):

        return np.mean(predictions, axis=0).reshape(1, -1)
    
    def cluster_detection(self, scans, eps=0.25, min_samples=4):
        data_processor = data_processor.DataProcessor(eps=eps, min_samples=min_samples)
        cluster_label = data_processor._apply_clustering(scans)
        features_dict = {}
        for label in np.unique(cluster_label):
            features= feature_extractor.calculate_cluster_features(scans, cluster_label)
            features_dict[label] = features
        return features_dict
        
        

    def cluster_classification(self, features_dict, model_classifier):
        classifier = ClusterClassifier()
        model_loaded = classifier.load_model(model_classifier)
        classified_clusters = {}
        
        if classifier.is_loaded():
            for cluster_id, features in features_dict.items():
                prediction = classifier.predict(features)
                classified_clusters[cluster_id] = prediction
            
    
        return classified_clusters
    
    # -------------------------
    #  Funzioni di salvataggio
    # -------------------------
    
    def save_scan(self, theta, distances, ):
        """Salva i dati delle nuove scansioni in file npy per avere più dati per training futuri."""
        directory = "scans"
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_number = len(os.listdir(directory))
        np.savez_compressed(os.path.join(directory, f'scan_{file_number:04d}.npz'), theta=theta, distances=distances)
        
    
    def save_results(self, df_results):
        """Salva i risultati in un file CSV."""
        directory = "results"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        file_number = len(os.listdir(directory))
        filename = f'predictions_{file_number:04d}.csv'
        filepath = os.path.join(directory, filename)
        df_results.to_csv(filepath, index=False)
        print(f"✅ File salvato: {filename}")

       


# -------------------------
#  Main Loop
# -------------------------
def main():
    IP = "172.16.35.58"
    PORT = 2122
    SCALE_FACTOR = 5000.0  # Normalizzazione delle distanze
    PREDICTION_BUFFER_SIZE = 1
    MODEL_PATH = 'my_model.h5'
    SCALER_PATH = 'scaler_y.pkl'

    predictions = np.empty((0, 3))
    scan_comm = ScanCommunication(ip=IP, port=PORT, model_path=MODEL_PATH, scaler_path=SCALER_PATH, scale_factor=SCALE_FACTOR, buffer_size=PREDICTION_BUFFER_SIZE)
    receiver = scan_comm.setup()
    df_results = pd.DataFrame(columns=['x', 'y', 'theta'])
    flag = False
    i = 0

    print("📡 Ricezione dati avviata (premi 's' per fermare)")

    try:
        while receiver:
            try:
                theta, distances, prediction = scan_comm.process_and_predict(receiver)
                predictions = np.vstack([predictions, prediction])

                if predictions.shape[0] == PREDICTION_BUFFER_SIZE:

                    prediction = scan_comm.mean_prediction(predictions)
                    df_results = pd.concat([
                        df_results,
                        pd.DataFrame(prediction, columns=['x', 'y', 'theta'])
                    ], ignore_index=True) 

                    print(f"Frame {i}: {prediction}")
                    i += 1

                    if flag:
                        scan_comm.save_scan(theta, distances)
                        
                    predictions = np.empty((0, 3)) 

                if keyboard.is_pressed('a'):

                    flag = not flag

                    if flag:
                        print("⏹ Salvataggio dati di scan attivato.")
                    else:
                        print("▶️ Salvataggio dati di scan disattivato.")

                   
                if keyboard.is_pressed('s'):
                    print("⏹ Interruzione richiesta.")
                    break

            except Exception as e:
                print(f"⚠️ Errore nel ciclo principale: {e}")
                continue

    finally:
        print("🔒 Chiusura connessione e salvataggio risultati...")
        receiver.close_connection()
        scan_comm.save_results(df_results)


if __name__ == "__main__":
    main()
'''
import numpy as np
import pandas as pd
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import joblib
import keyboard
import scansegmentapi.compact as CompactApi
from scansegmentapi.udp_handler import UDPHandler
from sklearn.cluster import DBSCAN
import feature_extractor
import data_processor
from classifier import ClusterClassifier


class ScanCommunication():
    def __init__(self, ip: str, port: int, model_path: str, scaler_path: str, 
                 classifier_path: str, scale_factor=60000.0, buffer_size=3,
                 eps=0.25, min_samples=4, lidar_range=10.0, 
                 localization_labels=[1, 2], non_landmark_value=0.0):
        self.ip = ip
        self.port = port
        self.scale_factor = scale_factor
        self.buffer_size = buffer_size
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.classifier_path = classifier_path
        self.lidar_range = lidar_range
        self.localization_labels = localization_labels
        self.non_landmark_value = non_landmark_value
        
        self.data_processor = data_processor.DataProcessor(eps=eps, min_samples=min_samples)
        self.classifier = ClusterClassifier()
        self.classifier.load_model(classifier_path)


    def setup(self):
        """Inizializza connessione, modello e scaler."""
        transport_layer = UDPHandler(self.ip, self.port, 65535)
        receiver = CompactApi.Receiver(transport_layer)
        self.model = tf.keras.models.load_model(self.model_path)
        self.scaler_y = joblib.load(self.scaler_path)
        return receiver



    def receive_lidar_data(self, receiver):
        """Riceve i segmenti e restituisce i dati grezzi del LiDAR."""
        segments, frame_numbers, _ = receiver.receive_segments(5)
        theta = np.concatenate([
            seg["Modules"][0]["SegmentData"][0]["ChannelTheta"].reshape(-1, 1)
            for seg in segments[:5]
        ], axis=0)

        distances = np.concatenate([
            seg["Modules"][0]["SegmentData"][0]["Distance"][0].reshape(-1, 1)
            for seg in segments[:5]
        ], axis=0)

        distances_normalized = distances[:1105].T / self.scale_factor
        distances_raw = distances[:1105].flatten()
        theta_flat = theta[:1105].flatten()
        
        return theta_flat, distances_raw, distances_normalized



    def cluster_detection_and_classification(self, angles, scans):
        """Clustering e classificazione dei dati LiDAR."""
        # Filtra punti validi e converti in cartesiano
        valid_indices = scans < self.lidar_range
        valid_scans = scans[valid_indices]
        valid_angles = angles[valid_indices]
        
        points_x = valid_scans * np.cos(valid_angles)
        points_y = valid_scans * np.sin(valid_angles)
        points = np.vstack((points_x, points_y)).T
        
        # Clustering
        cluster_labels = self.data_processor._apply_clustering(points)
        n_clusters, n_noise = self.data_processor.get_cluster_info(cluster_labels)
        centroids = self.data_processor.extract_centroids(points, cluster_labels)
        
        # Classifica ogni cluster
        cluster_results = []
        
        if n_clusters > 0:
            unique_labels = set(cluster_labels)
            
            for label in unique_labels:
                if label == -1:
                    continue
                
                cluster_points = self.data_processor.get_cluster_points(points, cluster_labels, label)
                features = feature_extractor.calculate_cluster_features(cluster_points, label)
                
                if features is None:
                    continue
                
                # Classificazione
                classification_result = self.classifier.predict(features)
                class_label = None
                confidence = 0.0
                
                try:
                    if '(' in classification_result:
                        parts = classification_result.split('(')
                        class_label = int(parts[0].strip().strip('[]'))
                        confidence = float(parts[1].strip(')'))
                    else:
                        class_label = int(classification_result.strip('[]'))
                except:
                    pass
                
                # Trova indici originali
                cluster_original_indices = []
                point_idx = 0
                for i, is_valid in enumerate(valid_indices):
                    if is_valid:
                        if cluster_labels[point_idx] == label:
                            cluster_original_indices.append(i)
                        point_idx += 1
                
                cluster_info = {
                    'cluster_id': label,
                    'centroid': centroids.get(label, [0, 0]),
                    'num_points': features['num_points'],
                    'class_label': class_label,
                    'confidence': confidence,
                    'original_indices': cluster_original_indices
                }
                
                cluster_results.append(cluster_info)
        
        return {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'all_points': points,
            'cluster_labels': cluster_labels,
            'clusters': cluster_results
        }
    

    def filter_lidar_by_landmarks(self, distances_raw, cluster_data):
        """Filtra i dati LiDAR mantenendo solo i punti dei cluster con labels valide."""
        filtered_distances = np.full_like(distances_raw, self.non_landmark_value)
        landmark_clusters = []
        
        for cluster in cluster_data['clusters']:
            if cluster['class_label'] in self.localization_labels:
                for idx in cluster['original_indices']:
                    if idx < len(filtered_distances):
                        filtered_distances[idx] = distances_raw[idx]
                landmark_clusters.append(cluster)
        
        return filtered_distances, landmark_clusters


    def predict_pose(self, distances_normalized):
        """Predice la posa del robot."""
        prediction = self.model.predict(distances_normalized, verbose=0).reshape(1, -1)
        return self.scaler_y.inverse_transform(prediction)


    def mean_prediction(self, predictions):
        """Calcola la media delle predizioni."""
        return np.mean(predictions, axis=0).reshape(1, -1)
    

    def save_scan(self, theta, distances):
        """Salva i dati delle scansioni."""
        directory = "scans"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_number = len(os.listdir(directory))
        np.savez_compressed(
            os.path.join(directory, f'scan_{file_number:04d}.npz'), 
            theta=theta, 
            distances=distances
        )
    

    def save_clusters(self, cluster_data, frame_number):
        """Salva i dati dei cluster."""
        directory = "clusters"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, f'clusters_frame_{frame_number:04d}.npz')
        np.savez_compressed(
            filepath,
            n_clusters=cluster_data['n_clusters'],
            n_noise_points=cluster_data['n_noise_points'],
            all_points=cluster_data['all_points'],
            cluster_labels=cluster_data['cluster_labels']
        )
        

    def save_results(self, df_results):
        """Salva i risultati in CSV."""
        directory = "results"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_number = len(os.listdir(directory))
        filepath = os.path.join(directory, f'predictions_{file_number:04d}.csv')
        df_results.to_csv(filepath, index=False)
        print(f"Risultati salvati: {filepath}")


def main():
    # Parametri
    IP = "172.16.35.58"
    PORT = 2122
    SCALE_FACTOR = 5000.0
    PREDICTION_BUFFER_SIZE = 1
    MODEL_PATH = 'my_model.h5'
    SCALER_PATH = 'scaler_y.pkl'
    CLASSIFIER_PATH = 'classifier_pipeline.pkl'
    EPS = 0.25
    MIN_SAMPLES = 4
    LIDAR_RANGE = 10.0
    LOCALIZATION_LABELS = [1, 2]
    NON_LANDMARK_VALUE = 0.0
    
    # Inizializzazione
    scan_comm = ScanCommunication(
        ip=IP, port=PORT, model_path=MODEL_PATH, scaler_path=SCALER_PATH,
        classifier_path=CLASSIFIER_PATH, scale_factor=SCALE_FACTOR, 
        buffer_size=PREDICTION_BUFFER_SIZE, eps=EPS, min_samples=MIN_SAMPLES,
        lidar_range=LIDAR_RANGE, localization_labels=LOCALIZATION_LABELS,
        non_landmark_value=NON_LANDMARK_VALUE
    )
    
    receiver = scan_comm.setup()
    predictions = np.empty((0, 3))
    df_results = pd.DataFrame(columns=['x', 'y', 'theta'])
    flag_save_scans = False
    frame_counter = 0

    print(f"Sistema avviato | Labels landmark: {LOCALIZATION_LABELS}")
    print("Comandi: 's'=stop | 'a'=toggle salvataggio scans\n")

    try:
        while receiver:
            try:
                # 1. Acquisizione dati
                theta, distances_raw, distances_normalized = scan_comm.receive_lidar_data(receiver)
                
                # 2. Clustering e classificazione
                cluster_data = scan_comm.cluster_detection_and_classification(theta, distances_raw)
                
                # 3. Filtraggio punti
                filtered_distances, landmark_clusters = scan_comm.filter_lidar_by_landmarks(distances_raw, cluster_data)
                
                # 4. Predizione (solo se ci sono landmark)
                if landmark_clusters:
                    filtered_distances_normalized = (filtered_distances / SCALE_FACTOR).reshape(1, -1)
                    prediction = scan_comm.predict_pose(filtered_distances_normalized)
                    predictions = np.vstack([predictions, prediction])
                    
                    if predictions.shape[0] == PREDICTION_BUFFER_SIZE:
                        final_prediction = scan_comm.mean_prediction(predictions)
                        predicted_pose = final_prediction[0]
                        
                        df_results = pd.concat([
                            df_results,
                            pd.DataFrame(final_prediction, columns=['x', 'y', 'theta'])
                        ], ignore_index=True)
                        
                        print(f"Frame {frame_counter} | Landmarks: {len(landmark_clusters)} | "
                              f"Pose: x={predicted_pose[0]:.2f} y={predicted_pose[1]:.2f} θ={predicted_pose[2]:.2f}")
                        
                        predictions = np.empty((0, 3))
                        
                        if flag_save_scans:
                            scan_comm.save_scan(theta, distances_raw)
                else:
                    print(f"Frame {frame_counter} | Nessun landmark - predizione saltata")
                
                scan_comm.save_clusters(cluster_data, frame_counter)
                frame_counter += 1

                # Gestione tasti
                if keyboard.is_pressed('a'):
                    flag_save_scans = not flag_save_scans
                    print(f"Salvataggio scans: {'ON' if flag_save_scans else 'OFF'}")
                    import time
                    time.sleep(0.3)
                
                if keyboard.is_pressed('s'):
                    print("Interruzione richiesta")
                    break

            except Exception as e:
                print(f"Errore: {e}")
                continue

    finally:
        print(f"\nChiusura | Frame processati: {frame_counter}")
        receiver.close_connection()
        scan_comm.save_results(df_results)


if __name__ == "__main__":
    main()
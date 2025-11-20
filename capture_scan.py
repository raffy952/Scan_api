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
                 classifier_path: str = None, scale_factor=60000.0, buffer_size=3,
                 eps=0.25, min_samples=4, lidar_range=10.0):
        self.ip = ip
        self.port = port
        self.scale_factor = scale_factor
        self.buffer_size = buffer_size
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.classifier_path = classifier_path
        self.lidar_range = lidar_range
        
        # Inizializza il processore dati per clustering
        self.data_processor = data_processor.DataProcessor(eps=eps, min_samples=min_samples)
        
        # Inizializza il classificatore se il path è fornito
        self.classifier = None
        if classifier_path and os.path.exists(classifier_path):
            self.classifier = ClusterClassifier()
            self.classifier.load_model(classifier_path)
            print("✅ Classificatore caricato con successo")


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

            distances_normalized = distances[:1105].T / self.scale_factor
            distances_raw = distances[:1105].flatten()
            theta_flat = theta[:1105].flatten()
            
            return theta_flat, distances_raw, distances_normalized

        except Exception as e:
            raise RuntimeError(f"Errore durante il processamento dei segmenti: {e}")


    def process_and_predict(self, receiver):
        """Riceve i segmenti, elabora i dati e calcola la predizione."""
        segments, frame_numbers, _ = receiver.receive_segments(5)
        theta, distances_raw, distances_normalized = self.scan_giver_processed(segments)
        prediction = self.model.predict(distances_normalized, verbose=0).reshape(1, -1)
        prediction = self.scaler_y.inverse_transform(prediction)
        return theta, distances_raw, distances_normalized, prediction


    def mean_prediction(self, predictions):
        """Calcola la media delle predizioni."""
        return np.mean(predictions, axis=0).reshape(1, -1)
    

    def cluster_detection_and_classification(self, angles, scans):
        """
        Esegue clustering sui dati LiDAR GREZZI e classifica i cluster trovati.
        
        Args:
            angles: array degli angoli della scansione
            scans: array delle distanze misurate
            
        Returns:
            dict con informazioni sui cluster e le loro classificazioni
        """
        # Filtra i punti validi (entro il range del LiDAR)
        valid_indices = scans < self.lidar_range
        valid_scans = scans[valid_indices]
        valid_angles = angles[valid_indices]
        
        # Converti in coordinate polari -> cartesiane (riferimento robot)
        points_x = valid_scans * np.cos(valid_angles)
        points_y = valid_scans * np.sin(valid_angles)
        points = np.vstack((points_x, points_y)).T
        
        # Applica clustering direttamente sui punti nel frame del robot
        cluster_labels = self.data_processor._apply_clustering(points)
        
        # Ottieni informazioni sui cluster
        n_clusters, n_noise = self.data_processor.get_cluster_info(cluster_labels)
        
        # Estrai centroidi (nel frame del robot)
        centroids = self.data_processor.extract_centroids(points, cluster_labels)
        
        # Analizza ogni cluster
        cluster_results = []
        
        if n_clusters > 0:
            unique_labels = set(cluster_labels)
            
            for label in unique_labels:
                if label == -1:  # Salta il rumore
                    continue
                
                # Ottieni i punti del cluster
                cluster_points = self.data_processor.get_cluster_points(
                    points, cluster_labels, label
                )
                
                # Calcola features geometriche
                features = feature_extractor.calculate_cluster_features(
                    cluster_points, label
                )
                
                if features is None:
                    continue
                
                # Classifica il cluster se il classificatore è disponibile
                classification = "Unknown"
                if self.classifier and self.classifier.is_loaded():
                    classification = self.classifier.predict(features)
                
                # Salva risultati
                cluster_info = {
                    'cluster_id': label,
                    'centroid': centroids.get(label, [0, 0]),
                    'num_points': features['num_points'],
                    'classification': classification,
                    'features': features,
                    'points': cluster_points
                }
                
                cluster_results.append(cluster_info)
        
        return {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'all_points': points,
            'cluster_labels': cluster_labels,
            'clusters': cluster_results
        }
    

    # -------------------------
    #  Funzioni di salvataggio
    # -------------------------
    def save_scan(self, theta, distances):
        """Salva i dati delle nuove scansioni in file npy per avere più dati per training futuri."""
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
        """Salva i dati dei cluster rilevati."""
        directory = "clusters"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        filename = f'clusters_frame_{frame_number:04d}.npz'
        filepath = os.path.join(directory, filename)
        
        # Prepara i dati per il salvataggio
        np.savez_compressed(
            filepath,
            n_clusters=cluster_data['n_clusters'],
            n_noise_points=cluster_data['n_noise_points'],
            all_points=cluster_data['all_points'],
            cluster_labels=cluster_data['cluster_labels']
        )
        

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
    # Parametri di connessione
    IP = "172.16.35.58"
    PORT = 2122
    SCALE_FACTOR = 5000.0
    PREDICTION_BUFFER_SIZE = 1
    
    # Parametri modelli
    MODEL_PATH = 'my_model.h5'
    SCALER_PATH = 'scaler_y.pkl'
    CLASSIFIER_PATH = 'classifier_pipeline.pkl'  # Path al modello di classificazione
    
    # Parametri clustering
    EPS = 0.25
    MIN_SAMPLES = 4
    LIDAR_RANGE = 10.0
    
    # Inizializzazione
    predictions = np.empty((0, 3))
    scan_comm = ScanCommunication(
        ip=IP, 
        port=PORT, 
        model_path=MODEL_PATH, 
        scaler_path=SCALER_PATH,
        classifier_path=CLASSIFIER_PATH,
        scale_factor=SCALE_FACTOR, 
        buffer_size=PREDICTION_BUFFER_SIZE,
        eps=EPS,
        min_samples=MIN_SAMPLES,
        lidar_range=LIDAR_RANGE
    )
    
    receiver = scan_comm.setup()
    df_results = pd.DataFrame(columns=['x', 'y', 'theta'])
    
    # Flag di controllo
    flag_save_scans = False
    flag_clustering = False
    
    frame_counter = 0

    print("📡 Ricezione dati avviata")
    print("   's' = ferma acquisizione")
    print("   'a' = attiva/disattiva salvataggio scans")
    print("   'c' = attiva/disattiva clustering e classificazione")

    try:
        while receiver:
            try:
                # Acquisizione e predizione posizione
                theta, distances_raw, distances_normalized, prediction = scan_comm.process_and_predict(receiver)
                predictions = np.vstack([predictions, prediction])

                if predictions.shape[0] == PREDICTION_BUFFER_SIZE:
                    # Calcola predizione media - QUESTA È LA NOSTRA STIMA DELLA POSIZIONE
                    final_prediction = scan_comm.mean_prediction(predictions)
                    predicted_pose = final_prediction[0]  # [x, y, theta] STIMATA dal modello
                    
                    df_results = pd.concat([
                        df_results,
                        pd.DataFrame(final_prediction, columns=['x', 'y', 'theta'])
                    ], ignore_index=True) 

                    print(f"\nFrame {frame_counter}:")
                    print(f"  Posizione stimata: x={predicted_pose[0]:.3f}, y={predicted_pose[1]:.3f}, θ={predicted_pose[2]:.3f}")
                    
                    # Clustering e classificazione (se attivato)
                    # Clustering DIRETTO sui dati LiDAR (frame del robot)
                    if flag_clustering:
                        cluster_data = scan_comm.cluster_detection_and_classification(
                            theta, distances_raw / SCALE_FACTOR
                        )
                        
                        print(f"  Cluster trovati: {cluster_data['n_clusters']}")
                        print(f"  Punti rumore: {cluster_data['n_noise_points']}")
                        
                        # Mostra classificazioni
                        for cluster in cluster_data['clusters']:
                            centroid = cluster['centroid']
                            # Converti centroide in coordinate polari per più intuitività
                            dist_from_robot = np.sqrt(centroid[0]**2 + centroid[1]**2)
                            angle_from_robot = np.arctan2(centroid[1], centroid[0])
                            
                            print(f"    Cluster {cluster['cluster_id']}: "
                                  f"{cluster['classification']} | "
                                  f"Punti: {cluster['num_points']} | "
                                  f"Distanza: {dist_from_robot:.2f}m | "
                                  f"Angolo: {np.degrees(angle_from_robot):.1f}°")
                        
                        # Salva dati cluster
                        scan_comm.save_clusters(cluster_data, frame_counter)
                    
                    # Salva scan raw (se attivato)
                    if flag_save_scans:
                        scan_comm.save_scan(theta, distances_raw)
                    
                    frame_counter += 1
                    predictions = np.empty((0, 3))

                # Gestione tasti
                if keyboard.is_pressed('a'):
                    flag_save_scans = not flag_save_scans
                    if flag_save_scans:
                        print("▶️ Salvataggio scan ATTIVATO")
                    else:
                        print("⏸️ Salvataggio scan DISATTIVATO")
                    # Piccolo delay per evitare toggle multipli
                    import time
                    time.sleep(0.3)
                
                if keyboard.is_pressed('c'):
                    flag_clustering = not flag_clustering
                    if flag_clustering:
                        print("▶️ Clustering e classificazione ATTIVATI")
                    else:
                        print("⏸️ Clustering e classificazione DISATTIVATI")
                    import time
                    time.sleep(0.3)
                
                if keyboard.is_pressed('s'):
                    print("⏹️ Interruzione richiesta.")
                    break

            except Exception as e:
                print(f"⚠️ Errore nel ciclo principale: {e}")
                import traceback
                traceback.print_exc()
                continue

    finally:
        print("\n🔒 Chiusura connessione e salvataggio risultati...")
        receiver.close_connection()
        scan_comm.save_results(df_results)
        print("✅ Completato!")


if __name__ == "__main__":
    main()
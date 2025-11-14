import numpy as np
import pandas as pd
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import joblib
import keyboard
import scansegmentapi.compact as CompactApi
from scansegmentapi.udp_handler import UDPHandler



class ScanCommunication():
    def __init__(self, ip:str, port:int, model_path:str, scaler_path:str, scale_factor=60000.0, buffer_size=3):
            self.ip = ip
            self.port = port
            self.scale_factor = scale_factor
            self.buffer_size = buffer_size
            self.model_path = model_path
            self.scaler_path = scaler_path


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
    SCALE_FACTOR = 60000.0  # Normalizzazione delle distanze
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

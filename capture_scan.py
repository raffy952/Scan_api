import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import keyboard
import scansegmentapi.compact as CompactApi
from scansegmentapi.udp_handler import UDPHandler


# -------------------------
#  Utility per il processing
# -------------------------
def scan_giver_processed(segments, scale_factor: float):
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

        distances = distances[:1105].T / scale_factor
        return theta, distances

    except Exception as e:
        raise RuntimeError(f"Errore durante il processamento dei segmenti: {e}")


# -------------------------
#  Funzioni operative
# -------------------------
def setup(ip: str, port: int):
    """Inizializza connessione, modello e scaler."""
    transport_layer = UDPHandler(ip, port, 65535)
    receiver = CompactApi.Receiver(transport_layer)
    model = tf.keras.models.load_model('my_model.h5')
    scaler_y = joblib.load('scaler_y.pkl')
    df_results = pd.DataFrame(columns=['x', 'y', 'theta'])
    return receiver, model, scaler_y, df_results


def process_and_predict(receiver, model, scaler_y, scale_factor):
    """Riceve i segmenti, elabora i dati e calcola la predizione."""
    segments, frame_numbers, _ = receiver.receive_segments(5)
    theta, distances = scan_giver_processed(segments, scale_factor)
    prediction = model.predict(distances, verbose=0).reshape(1, -1)
    prediction = scaler_y.inverse_transform(prediction)
    return theta, distances, prediction

def mean_prediction(predictions):

    return np.mean(predictions, axis=0).reshape(1, -1)


# -------------------------
#  Main Loop
# -------------------------
def main():
    IP = "172.16.35.58"
    PORT = 2122
    SCALE_FACTOR = 60000.0  # oppure 5000.0
    PREDICTION_BUFFER_SIZE = 3
    predictions = np.empty((0, 3))

    receiver, model, scaler_y, df_results = setup(IP, PORT)
    i = 0

    print("📡 Ricezione dati avviata (premi 's' per fermare)")

    try:
        while receiver:
            try:
                theta, distances, prediction = process_and_predict(receiver, model, scaler_y, SCALE_FACTOR)
                predictions = np.vstack([predictions, prediction])

                if predictions.shape[0] == PREDICTION_BUFFER_SIZE:

                    prediction = mean_prediction(predictions)
                    df_results = pd.concat([
                        df_results,
                        pd.DataFrame(prediction, columns=['x', 'y', 'theta'])
                    ], ignore_index=True) 

                    print(f"Frame {i}: {prediction}")
                    i += 1
                    predictions = np.empty((0, 3)) 

                if keyboard.is_pressed('s'):
                    print("⏹ Interruzione richiesta.")
                    break

            except Exception as e:
                print(f"⚠️ Errore nel ciclo principale: {e}")
                continue

    finally:
        print("🔒 Chiusura connessione e salvataggio risultati...")
        receiver.close_connection()
        df_results.to_csv('predictions.csv', index=False, sep=';')
        print("✅ File salvato: predictions.csv")


if __name__ == "__main__":
    main()

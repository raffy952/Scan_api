from capture_scan import ScanCommunication
from sklearn.cluster import DBSCAN
import feature_extractor



def main():

    MODEL_CLASSIFICATION_PATH = 'model/best_params_1.pkl'
    MODEL_REGRESSION_PATH = 'model/best_reg_model.pkl'
    SCALER_PATH = 'model/scaler.pkl'
    IP_ADDRESS = '192.16.35.58'
    PREDICTION_BUFFER_SIZE = 5
    PORT = 2122
    SCALER_FACTOR = 5000.0

    scan_comm = ScanCommunication(ip=IP_ADDRESS, port=PORT, model_path=MODEL_REGRESSION_PATH, 
                                  scaler_path=SCALER_PATH, scale_factor=SCALER_FACTOR, buffer_size=PREDICTION_BUFFER_SIZE)
    



    pass


if __name__ == "__main__":

    main()

import scansegmentapi.compact as CompactApi
from scansegmentapi.udp_handler import UDPHandler
import numpy as np
from sklearn.cluster import DBSCAN
import joblib
from tensorflow.keras.models import load_model
from feature_extractor import calculate_cluster_features


class CaptureLidarData:
    def __init__(self, ip="172.16.35.58", port=2122, lidar_type='sick'):
        self.port = port
        self.lidar_type = lidar_type
        self.receive = None
        self.ip = ip
    def lidar_protocol(self):
        if self.lidar_type == 'sick':
            self.transport = UDPHandler(self.ip, self.port, 65535)
            self.receive = CompactApi.Receiver(self.transport)

        elif self.lidar_type == 'r2000':
            pass

        else:
            raise ValueError("Unsupported LIDAR type")

    def get_scan(self):
        return self.receive.receive_segments(1)


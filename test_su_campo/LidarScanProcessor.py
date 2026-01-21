
from sklearn.cluster import DBSCAN
import numpy as np
from feature_extractor import calculate_cluster_features
from tensorflow.keras.models import load_model

class LidarScanProcessor:
    def __init__(self, eps, min_samples, classification_model, scaler, regression_model,
                 feature_order, width_rectangle, width_pallet_complete1, width_pallet_complete2, tolerance):
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.classification_model = classification_model
        self.scaler = scaler
        self.regression_model = regression_model
        self.feature_order = feature_order
        self.width_rectangle = width_rectangle
        self.width_pallet_complete1 = width_pallet_complete1
        self.width_pallet_complete2 = width_pallet_complete2
        self.tolerance = tolerance

    def get_clusters(self, x, y):
        points = np.column_stack((x, y))
        return self.dbscan.fit_predict(points)

    def classify_clusters(self, x, y, labels):
        unique_labels = set(labels)
        classified_clusters = {}
        pts = np.column_stack((x, y))

        for lab in unique_labels:
            if lab == -1: continue
            cluster_pts = pts[labels == lab]
            if len(cluster_pts) < 3: continue

            features = calculate_cluster_features(cluster_pts)
            feature_vector = [features[f] for f in self.feature_order]
            
            classification = self.classification_model.predict([feature_vector])[0]
            probs = self.classification_model.predict_proba([feature_vector])[0]
            
            classified_clusters[lab] = {
                'points': cluster_pts,
                'centroid': np.mean(cluster_pts, axis=0),
                'classification': classification,
                'probability': np.max(probs)
            }
        return classified_clusters
    
    def filter_logic(self, clusters_info):
        label1_clusters = [lab for lab, info in clusters_info.items() if info['classification'] == 1]
        valid_l1 = None
        for lab in label1_clusters:
            width = np.ptp(clusters_info[lab]['points'][:, 0])
            if abs(width - self.width_pallet_complete1) <= self.tolerance or abs(width - self.width_pallet_complete2) <= self.tolerance:
                valid_l1 = lab
                break
        
        label2_clusters = [lab for lab, info in clusters_info.items() if info['classification'] == 2]
        valid_l2_pair = []
        for i, lab1 in enumerate(label2_clusters):
            for lab2 in label2_clusters[i+1:]:
                dist = np.linalg.norm(clusters_info[lab1]['centroid'] - clusters_info[lab2]['centroid'])
                if abs(dist - self.width_rectangle) <= self.tolerance:
                    valid_l2_pair = [lab1, lab2]
                    break
            if valid_l2_pair: break

        final_valid_ids = []
        if valid_l1 is not None: final_valid_ids.append(valid_l1)
        final_valid_ids.extend(valid_l2_pair)
        return final_valid_ids


    def pose_estimation(self, distances, labels, valid_ids):
        mask = np.isin(labels, valid_ids)
        filtered_dist_input = np.where(mask, distances, 60.0)
        
        n = len(filtered_dist_input)
        if n >= 1105:
            final_input = filtered_dist_input[:1105]
        else:
            pad_left = (1105 - n) // 2
            pad_right = 1105 - n - pad_left
            final_input = np.pad(filtered_dist_input, (pad_left, pad_right), constant_values=60.0)
            
        prediction_scaled = self.regression_model.predict(final_input.reshape(1, -1) / 60.0, verbose=0)
        return self.scaler.inverse_transform(prediction_scaled)

import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN, HDBSCAN

class DataProcessor:
    """Classe per l'elaborazione dei dati LiDAR e il clustering"""
    
    def __init__(self, eps=0.25, min_samples=4, noise_std=0.005):
        self.eps = eps
        self.min_samples = min_samples
        self.noise_std = noise_std
        self.lidar_points_history = []
    
    def process_lidar_scan(self, robot_pose, angles, scans, lidar_range):
        """
        Elabora una scansione LiDAR e restituisce i punti validi con clustering
        """
        # Filtra i punti validi
        valid_indices = scans < lidar_range
        valid_scans = scans[valid_indices] 
        valid_angles = angles[valid_indices]
        
        # Aggiungi rumore ai dati
        noise = np.random.normal(0, self.noise_std, len(valid_scans))
        
        points = np.empty((0, 2))
        cluster_labels = np.array([])
        
        if len(valid_scans) > 0:
            # Converti in coordinate mondiali (per visualizzazione)
            world_angles = robot_pose[2] + valid_angles
            points_x_global = robot_pose[0] + (valid_scans + noise) * np.cos(world_angles) 
            points_y_global = robot_pose[1] + (valid_scans + noise) * np.sin(world_angles) 
            points_global = np.vstack((points_x_global, points_y_global)).T


            # Coordinate locali (calcolo features e clustering)
            local_angles = valid_angles
            points_x_local = (valid_scans + noise) * np.cos(local_angles) 
            points_y_local = (valid_scans + noise) * np.sin(local_angles) 
            points_local = np.vstack((points_x_local, points_y_local)).T
            # Salva i punti nella storia
            self.lidar_points_history.append(points_global)
            
            # Applica clustering DBSCAN
            cluster_labels = self._apply_clustering(points_local)
        
        return points_global, cluster_labels, points_local
    
    def _apply_clustering(self, points):
        """Applica l'algoritmo DBSCAN ai punti"""
        if len(points) == 0:
            return np.array([])
        
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
        #db = HDBSCAN(cluster_selection_epsilon=0.2, metric='canberra').fit(points)
        return db.labels_
    
    def get_cluster_info(self, cluster_labels):
        """Restituisce informazioni sui cluster trovati"""
        if len(cluster_labels) == 0:
            return 0, 0
        
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise_points = list(cluster_labels).count(-1)
        
        return n_clusters, n_noise_points
    
    def extract_centroids(self, points, cluster_labels):
        """Estrae i centroidi dei cluster"""
        centroids = {}
        
        if len(cluster_labels) > 0:
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                if label == -1:  # Ignora i punti rumore
                    continue
                
                points_in_cluster = points[cluster_labels == label]
                centroid = np.median(points_in_cluster, axis=0)
                centroids[label] = centroid
        
        return centroids
    
    def get_cluster_points(self, points, cluster_labels, cluster_id):
        """Restituisce i punti appartenenti a un cluster specifico"""
        if cluster_id in cluster_labels:
            return points[cluster_labels == cluster_id]
        return np.empty((0, 2))
    
    def get_points_history_size(self):
        """Restituisce il numero di scansioni salvate"""
        return len(self.lidar_points_history)
    
    def clear_history(self):
        """Pulisce la storia dei punti LiDAR"""
        self.lidar_points_history.clear()
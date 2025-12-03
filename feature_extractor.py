import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

def calculate_cluster_features(points):
    """
    Calcola tutte le features geometriche per un cluster di punti
    """
    # if len(points) < 3:
    #     return None
    
    features = {}
    #features['cluster_id'] = cluster_id
    features['num_points'] = len(points)
    
    # 1. Bounding box per lunghezza e larghezza
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
        features['convex_area'] = hull.volume  # in 2D volume = area
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

def calculate_perimeter(points):
    """Calcola il perimetro di un poligono definito da punti ordinati"""
    #if len(points) < 3:
        #return 0
    perimeter = 0
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        perimeter += np.linalg.norm(p2 - p1)
    return perimeter

def calculate_mean_curvature(points):
    """Calcola la curvatura media del cluster"""
    # if len(points) < 3:
    #     return 0
    
    # Ordina i punti per distanza dal centroide (approssimazione)
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
        
        # Calcola la curvatura usando tre punti consecutivi
        try:
            # Vettori
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Lunghezze
            l1 = np.linalg.norm(v1)
            l2 = np.linalg.norm(v2)
            
            if l1 > 0 and l2 > 0:
                # Prodotto vettoriale per il seno dell'angolo
                cross = np.cross(np.append(v1, 0), np.append(v2, 0))
                # Curvatura = |sin(theta)| / lunghezza_media
                curvature = abs(cross) / ((l1 + l2) / 2)
                curvatures.append(curvature)
        except:
            continue
    
    return np.mean(curvatures) if curvatures else 0
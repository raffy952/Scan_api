import pickle
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")


class ClusterClassifier:
    """Classe per gestire la classificazione dei cluster usando ML"""
    
    def __init__(self):
        self.classifier_pipeline = None
        self.model_loaded = False
        self.feature_names = ['num_points', 'pca_length', 'pca_width', 'convex_area', 'convex_perimeter', 'aspect_ratio','mean_curvature']

        #self.feature_names = ['convex_perimeter', 'pca_length', 'num_points']

        #with open('scaler.pkl', 'rb') as f:
            #self.scaler = pickle.load(f)

    def load_model(self, model_path):
        """Carica la pipeline di classificazione da file pickle"""
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.classifier_pipeline = pickle.load(f)
                    self.model_loaded = True
                    print(f"Pipeline caricata con successo da: {model_path}")
                    return True
            else:
                print(f"File modello non trovato: {model_path}")
                return False
        except Exception as e:
            print(f"Errore nel caricamento del modello: {e}")
            self.model_loaded = False
            return False

    def predict(self, features_dict):
        """Predice la classe del cluster usando la pipeline caricata"""
        if not self.model_loaded or self.classifier_pipeline is None:
            return "Unknown"
        
        try:
            # Prepara le features per la predizione nell'ordine corretto
            features_list = []
            for name in self.feature_names:
                if name in features_dict:
                    value = features_dict[name]
                    features_list.append(value)
                else:
                    features_list.append(0.0)  # valore di default se manca la feature
            
            # Converti in array numpy e reshape per singola predizione
            features_array = np.array(features_list).reshape(1, -1)
           
            
            # Verifica che non ci siano valori invalidi
            if not np.all(np.isfinite(features_array)):
                print("Attenzione: valori non finiti nelle features")
                return "Invalid"
            
            # Predizione usando la pipeline (che gestisce automaticamente preprocessing + classificazione)
            #features_array = self.scaler.transform(features_array)
            prediction = self.classifier_pipeline.predict(features_array)
            
            # Se la pipeline supporta predict_proba, mostra anche la confidenza
            if hasattr(self.classifier_pipeline, 'predict_proba'):
                try:
                    probabilities = self.classifier_pipeline.predict_proba(features_array)
                    max_prob = np.max(probabilities)
                    
                    return f"{prediction} ({max_prob:.2f})"
                except Exception as e:
                    print(f"Errore nel calcolo delle probabilità: {e}")
                    return str(prediction)
            else:
                return str(prediction)
                
        except Exception as e:
            print(f"Errore nella predizione: {e}")
            print(f"Features utilizzate: {dict(zip(self.feature_names, features_list))}")
            return "Error"

    def is_loaded(self):
        """Restituisce True se il modello è stato caricato correttamente"""
        return self.model_loaded
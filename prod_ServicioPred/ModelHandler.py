from pickle import load
import asyncio
import pandas as pd


class ModelH():
    def __init__(self, model_path: str="Model/model_pipeline.pkl"):
        self.model_path = model_path
        self.model = None
        self.lock = asyncio.Lock()
        self.load_model()

    def load_model(self):
        """Carga el modelo desde disco"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = load(f)
        except Exception as e:
            print("Error loading model from path:", self.model_path, "Error:", e)
    
    async def reload_model(self):
        async with self.lock:
            self.load_model()

    async def predict(self, input_data: pd.DataFrame):
        async with self.lock:
            result = self.model.predict(input_data)
            return result.tolist()
        
    async def get_features(self):
        async with self.lock:
            features = [ft.split('__')[1] for ft in self.model[:-1].get_feature_names_out()]
            return features
        
    def check_model_loaded(self):
        if self.model is None:
            return False
        return True
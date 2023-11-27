# Importar modulos necesarios
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from config.core import app_config, model_config
from processing.data_manager import save_pipeline
from config.core import DATASET_DIR 

class RecommendationModel(BaseEstimator, TransformerMixin):
    def __init__(self, input_data):
        self.atributos_deseados = model_config.atributos_deseados
        self.scaling_method = model_config.scaling_method
        self.generos_usuario = model_config.generos_usuario
        self.sentimiento_usuario = model_config.sentimiento_usuario
        self.n_components = model_config.n_components
        self.top_n = model_config.top_n
        self.data_file = app_config.data_file
        self.input_data = input_data
        
    def fit(self, X=None, y=None):
        return self    
     
    def transform(self, X):
        return self

    def predict(self, X):
        # Inicializa con los parametros de usuario
        if X is not None:
            self.generos_usuario = X['generos_usuario']
            self.sentimiento_usuario = X['sentimiento_usuario']  
        # Lee los datos
        df = pd.read_csv(f"{DATASET_DIR}/{self.data_file}", sep='|')
        subset_df = df[(df['genero_principal'].isin(self.generos_usuario)) & (df['sentimiento'] == self.sentimiento_usuario)]
        if subset_df.shape[0] == 0:
            subset_df = df[(df['genero_principal'].isin(self.generos_usuario)) | (df['sentimiento'] == self.sentimiento_usuario)]
        atributos = subset_df[self.atributos_deseados].values    
        
        if self.scaling_method == "StandardScaler":
            scaler = StandardScaler()
        elif self.scaling_method == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif self.scaling_method == "RobustScaler":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Método de escalado no reconocido: {self.scaling_method}")
        atributos = scaler.fit_transform(atributos)     
        svd = TruncatedSVD(n_components=self.n_components)
        atributos_latentes = svd.fit_transform(atributos)
        atributos_latentes_sparse = csr_matrix(atributos_latentes)   
       
        similitud = cosine_similarity(atributos_latentes_sparse) if self.n_components < min(atributos.shape) else cosine_similarity(atributos)

        indices_recomendaciones = similitud.sum(axis=0).argsort()[::-1]
        recomendaciones = df.iloc[indices_recomendaciones].head(self.top_n)
        return recomendaciones
    
# Crear la canalización
recommendation_pipeline = Pipeline([
    ('recommendation_model', RecommendationModel(input_data = None)),    
])

recomendaciones = recommendation_pipeline.predict(X=None)

save_pipeline(pipeline_to_persist=recommendation_pipeline)

print("Recomendaciones:")
print(recomendaciones)

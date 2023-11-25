# Importar modulos necesarios
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
import joblib

## Funciones del pipeline
# Cargar datos
df = pd.read_csv("./data/df_spotify.csv", sep='|')

# Definir la clase para el escalado y la selección de atributos
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, atributos_deseados, scaling_method, generos_usuario, sentimiento_usuario):
        self.atributos_deseados = atributos_deseados
        self.scaling_method = scaling_method
        self.generos_usuario = generos_usuario
        self.sentimiento_usuario = sentimiento_usuario

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        subset_df = self.filter_data(X)
        
        atributos = subset_df[self.atributos_deseados].values

        if self.scaling_method == "StandardScaler":
            scaler = StandardScaler()
        elif self.scaling_method == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif self.scaling_method == "RobustScaler":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Método de escalado no reconocido: {self.scaling_method}")

        atributos_transformados = scaler.fit_transform(atributos)
        return atributos_transformados
    
    def filter_data(self, X):
        generos_usuario = self.generos_usuario
        seleccion_usuario = self.sentimiento_usuario

        subset_df = X[(X['genero_principal'].isin(generos_usuario)) & (X['sentimiento'] == seleccion_usuario)]
        if subset_df.shape[0] == 0:
            subset_df = X[(X['genero_principal'].isin(generos_usuario)) | (X['sentimiento'] == seleccion_usuario)]

        return subset_df

# Definir la clase para el modelo de recomendación
class RecommendationModel(BaseEstimator):
    def __init__(self, df, n_components, top_n):
        self.df = df
        self.n_components = n_components
        self.top_n = top_n
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        if self.n_components < min(X.shape):
            similitud = cosine_similarity(X)
        else:
            svd = TruncatedSVD(n_components=self.n_components)
            atributos_latentes = svd.fit_transform(X)
            atributos_latentes_sparse = csr_matrix(atributos_latentes)
            similitud = cosine_similarity(atributos_latentes_sparse)

        indices_recomendaciones = similitud.sum(axis=0).argsort()[::-1]
        recomendaciones = self.df.iloc[indices_recomendaciones].head(self.top_n)
        return recomendaciones
    
# Crear la canalización
pipeline = Pipeline([
    ('feature_engineering', FeatureEngineering(
        atributos_deseados=['valence', 'year', 'acousticness', 'danceability', 'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo'], 
        scaling_method="RobustScaler",
        generos_usuario=['Pop'],
        sentimiento_usuario='Amor')),
    ('recommendation_model', RecommendationModel(df=df, n_components=10, top_n=10))
])

# Entrenar el pipeline
pipeline.fit(df)

# Guardar el modelo como un archivo pickle
joblib.dump(pipeline, 'model-pkl/recommendation_model.pkl')

# Realizar predicciones en el conjunto completo
recomendaciones = pipeline.predict(df)

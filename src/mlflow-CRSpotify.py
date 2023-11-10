 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 21:02:20 2023

@author: Santiago Pachon, Kelly Nino, Felipe Valencia y Diana Villalba
"""

# Importe el conjunto de datos de diabetes y divídalo en entrenamiento y prueba usando scikit-learn
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

#Importe MLFlow para registrar los experimentos, el regresor de bosques aleatorios y la métrica de error cuadrático medio
import mlflow
import mlflow.sklearn

# Importar data
df = pd.read_csv("df_spotify.csv", sep='|')

# Metrica
def average_relevance_score(recommendations, attribute_weights):
    """
    Calcula el puntaje de relevancia promedio de las recomendaciones.

    Parameters:
    - recommendations: DataFrame que contiene las canciones recomendadas por el modelo.
    - attribute_weights: Diccionario que asigna un peso a cada atributo utilizado en las recomendaciones.

    Returns:
    - Puntaje de relevancia promedio.
    """
    if recommendations.empty:
        return 0.0

    total_relevance = 0.0
    total_weight = 0.0

    for index, row in recommendations.iterrows():
        relevance = 0.0
        for attribute, weight in attribute_weights.items():
            relevance += row[attribute] * weight
        total_relevance += relevance
        total_weight += sum(attribute_weights.values())

    average_relevance = total_relevance / total_weight
    return average_relevance

# Modelo
def custom_recommendation_model(df, generos_usuario, seleccion_usuario, n_components, scaling_method, top_n):
    
    subset_df = df[(df['genero_principal'].isin(generos_usuario)) | (df['sentimiento'] == seleccion_usuario)]
    atributos_deseados = ['valence', 'year', 'acousticness', 'danceability', 'energy', 'explicit',
                         'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo']
    
    atributos = subset_df[atributos_deseados].values
    
    # Aplicar el escalado
    if scaling_method == "StandardScaler":
        scaler = StandardScaler()
        atributos = scaler.fit_transform(atributos)
    elif scaling_method == "MinMaxScaler":
        scaler = MinMaxScaler()
        atributos = scaler.fit_transform(atributos)
    elif scaling_method == "RobustScaler":
        scaler = RobustScaler()
        atributos = scaler.fit_transform(atributos)
    
    # Reducción de dimensionalidad (SVD)
    # n_components = min(n_components, min(atributos.shape) - 1)
    svd = TruncatedSVD(n_components=n_components)
    atributos_latentes = svd.fit_transform(atributos)
    
    similitud = cosine_similarity(atributos_latentes) if n_components < min(atributos.shape) else cosine_similarity(atributos)
    indices_recomendaciones = similitud.sum(axis=0).argsort()[::-1]
    recomendaciones = subset_df.iloc[indices_recomendaciones].head(top_n)
    
    return recomendaciones

# defina el servidor para llevar el registro de modelos y artefactos
#mlflow.set_tracking_uri('http://localhost:5000')
# registre el experimento
experiment = mlflow.set_experiment("sklearn-spotify")

# Aquí se ejecuta MLflow sin especificar un nombre o id del experimento. MLflow los crea un experimento para este cuaderno por defecto y guarda las características del experimento y las métricas definidas. 
# Para ver el resultado de las corridas haga click en Experimentos en el menú izquierdo. 
with mlflow.start_run(experiment_id=experiment.experiment_id):
    pass
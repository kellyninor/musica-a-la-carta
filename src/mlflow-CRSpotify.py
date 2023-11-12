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
from scipy.sparse import csr_matrix

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
    
    subset_df = df[(df['genero_principal'].isin(generos_usuario)) & (df['sentimiento'] == seleccion_usuario)]
    if subset_df.shape[0] > 0:
        pass
    else: 
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
    
    # Convertir a matriz dispersa
    atributos_latentes_sparse = csr_matrix(atributos_latentes)
    
    # Calcular similitud del coseno
    similitud = cosine_similarity(atributos_latentes_sparse) if n_components < min(atributos.shape) else cosine_similarity(atributos)
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

    # Pesos de la metrica
    attribute_weights = {
        'valence': 0.3,
        'danceability': 0.2,
        'energy': 0.2,
        'acousticness': 0.1,
        'instrumentalness': 0.2
    }

    # defina los parámetros del modelo
    gener_user = ['Otro'] # Pop - Jazz - Hip-Hop/Rap - Rock - Soul - Clásica - Country - Metal - Folk - Indie/Alternativo - R&B - Punk - Electrónica - Reggaetón - Dancehall - Blues - Gospel
    sentm_user = 'Euforia' # Melancolía - Amor - Otro - Euforia - Felicidad - Tristeza - Energía - Relajación - Ira
    n_compt = 10
    scaling_meth = "RobustScaler" # StandardScaler - MinMaxScaler - RobustScaler
    top_n = 10
    
    # Modelo con los parametros relaciones y ejecución de la recomendación
    recomendaciones = custom_recommendation_model(df, generos_usuario = gener_user, seleccion_usuario = sentm_user, n_components = n_compt, scaling_method = scaling_meth, top_n = top_n)
        
   # Registre los parámetros
    mlflow.log_param("generos_user", gener_user)
    mlflow.log_param("sentimiento_user", sentm_user)
    mlflow.log_param("n_components", n_compt)
    mlflow.log_param("scaling_method", scaling_meth)
    mlflow.log_param("top_n", top_n)
  
    # Registre el modelo
    mlflow.sklearn.log_model(custom_recommendation_model, "recomend-spotify-model")
  
    # Cree y registre la métrica de interés
    # Calcula la métrica de relevancia promedio
    average_relevance = average_relevance_score(recomendaciones, attribute_weights)
    mlflow.log_metric("average_relevance", average_relevance)
    print(average_relevance)


def system_cosine_sim(generos_usuario, seleccion_usuario):
    subset_df = df[(df['genero_principal'].isin(generos_usuario)) | (df['sentimiento'] == seleccion_usuario)]
    atributos_deseados = ['valence', 'year', 'acousticness', 'danceability', 'energy', 'explicit',
     'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo']
   
    atributos = subset_df[atributos_deseados].values
    similitud = cosine_similarity(atributos)
    
    indices_recomendaciones = similitud.sum(axis=0).argsort()[::-1]
    recomendaciones = subset_df.iloc[indices_recomendaciones].head(10)
    return recomendaciones


def system_svd(generos_usuario, seleccion_usuario):
    subset_df = df[(df['genero_principal'].isin(generos_usuario)) | (df['sentimiento'] == seleccion_usuario)]
    atributos_deseados = ['valence', 'year', 'acousticness', 'danceability', 'energy', 'explicit',
     'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo']
    atributos = subset_df[atributos_deseados].values

    scaler = StandardScaler() # Estandarización
    atributos_estandarizados = scaler.fit_transform(atributos)

    n_components = min(atributos.shape[0], atributos.shape[1]) - 1 # Componentes
    svd = TruncatedSVD(n_components=n_components) # Modelo
    svd.fit(atributos_estandarizados)

    atributos_latentes = svd.transform(atributos_estandarizados)
    similitud = cosine_similarity(atributos_latentes)

    indices_recomendaciones = similitud.sum(axis=0).argsort()[::-1]
    recomendaciones = subset_df.iloc[indices_recomendaciones].head(10)
    return recomendaciones
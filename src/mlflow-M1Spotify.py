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
def system_cosine_sim(df, generos_usuario, seleccion_usuario):

    subset_df = df[(df['genero_principal'].isin(generos_usuario)) & (df['sentimiento'] == seleccion_usuario)]
    if subset_df.shape[0] > 0:
        pass
    else: 
        subset_df = df[(df['genero_principal'].isin(generos_usuario)) | (df['sentimiento'] == seleccion_usuario)]

    atributos_deseados = ['valence', 'year', 'acousticness', 'danceability', 'energy', 'explicit',
     'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo']
   
    atributos = subset_df[atributos_deseados].values
    similitud = cosine_similarity(atributos)
    
    indices_recomendaciones = similitud.sum(axis=0).argsort()[::-1]
    recomendaciones = subset_df.iloc[indices_recomendaciones].head(10)
    return recomendaciones

# defina el servidor para llevar el registro de modelos y artefactos
#mlflow.set_tracking_uri('http://localhost:5000')
# registre el experimento
experiment = mlflow.set_experiment("M1 (Modelo de Similitud de Coseno)")

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
    
    # Modelo con los parametros relaciones y ejecución de la recomendación
    recomendaciones = system_cosine_sim(df, generos_usuario = gener_user, seleccion_usuario = sentm_user)
        
   # Registre los parámetros
    mlflow.log_param("generos_user", gener_user)
    mlflow.log_param("sentimiento_user", sentm_user)
  
    # Registre el modelo
    mlflow.sklearn.log_model(system_cosine_sim, "recomend-spotify-model")
  
    # Cree y registre la métrica de interés
    # Calcula la métrica de relevancia promedio
    average_relevance = average_relevance_score(recomendaciones, attribute_weights)
    mlflow.log_metric("average_relevance", average_relevance)
    print(average_relevance)

 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 21:02:20 2023

@author: Santiago Pachon, Kelly Nino, Felipe Valencia y Diana Villalba
"""

# Importe el conjunto de datos de diabetes y divídalo en entrenamiento y prueba usando scikit-learn
import pandas as pd
import numpy as np

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
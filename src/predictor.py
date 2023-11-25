import pandas as pd
import joblib
from pipeline import FeatureEngineering  
from pipeline import RecommendationModel 


def make_prediction(data, model_path="./model-pkl/recommendation_model.pkl"):
    """
    Realiza predicciones utilizando el modelo de recomendación cargado desde el archivo pkl.

    Parameters:
    - data: DataFrame con los datos para hacer la predicción.
    - model_path: Ruta del archivo pkl que contiene el modelo de recomendación.

    Returns:
    - DataFrame con las recomendaciones.
    """
    # Cargar el modelo desde el archivo pkl
    recommendation_model = joblib.load(model_path)

    # Realizar la predicción
    recommendations = recommendation_model.predict(data)

    return recommendations

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar datos
    data = pd.read_csv("./data/df_spotify.csv", sep='|')

    # Realizar la predicción
    predictions = make_prediction(data)

    # Imprimir las recomendaciones
    print("Recomendaciones:")
    print(predictions)



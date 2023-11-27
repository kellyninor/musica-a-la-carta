import typing as t
import pandas as pd
from config.core import app_config, model_config
from processing.data_manager import load_pipeline
from pipeline import RecommendationModel

pipeline_file_name = f"{app_config.pipeline_save_file}.pkl"
recommendation_pipeline = load_pipeline(file_name=pipeline_file_name)

def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict]
) -> dict:
    """
    Realiza predicciones utilizando el modelo de recomendación cargado desde el archivo pkl.

    Parameters:
    - input_data: DataFrame con los datos para hacer la predicción.
    - default_data: DataFrame con los datos predeterminados si no se proporciona input_data.

    Returns:
    - DataFrame con las recomendaciones.
    """
    # Realizar la predicción
    atributos = recommendation_pipeline.predict(input_data)
    recommendations = recommendation_pipeline.predict(atributos)

    return recommendations

# Ejemplo de uso
if __name__ == "__main__":
    # Datos proporcionados por el usuario
    user_data = {
        'generos_usuario': ['Rock'],
        'sentimiento_usuario': 'Amor'
    }

    # Realizar la predicción utilizando los datos del usuario o los predeterminados
    predictions = make_prediction(input_data=user_data)

    # Imprimir las recomendaciones
    print("Recomendaciones:")
    print(predictions)



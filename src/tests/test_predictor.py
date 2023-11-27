import unittest
import pandas as pd
import joblib
from pathlib import Path
from unittest.mock import MagicMock

class TestPredictor(unittest.TestCase):

    def setUp(self):
        # Datos de prueba
        file_name = "df_spotify.csv"  
        self.env_dir = Path(__file__).resolve().parent.parent
        path_csv = f"{self.env_dir}/model/datasets/{file_name}"
        self.test_data = pd.read_csv(path_csv, sep='|')

    def test_make_prediction(self):
        # Mock para joblib.load
        joblib_mock = MagicMock()
        joblib_mock.return_value = MagicMock(predict=MagicMock(return_value=pd.DataFrame()))

        # Mock para joblib
        with unittest.mock.patch("joblib.load", joblib_mock):           
            # Verificar que la función de predicción devuelve un DataFrame
            file_pkl = "recommendation_model.pkl"
            #env_dir = Path(__file__).resolve().parent.parent
            model_path=f"{self.env_dir}/model/trained/{file_pkl}"
            recommendation_model = joblib.load(model_path)

            # Realizar la predicción
            recommendations = recommendation_model.predict(self.test_data)
            self.assertIsInstance(recommendations, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
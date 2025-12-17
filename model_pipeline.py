from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from preprocessing import Preprocessor

class ModelPipeline:

    def build_pipeline(self, numeric_features):
        try:
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numeric_features)
                ]
            )

            pipeline = Pipeline(steps=[
                ("cleaning", Preprocessor()),
                ("scaling", preprocessor),
                ("model", LinearRegression())
            ])
            return pipeline

        except Exception as e:
            raise Exception(f"Pipeline creation error: {e}")

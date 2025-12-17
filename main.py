import sys
import os

# Add src folder to path (Windows-safe)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import DataLoader
from model_pipeline import ModelPipeline
from trainer import ModelTrainer
from evaluator import Evaluator

def main():
    try:
        # Get absolute paths for data files
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        calories_path = os.path.join(BASE_DIR, "data", "calories.csv")
        exercise_path = os.path.join(BASE_DIR, "data", "exercise.csv")

        # Load data
        loader = DataLoader(calories_path, exercise_path)
        data = loader.load_and_merge()

        # Features & target
        X = data.drop(columns=["Calories", "User_ID"])
        y = data["Calories"]

        numeric_features = X.columns.tolist()

        # Build ML pipeline
        pipeline_builder = ModelPipeline()
        pipeline = pipeline_builder.build_pipeline(numeric_features)

        # Train & save model
        trainer = ModelTrainer()
        model, X_test, y_test = trainer.train_and_save(
            pipeline, X, y, os.path.join(BASE_DIR, "artifacts", "final_model.pkl")
        )

        # Evaluate
        evaluator = Evaluator()
        results = evaluator.evaluate(model, X_test, y_test)

        print("Evaluation Results:")
        for k, v in results.items():
            print(f"{k}: {v}")

    except Exception as e:
        print("Project Error:", e)

if __name__ == "__main__":
    main()

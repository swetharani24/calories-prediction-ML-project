import pandas as pd

class DataLoader:
    def __init__(self, calories_path, exercise_path):
        self.calories_path = calories_path
        self.exercise_path = exercise_path

    def load_and_merge(self):
        try:
            calories = pd.read_csv(self.calories_path)
            exercise = pd.read_csv(self.exercise_path)

            # Merge on User_ID
            data = exercise.merge(calories, on="User_ID")
            return data

        except Exception as e:
            raise Exception(f"Error loading data: {e}")

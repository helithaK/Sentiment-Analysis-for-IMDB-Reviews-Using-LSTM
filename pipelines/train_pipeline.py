from models.train import train_model
from utils.visualization import plot_training

def run_training_pipeline(data_path, save_model_path):
    history = train_model(data_path, save_model_path)
    plot_training(history)
    print("Training complete. Model saved at", save_model_path)
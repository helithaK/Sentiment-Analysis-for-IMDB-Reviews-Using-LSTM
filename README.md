# Sentiment Analysis for IMDB Reviews using LSTM

This project implements a sentiment analysis pipeline using a Long Short-Term Memory (LSTM) neural network to classify IMDB reviews as positive or negative. The project is designed with a modular programming structure for ease of scalability and maintainability.

## Project Structure

```
sentiment_analysis/
|-- data/
|   |-- preprocess_data.py  # Data preprocessing and tokenization
|-- models/
|   |-- model.py            # LSTM model creation
|   |-- train.py            # Model training
|-- utils/
|   |-- visualization.py    # Visualization utilities for training performance
|   |-- metrics.py          # Custom metrics and evaluation
|-- pipelines/
|   |-- train_pipeline.py   # Training pipeline
|   |-- inference_pipeline.py # Inference pipeline
|-- main.py                 # Main script to execute the pipeline
|-- requirements.txt        # Dependencies
|-- README.md               # Project description
```

## Features

- Preprocessing of raw text data with tokenization and padding.
- LSTM model implementation for sentiment classification.
- Training pipeline with loss and accuracy visualization.
- Inference pipeline for real-time predictions on new data.
- Modular design for easy extension and experimentation.

## Getting Started

### Prerequisites

Ensure you have Python 3.8 or later installed. Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

### Dataset

The dataset should be a `.txt` file with each line containing a label (0 or 1) and a review, separated by a tab (`\t`). Example:

```
1	This movie was amazing, I loved every part of it!
0	The film was dull and boring, a complete waste of time.
```

### Running the Project

#### Training the Model

Use the following command to train the model:

```bash
python main.py --mode train --data_path <path_to_dataset> --save_model <path_to_save_model>
```

- `--data_path`: Path to the dataset file.
- `--save_model`: Path to save the trained model.

#### Running Inference

To classify new reviews, use the inference mode:

```bash
python main.py --mode inference --model_path <path_to_model> --input_texts <text1> <text2> ...
```

- `--model_path`: Path to the trained model file.
- `--input_texts`: List of texts to classify.

### Example

#### Training

```bash
python main.py --mode train --data_path data/imdb_reviews.txt --save_model models/sentiment_model.h5
```

#### Inference

```bash
python main.py --mode inference --model_path models/sentiment_model.h5 --input_texts "I loved this movie!" "It was terrible."
```

Output:

```
Text: I loved this movie! => Sentiment: Positive
Text: It was terrible. => Sentiment: Negative
```

## Modules

### Data Preprocessing (`data/preprocess_data.py`)

- Loads and preprocesses the dataset.
- Tokenizes and pads text sequences.

### Model (`models/model.py`)

- Implements the LSTM model for text classification.
- Configures model compilation and training parameters.

### Training (`models/train.py`)

- Handles training of the model.
- Splits data into training and validation sets.

### Visualization (`utils/visualization.py`)

- Plots training and validation metrics.

### Metrics (`utils/metrics.py`)

- Computes evaluation metrics like accuracy, classification report, and confusion matrix.

### Pipelines

- **Train Pipeline (`pipelines/train_pipeline.py`)**: Encapsulates the end-to-end training process.
- **Inference Pipeline (`pipelines/inference_pipeline.py`)**: Provides functionality for real-time sentiment analysis.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- TensorFlow for deep learning framework.
- IMDB dataset for sentiment analysis.


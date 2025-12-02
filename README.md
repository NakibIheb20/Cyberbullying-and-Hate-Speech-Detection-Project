# Cyberbullying and Hate Speech Detection Project

This project focuses on detecting cyberbullying and hate speech in text data. It utilizes a machine learning model, specifically a fine-tuned RoBERTa-based classifier, to categorize text into three classes: "normal", "offensive", and "hatespeech".

The project is structured into several components:
- **Data**: Contains the raw and cleaned datasets.
- **Notebooks**: Jupyter notebooks for data exploration, model training, and evaluation.
- **Src**: Source code for tasks like loading data into a database.

## Table of Contents
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Data](#data)
  - [Dataset](#dataset)
  - [Loading Data into MongoDB](#loading-data-into-mongodb)
- [Model](#model)
  - [Training](#training)
  - [Inference](#inference)
- [Usage](#usage)

## Project Overview

The core of this project is a natural language processing (NLP) model that can identify and classify harmful online content. The model is built using the `spaCy` library and is based on the RoBERTa architecture, which is known for its strong performance on various NLP tasks. The model is trained on the HateXplain dataset, which provides fine-grained annotations for hate speech.

## Getting Started

Follow these instructions to set up the project environment and get it running on your local machine.

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) should be installed on your system.
- [MongoDB](https://www.mongodb.com/try/download/community) should be installed and running on your local machine.

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create and activate the Conda environment:**
   The `environment.yml` file contains all the necessary dependencies for this project. Use it to create a new Conda environment.
   ```bash
   conda env create -f environment.yml
   conda activate harcelement-nlp
   ```

## Data

### Dataset

The dataset used in this project is **HateXplain**, which can be found in the `data/` directory. The original dataset is `hateXplain.csv`, and the cleaned version used for training is `hateXplain_cleaned.csv`.

### Loading Data into MongoDB

The project includes a script to load the cleaned dataset into a MongoDB database. This is useful for managing the data and making it accessible to other parts of the application (e.g., a web API).

To load the data, run the following command from the root directory of the project:
```bash
python src/load_to_mongo.py
```
This will load the data from `data/hateXplain_cleaned.csv` into a database named `CyberBullying` and a collection named `messages`.

## Model

The machine learning model is a text classifier built with `spaCy` and `transformers`. It is a fine-tuned RoBERTa model that has been trained to distinguish between "normal", "offensive", and "hatespeech" content.

### Training

The model was trained using the `notebooks/Model_cyber_bullying_fine_tunnig_model.ipynb` notebook. This notebook covers the following steps:
1.  **Data Preparation**: Loading the cleaned dataset and splitting it into training and development sets.
2.  **Model Configuration**: Defining the model architecture and training parameters in a `config.cfg` file for `spaCy`.
3.  **Model Training**: Running the `spaCy` training process to fine-tune the RoBERTa model on the custom dataset.
4.  **Evaluation**: Visualizing the training loss and performance metrics.

The trained model is saved and can be used for inference.

### Inference

To use the trained model to make predictions on new text, you can load the model from the output directory (`output_roberta_textcat/model-best`) and use it to classify text.

Here is an example of how to perform inference:
```python
import spacy

# Load the trained model
nlp = spacy.load("output_roberta_textcat/model-best")

# Example text
text = "This is an example of a hateful comment."

# Process the text with the model
doc = nlp(text)

# Get the predicted label and confidence score
predicted_label = max(doc.cats, key=doc.cats.get)
confidence = doc.cats[predicted_label]

print(f"Text: {text}")
print(f"Predicted Label: {predicted_label}")
print(f"Confidence: {confidence}")
```

## Usage

The primary use of this project is to provide a tool for detecting and filtering harmful content. The trained model can be integrated into various applications, such as:
- A web service with a REST API for text classification.
- A content moderation system for online platforms.
- A research tool for analyzing online hate speech.

The project also includes a `fastapi` dependency, suggesting that it can be extended to include a web API for serving the model.

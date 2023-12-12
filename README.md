# README for Document Question Answering Pipeline

This README provides an overview of a Python codebase designed for document-based question answering using deep learning models from the Hugging Face `transformers` and `datasets` libraries.

## Installation

Before running the scripts, ensure that the necessary packages are installed. Run the following commands in your Python environment:

```bash
!pip install transformers
!pip install datasets
!pip install evaluate
!pip install json
!pip install csv
!pip install pandas
```

## Import Libraries

The script begins by importing necessary Python libraries:

```python
import pandas as pd
import json
import csv
import os
```

## Load and Process Dataset

The dataset used is `SantiagoPG/doc_qa`, loaded using Hugging Face's `datasets` library. It is converted into a pandas DataFrame for ease of manipulation:

```python
from datasets import load_dataset

dataset = load_dataset("SantiagoPG/doc_qa")

dataset.set_format(type='pandas')
dataset = dataset['train'][:]
```

The dataset undergoes various preprocessing steps, including cleaning and formatting columns, and removing unnecessary columns.

## Data Analysis

The script includes basic data analysis operations, such as calculating dataset shape, data types, missing values, and unique counts. There's also a visualization section where a histogram and word cloud are generated to analyze question lengths and frequencies.

## Text Cleaning

A custom function `clean_text` is used to clean the `doc_text` field in the dataset, removing URLs and standardizing whitespace.

## Model Preparation

Two models, `AutoTokenizer` and `AutoModelForDocumentQuestionAnswering`, are loaded from the Hugging Face library. The script checks for CUDA availability for GPU acceleration.

## Data Tokenization

A custom function `tokenize_and_format` is defined to tokenize the dataset, preparing it for input into the neural network.

## Dataset and DataLoader

A custom PyTorch `Dataset` class `QADataset` is defined to handle the tokenized data. A `DataLoader` is then created for batch processing.

## Training

The training loop includes forward and backward passes, loss calculation, and optimization steps. It uses the AdamW optimizer and tracks the loss across epochs.

## Inference Function

An `answer_question` function is provided to make predictions on new data, tokenizing the input question and context, and decoding the model's output into an answer string.


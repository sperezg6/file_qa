import json
import os
import pandas as pd
import concurrent.futures
import datasets
from datasets import load_dataset
from pathlib import Path
import json


dataset = load_dataset("SantiagoPG/doc_qa")
# Convert the 'train' split to a DataFrame
dataset.set_format(type='pandas')
dataset = dataset['train'][:]

# Display the first few rows of the DataFrame
print(dataset.head())

dataset = dataset.drop(columns=['data_split'])

# Function to clean the data
def clean_column(data):
    # Check if the data is a string that starts with '[' and ends with ']'
    if isinstance(data, str) and data.startswith('[') and data.endswith(']'):
        # Remove the brackets and extra quotes
        return data[1:-1].replace("'", "")
    return data

# Apply this function to your DataFrame columns
dataset['page_ids'] = dataset['page_ids'].apply(clean_column)
dataset['answers'] = dataset['answers'].apply(clean_column)
print(dataset.head())

def extract_text(ocr_output):
    text_content = []
    for item in ocr_output.get('LINE', []):
        text = item.get('Text', '')
        text_content.append(text)
    return ' '.join(text_content)



def process_ocr(file_path):
    try:
        with open(file_path, 'r') as file:
            ocr_data = json.load(file)
        return extract_text(ocr_data)
    except json.decoder.JSONDecodeError as e:
        print(f"Error processing file: {file_path}")
        print(e)
        return ""


# Assuming OCR files are named like 'ffbf0023_p0.json', 'ffbf0023_p1.json', etc.
ocr_files = list(Path('ocr').glob('*.json'))

# Process each OCR file
with concurrent.futures.ThreadPoolExecutor() as executor:
    page_texts = {file.stem: text for file, text in zip(ocr_files, executor.map(process_ocr, ocr_files))}

# Function to concatenate texts for a given list of page IDs
def get_combined_text(page_ids):
    page_texts_combined = [page_texts.get(page_id, '') for page_id in page_ids.split(', ')]
    return ' '.join(page_texts_combined)

# Update the dataset with combined page texts
dataset['doc_text'] = dataset['page_ids'].apply(get_combined_text)
print(dataset['doc_text'].head())

dataset.to_csv('dataset.csv', index=False)
print(dataset.head())
print("done")

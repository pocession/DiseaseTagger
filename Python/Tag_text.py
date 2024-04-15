import argparse
import os
import csv
from transformers import pipeline

def parse_arguments():
    parser = argparse.ArgumentParser(description="Perform NER on text using a specified model and save the output as CSV.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory where the model is stored.")
    parser.add_argument("--text", type=str, required=True, help="Text to perform NER on.")
    parser.add_argument("--output_file", type=str, default="ner_output.csv", help="Output CSV file to save the results.")
    return parser.parse_args()

def perform_ner(model_dir, text):
    token_classifier = pipeline("token-classification", model=model_dir, aggregation_strategy="simple")
    return token_classifier(text)

def save_output_to_csv(output, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["entity_group", "score", "word", "start", "end"])
        for item in output:
            writer.writerow([item['entity_group'], item['score'], item['word'], item['start'], item['end']])

def main():
    """
    Main function to run the NER pipeline.

    Steps:
    1. Parse command line arguments to get the model directory, input text, and output file path.
    2. Perform Named Entity Recognition (NER) on the provided text using a model specified by the model directory.
    3. Save the NER output in a CSV file.

    Parameters:
    - model_dir (str): Path to the directory containing the pretrained model.
    - text (str): The text on which NER should be performed.
    - output_file (str): Path where the CSV file containing the NER results will be saved.

    Outputs:
    - A CSV file named as specified by `output_file`, containing the NER results with columns for entity group, score, word, start, and end indices.
    
    Examples:
    python Tag_text.py --model_dir "../models/biobert-v1.1-20240414" --text "your_text" --output_file "../results/ner_result.csv"
    """
    args = parse_arguments()
    model_dir = os.path.abspath(args.model_dir)
    output = perform_ner(model_dir, args.text)
    save_output_to_csv(output, args.output_file)
    print(f"NER output saved to {args.output_file}")

if __name__ == "__main__":
    main()

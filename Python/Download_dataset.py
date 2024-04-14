import argparse
import os
from datasets import load_dataset

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
    argparse.Namespace: The namespace containing the arguments.
    """
    parser = argparse.ArgumentParser(description="Download a dataset from Hugging Face and save it locally.")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to download. Example: 'ncbi_disease'")
    parser.add_argument("--save_directory", type=str, default="../datasets", help="Directory to save the dataset. Example: './datasets'")
    return parser.parse_args()

def main(args):
    """
    Main function to download and save a dataset from Hugging Face.

    Parameters:
    args (Namespace): Command line arguments containing dataset_name and save_directory.

    The function performs the following:
    - Downloads the dataset specified by dataset_name from the Hugging Face datasets library.
    - Prints the first sample of the 'train' split to check its content.
    - Saves the entire dataset to a specified directory.

    Example:
    python Download_dataset.py ncbi_disease --save_directory ../datasets
    """
    # Load the dataset
    print(f"Downloading '{args.dataset_name}' dataset...")
    dataset = load_dataset(args.dataset_name)

    # Check the first sample
    print("The first iten in the train data: ")
    print("Sample data from 'train' split:")
    print("Tokens:", dataset["train"][0]["tokens"])
    print("NER Tags:", dataset["train"][0]["ner_tags"])

    # Save the dataset to disk
    save_path = os.path.join(args.save_directory, args.dataset_name)
    print(f"Saving dataset to '{save_path}'...")
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)
    print("Dataset saved successfully.")

if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()
    
    # Execute main function
    main(args)

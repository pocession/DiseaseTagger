import argparse
import os
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
    argparse.Namespace: The namespace containing the arguments.
    """
    parser = argparse.ArgumentParser(description="Download and save a Hugging Face model, and visualize weights.")
    parser.add_argument("model_id", type=str, help="Identifier for the model on Hugging Face. Example: 'dmis-lab/biobert-v1.1'")
    parser.add_argument("--save_directory", type=str, default="./models", help="Directory to save the model and tokenizer. Example: './models'")
    parser.add_argument("--show_parameters", type=bool, default=False, help="Show model parameters. Example: True")
    parser.add_argument("--show_layer0", type=bool, default=False, help="Show the weights for the first layer. Example: False")
    return parser.parse_args()

def main(args):
    """
    Main function to download, save a model, and visualize its weights.

    Parameters:
    args (Namespace): Command line arguments containing model_id and save_directory.

    The function performs the following:
    - Downloads the model and tokenizer specified by model_id from the Hugging Face Hub.
    - Saves the model and tokenizer in the specified directory.
    - Prints the model's parameter names and sizes.
    - Visualizes the weight distribution of the model's first encoder layer's query weights.

    An example:
    python Download_model.py dmis-lab/biobert-v1.1 --save_directory ../models --show_parameter False --show_layer0 False
    """
    # Load model and tokenizer from Hugging Face Hub
    print(f"Downloading model and tokenizer for '{args.model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id)

    # Ensure the save directory exists
    os.makedirs(args.save_directory, exist_ok=True)

    # Save the model and tokenizer
    model.save_pretrained(args.save_directory)
    tokenizer.save_pretrained(args.save_directory)
    print(f"Saving model and tokenizer to '{args.save_directory}'...")

    # Print the model parameters
    if args.show_parameters:
        print("Model parameters:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.size()}")

    if not args.show_layer0:

        # Fetch weights from the first attention layer's query
        encoder_layer_0_weights = model.encoder.layer[0].attention.self.query.weight.data

        # Flatten the weights for visualization
        weights_flat = encoder_layer_0_weights.flatten().numpy()

        # Plot the distribution of weights
        plt.figure(figsize=(10, 6))
        sns.histplot(weights_flat, bins=50, kde=True)
        plt.title('Distribution of Weights in Encoder Layer 0 - Query')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.show()

if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()
    
    # Execute main function
    main(args)

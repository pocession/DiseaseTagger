import argparse
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_from_disk
import evaluate
import numpy as np
import csv

def parse_arguments():
    """
    Parse command line arguments to the script.
    """
    parser = argparse.ArgumentParser(description="Train a BioBERT model on the NCBI disease dataset.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory where the pretrained model is stored.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory where the dataset is stored.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the trained model will be saved.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs for training.")
    return parser.parse_args()

def adjust_path(path):
    """
    Prepend the working directory to the provided path if it is not an absolute path.
    """
    if os.path.isabs(path):
        return path
    return os.path.join(os.getcwd(), path)

def align_labels_with_tokens(labels, word_ids):
    """
    Align token labels with corresponding word IDs, especially for subwords.
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels

def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenize and align labels from the dataset using the tokenizer.
    This involves handling the splitting of words into subwords/subtokens.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def compute_metrics(eval_preds, label_names):
    """
    Compute precision, recall, F1-score, and accuracy for the model's predictions.
    """
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    metric = evaluate.load("seqeval")
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

def main():
    """
    Main function to execute the model training process.

    Parameters:
    --model_dir (str): Path to the directory containing the pretrained BioBERT model.
    --dataset_dir (str): Path to the directory containing the NCBI disease dataset.
    --output_dir (str): Path to the directory where the trained model will be saved.
    --num_epochs (int): The number of training epochs to perform.

    This script sets up a token classification task with BioBERT, processes the dataset,
    aligns tokens with their respective labels, sets up training arguments, and performs training.
    Example:
    python Retrain_model.py --model_dir ../models/biobert-v1.1-20240414 --dataset_dir ../datasets/ncbi_disease --output_dir ../models/biobert-v1.1-20240415 --num_epochs 3
    """
    
    args = parse_arguments()

    # Adjust path arguments
    model_dir = adjust_path(args.model_dir)
    dataset_dir = adjust_path(args.dataset_dir)
    output_dir = adjust_path(args.output_dir)

    # Initialize tokenizer and model from pretrained versions
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    # Load and process the dataset
    dataset = load_from_disk(dataset_dir)
    tokenized_datasets = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # Prepare for training: setup data collator, training arguments, and trainer
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    label_names = ["O", "B-Disease", "I-Disease"]  # Modify as needed
    training_args = TrainingArguments(
        args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, label_names),
        tokenizer=tokenizer,
    )
    # Execute training
    trainer.train()

    # Save the trained model
    model.save_pretrained(output_dir)

    # Evaluate the model and save the results
    eval_results = trainer.evaluate()

    # Save results to CSV
    results_file = os.path.join(output_dir, 'evaluation_results.csv')
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the headers
        writer.writerow(['Metric', 'Value'])
        # Write the metric values
        for key, value in eval_results.items():
            writer.writerow([key, value])

if __name__ == "__main__":
    main()

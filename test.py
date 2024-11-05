import numpy as np
import os
import argparse  # Added for command-line arguments
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from multinet.data import utils  
from multinet.model.multinet import MultiNet  

def load_test_data(cache_file):
    """Load the cached test dataset."""
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file {cache_file} does not exist.")
    
    test_images, test_audio_features, test_labels = utils.load_cached_dataset(cache_file)
    return test_images, test_audio_features, test_labels

def plot_confusion_matrix(true_labels, predicted_labels, class_labels, save_path):
    """Plot and save the confusion matrix as a heatmap."""
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.close()

def save_classification_report(true_labels, predicted_labels, class_labels, save_path):
    """Save the classification report (F1, precision, recall) to a text file in tabular format."""
    report = classification_report(true_labels, predicted_labels, target_names=class_labels, output_dict=True)
    
    # Open the file to write the report in tabular format
    with open(os.path.join(save_path, "classification_report.txt"), "w") as f:
        # Write the header row
        f.write(f"{'Class':<15}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}\n")
        f.write(f"{'-'*51}\n")  # Add a separator
        
        # Write each class's metrics
        for class_name in class_labels:
            metrics = report[class_name]
            f.write(f"{class_name:<15}{metrics['precision']:<12.4f}{metrics['recall']:<12.4f}{metrics['f1-score']:<12.4f}\n")
        
        # Add overall metrics (accuracy, macro avg, weighted avg)
        f.write(f"{'-'*51}\n")
        f.write(f"{'Accuracy':<15}{report['accuracy']:<12.4f}\n")
        
        for avg_type in ['macro avg', 'weighted avg']:
            metrics = report[avg_type]
            f.write(f"{avg_type:<15}{metrics['precision']:<12.4f}{metrics['recall']:<12.4f}{metrics['f1-score']:<12.4f}\n")


def evaluate_model(test_images, test_audio_features, test_labels, model, class_labels, save_path):
    """Evaluate the model and calculate class-wise accuracy, precision, recall, and F1-score."""
    # Get model predictions
    predictions = model.predict([test_audio_features, test_images])
    
    # Convert predictions to class indices (argmax)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate overall accuracy
    overall_accuracy = np.mean(predicted_classes == test_labels)
    print(f"Overall accuracy: {overall_accuracy:.4f}")

    # Save classification metrics
    save_classification_report(test_labels, predicted_classes, class_labels, save_path)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(test_labels, predicted_classes, class_labels, save_path)

def main(test_cache_file, model_weights_path, save_dir):
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load test dataset from the cache file
    test_images, test_audio_features, test_labels = load_test_data(test_cache_file)

    # Load your trained model
    model = MultiNet(
        timnet_input_shape=(215, 39),  # Adjust input shapes as per your data
        class_labels=("angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"),
        p_att_lite_input_shape=(224, 224, 3),
        dropout_rate=0.5
    )
    
    # Compile the model (with the same configuration as used during training)
    model.compile_model(learning_rate=0.001)
    
    # Load the trained model weights
    model.combined_model.load_weights(model_weights_path)

    # Evaluate the model and save plots/reports
    evaluate_model(test_images, test_audio_features, test_labels, model.combined_model, model.class_labels, save_dir)

if __name__ == "__main__":
    # Define command-line argument parsing
    parser = argparse.ArgumentParser(description="Test and evaluate MultiNet model.")
    
    parser.add_argument('--test_cache_file', type=str, required=True, help='Path to the cached test dataset file (e.g., test_v1.npz)')
    parser.add_argument('--model_weights_path', type=str, required=True, help='Path to the trained model weights file (e.g., model_weights.h5)')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory where the evaluation results will be saved (plots and report)')

    # Parse arguments from the command line
    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(args.test_cache_file, args.model_weights_path, args.save_dir)

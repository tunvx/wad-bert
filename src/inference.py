import argparse
from time import time

import torch
import torch.nn.functional as F
from model import WebAttackBertClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_class_map():
    return {'anomaly': 0, 'benign': 1}

def get_class_names():
    return ['anomaly', 'benign']

def get_prediction(model, request: str, device):
    """
    Perform inference on the given input string.

    Args:
        model: The trained model.
        request (str): The input string to classify.
        device: The device on which inference is performed.

    Returns:
        dict: The predicted class name and confidence score.
    """
    model.eval()
    with torch.no_grad():
        outputs = model([request])

        # Handle cases where model returns logits or plain tensor
        logits = outputs.logits if hasattr(outputs, "logits") else outputs  
        probabilities = F.softmax(logits, dim=1)  # Apply softmax to logits
        confidence, pred = torch.max(probabilities, dim=1)
        class_idx = pred.item()
        confidence_score = confidence.item()
        return {
            'class_idx': class_idx,
            'class_name': get_class_names()[class_idx],
            'confidence_score': confidence_score
        }

def load_model(model_path):
    """
    Load a trained WebAttackBertClassifier model from a checkpoint.

    Args:
        model_path (str): Path to the model checkpoint.

    Returns:
        WebAttackBertClassifier: The loaded model.
    """
    net = WebAttackBertClassifier.load_from_checkpoint(model_path)
    
    net.eval()
    return net

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=str, required=True, help="The input string to classify.")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to use for inference (e.g., 'cuda:0' or 'cpu').")
    args = parser.parse_args()

    model = load_model(args.model).to(args.device)
    request = args.request

    # Perform inference
    start = time()
    prediction = get_prediction(model, request, args.device)
    for _ in range(1000):
        get_prediction(model, request, args.device)
    duration = time() - start
    
    print("\nOUTPUT:")
    print(prediction)
    print(f"Inference time: {duration:.4f} sec(s) per 100 predictions")

if __name__ == "__main__":
    main()

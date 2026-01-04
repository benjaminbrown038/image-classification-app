import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

def load_model():
    """
    Loads a pretrained ResNet-18 model and preprocessing pipeline.
    Swap this function to load custom models or datasets.
    """
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    preprocess = weights.transforms()
    class_names = weights.meta["categories"]

    return model, preprocess, class_names


@torch.no_grad()
def predict(image, model, preprocess, class_names, topk=5):
    """
    Runs inference on a single image and returns top-k predictions.
    """
    input_tensor = preprocess(image).unsqueeze(0)

    outputs = model(input_tensor)
    probabilities = torch.softmax(outputs, dim=1)[0]

    top_probs, top_idxs = probabilities.topk(topk)

    results = [
        (class_names[idx], prob.item())
        for idx, prob in zip(top_idxs, top_probs)
    ]

    return results

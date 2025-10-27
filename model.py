# model.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load pretrained ResNet18
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    return model

# Define image preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Predict top 5 ImageNet classes
def predict(model, image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)
        _, indices = torch.sort(outputs, descending=True)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    from torchvision.models import ResNet18_Weights
    labels = ResNet18_Weights.DEFAULT.meta["categories"]
    top5 = [(labels[idx], float(probs[idx])) for idx in indices[0][:5]]
    return top5

import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

for filename in os.listdir(image_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        img_path = os.path.join(image_folder, filename)
        image = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0).cpu().numpy()



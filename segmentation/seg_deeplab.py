import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model = model.to(device)
model.eval()

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
        mask = np.zeros_like(output_predictions, dtype=np.uint8)
        mask[output_predictions > 0] = 255
        mask_image = Image.fromarray(mask)
        mask_image.save(os.path.join(output_folder, f'mask_{filename}'))
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(image)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title('Segmentation Mask')
        plt.imshow(mask_image, cmap='gray')
        plt.axis('off')

        plt.show()
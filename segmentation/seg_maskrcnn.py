import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model = model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

for filename in os.listdir(image_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        img_path = os.path.join(image_folder, filename)
        image = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_batch)
        masks = output[0]['masks'].cpu().numpy()

        if masks.shape[0] > 0:
            mask_image = np.max(masks, axis=0) > 0.5
            mask_image = (mask_image * 255).astype(np.uint8)
            mask_image = np.squeeze(mask_image)
        else:
            mask_image = np.zeros((64, 64), dtype=np.uint8)
        mask_image_pil = Image.fromarray(mask_image)
        mask_image_pil.save(os.path.join(output_folder, f'mask_{filename}'))
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(image)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title('Segmentation Mask')
        plt.imshow(mask_image_pil, cmap='gray')
        plt.axis('off')

        plt.show()
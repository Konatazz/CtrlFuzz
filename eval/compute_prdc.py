import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

def compute_pairwise_distance(data_x, data_y=None):
    if data_y is None:
        data_y = data_x
    dists = torch.cdist(data_x, data_y, p=2)
    return dists

def get_kth_value(unsorted, k):
    return torch.topk(unsorted, k, largest=False)[0][:, -1]

def compute_nearest_neighbour_distances(input_features, nearest_k):
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, nearest_k + 1)
    return radii

def compute_prdc(real_features, fake_features, nearest_k):
    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(real_features, fake_features)
    precision = (distance_real_fake < real_nearest_neighbour_distances.unsqueeze(1)).any(dim=0).float().mean().item()
    recall = (distance_real_fake < fake_nearest_neighbour_distances.unsqueeze(0)).any(dim=1).float().mean().item()
    density = (1. / float(nearest_k)) * (distance_real_fake < real_nearest_neighbour_distances.unsqueeze(1)).sum(
        dim=0).float().mean().item()
    density = max(0, min(density, 1.0))
    coverage = (distance_real_fake.min(dim=1).values < real_nearest_neighbour_distances).float().mean().item()
    mean_distance_real_fake = distance_real_fake.mean().item()
    mean_distance_real = real_nearest_neighbour_distances.mean().item()
    mean_distance_fake = fake_nearest_neighbour_distances.mean().item()
    std_dev_real = real_nearest_neighbour_distances.std().item()
    std_dev_fake = fake_nearest_neighbour_distances.std().item()

    return {
        'precision': precision,
        'recall': recall,
        'density': density,
        'coverage': coverage,
        'mean_distance_real_fake': mean_distance_real_fake,
        'mean_distance_real': mean_distance_real,
        'mean_distance_fake': mean_distance_fake,
        'std_dev_real': std_dev_real,
        'std_dev_fake': std_dev_fake
    }

model = models.inception_v3(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_paths):
    features = []
    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)

            feature = model(img_tensor)
            features.append(feature.squeeze(0).numpy())
    return torch.tensor(np.vstack(features))

real_folder = ''
fake_folder = ''

real_images = sorted(
    [os.path.join(real_folder, f) for f in os.listdir(real_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
fake_images = sorted(
    [os.path.join(fake_folder, f) for f in os.listdir(fake_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

real_features = extract_features(real_images)
fake_features = extract_features(fake_images)

nearest_k = 5
prdc_results = compute_prdc(real_features, fake_features, nearest_k)

# print(f"Precision: {prdc_results['precision']:.4f}")
# print(f"Recall: {prdc_results['recall']:.4f}")
# print(f"Density: {prdc_results['density']:.4f}")
# print(f"Coverage: {prdc_results['coverage']:.4f}")
# print(f"Mean Distance (Real-Fake): {prdc_results['mean_distance_real_fake']:.4f}")
# print(f"Mean Distance (Real): {prdc_results['mean_distance_real']:.4f}")
# print(f"Mean Distance (Fake): {prdc_results['mean_distance_fake']:.4f}")
# print(f"Std Dev (Real): {prdc_results['std_dev_real']:.4f}")
# print(f"Std Dev (Fake): {prdc_results['std_dev_fake']:.4f}")
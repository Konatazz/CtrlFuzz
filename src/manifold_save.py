import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean
import pickle
import joblib

def find_intersection_with_hull(point, center, hull, class_points):
    v = point - center
    v = v / np.linalg.norm(v)
    min_t = float('inf')
    intersection_point = None

    for simplex in hull.simplices:
        p1 = class_points[simplex[0]]
        p2 = class_points[simplex[1]]

        t, intersects = line_intersection(center, v, p1, p2)
        if intersects and t < min_t:
            min_t = t
            intersection_point = center + t * v

    if intersection_point is not None:
        distance_to_boundary = np.linalg.norm(intersection_point - center)
        return distance_to_boundary
    else:
        return None

def line_intersection(center, direction, p1, p2):
    edge = p2 - p1
    direction = np.atleast_2d(direction).T
    edge = np.atleast_2d(edge).T
    matrix = np.hstack([direction, -edge])

    if np.linalg.det(matrix) == 0:
        return None, False

    b = p1 - center
    solution = np.linalg.solve(matrix, b)
    t = solution[0]
    u = solution[1]

    if t > 0 and 0 <= u <= 1:
        return t, True
    else:
        return None, False


def calculate_distance_ratio(point, class_value, category_centers, category_convex_hulls, X_lda, y_encoded):
    center = category_centers[class_value]
    hull = category_convex_hulls[class_value]
    class_points = X_lda[y_encoded == class_value]
    boundary_distance = find_intersection_with_hull(point, center, hull, class_points)
    if boundary_distance is None:
        return None
    point_distance = np.linalg.norm(point - center)
    ratio = point_distance / boundary_distance if boundary_distance > 0 else 0
    return ratio



def save_semantic_space(folder_path, category_centers, category_convex_hulls, lda_model, label_encoder, X_lda, y_encoded):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    semantic_data = {
        'category_centers': category_centers,
        'category_convex_hulls': {class_value: X_lda[y_encoded == class_value].tolist() for class_value in np.unique(y_encoded)},
        'X_lda': X_lda.tolist(),
        'y_encoded': y_encoded.tolist()
    }
    with open(os.path.join(folder_path, 'semantic_data.pkl'), 'wb') as file:
        pickle.dump(semantic_data, file)
    joblib.dump(lda_model, os.path.join(folder_path, 'lda_model.pkl'))
    joblib.dump(label_encoder, os.path.join(folder_path, 'label_encoder.pkl'))

def load_semantic_space(folder_path):
    with open(os.path.join(folder_path, 'semantic_data.pkl'), 'rb') as file:
        semantic_data = pickle.load(file)
    category_centers = semantic_data['category_centers']
    category_convex_hulls = {}
    for class_value, points in semantic_data['category_convex_hulls'].items():
        class_points = np.array(points)
        category_convex_hulls[class_value] = ConvexHull(class_points)
    X_lda = np.array(semantic_data['X_lda'])
    y_encoded = np.array(semantic_data['y_encoded'])
    lda_model = joblib.load(os.path.join(folder_path, 'lda_model.pkl'))
    label_encoder = joblib.load(os.path.join(folder_path, 'label_encoder.pkl'))

    return category_centers, category_convex_hulls, lda_model, label_encoder, X_lda, y_encoded


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model.to(device)

    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for img_file in tqdm(os.listdir(class_path), desc=f"Processing {class_dir}"):
                img_path = os.path.join(class_path, img_file)
                img = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    features = model(img_tensor).cpu().numpy().flatten()
                X.append(features)
                y.append(class_dir)

    X = np.array(X)
    y = np.array(y)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X, y_encoded)

    category_centers = {}
    category_convex_hulls = {}

    plt.figure(figsize=(10, 8))

    for class_value in np.unique(y_encoded):
        class_points = X_lda[y_encoded == class_value]

        plt.scatter(class_points[:, 0], class_points[:, 1],
                    label=label_encoder.inverse_transform([class_value])[0], alpha=0.6)

        center = np.mean(class_points, axis=0)
        category_centers[class_value] = center

        if len(class_points) >= 3:
            hull = ConvexHull(class_points)
            category_convex_hulls[class_value] = hull
            for simplex in hull.simplices:
                plt.plot(class_points[simplex, 0], class_points[simplex, 1], 'k-', linewidth=1)

    for class_value, center in category_centers.items():
        plt.scatter(center[0], center[1], marker='x', color='red', s=100)


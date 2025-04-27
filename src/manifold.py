# manifold.py

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def plot_with_circles(features, labels, class_names, ax, alpha=0.2):
    for class_idx, class_name in enumerate(class_names):
        class_features = features[labels == class_idx]
        centroid = np.mean(class_features, axis=0)
        radius = np.mean(np.linalg.norm(class_features - centroid, axis=1))
        circle = plt.Circle(centroid, radius, color=plt.cm.Set1(class_idx / len(class_names)), alpha=alpha)
        ax.add_patch(circle)
        ax.scatter(class_features[:, 0], class_features[:, 1], label=class_name, alpha=0.5, edgecolors='k')

def plot_with_convex_hulls(features, labels, class_names, ax, alpha=0.2):
    for class_idx, class_name in enumerate(class_names):
        class_features = features[labels == class_idx]
        if len(class_features) < 3:
            ax.scatter(class_features[:, 0], class_features[:, 1], label=class_name, alpha=0.5, edgecolors='k')
            continue
        hull = ConvexHull(class_features)
        hull_points = class_features[hull.vertices]
        ax.fill(hull_points[:, 0], hull_points[:, 1],
                alpha=alpha, label=f'{class_name} Hull',
                color=plt.cm.Set1(class_idx / len(class_names)))
        ax.scatter(class_features[:, 0], class_features[:, 1], label=class_name, alpha=0.5, edgecolors='k')

def plot_decision_boundaries(X, y, class_names, ax, clf, mesh_step=0.02):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step),
                         np.arange(y_min, y_max, mesh_step))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.Set1)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k', alpha=0.6)
    handles, _ = scatter.legend_elements()
    ax.legend(handles, class_names)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title('')

def plot_with_kmeans_boundaries(features, labels, class_names, ax, n_clusters, mesh_step=0.02):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    y_kmeans = kmeans.predict(features)
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step),
                         np.arange(y_min, y_max, mesh_step))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.Set1)
    scatter = ax.scatter(features[:, 0], features[:, 1], c=y_kmeans, cmap=plt.cm.Set1, edgecolor='k', alpha=0.6)
    handles, _ = scatter.legend_elements()


def main():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    features = []
    labels = []

    with torch.no_grad():
        for inputs, label in data_loader:
            inputs = inputs.to(device)
            output = model(inputs)
            output = output.view(output.size(0), -1)
            features.append(output.cpu().numpy())
            labels.append(label.numpy())

    lda = LinearDiscriminantAnalysis(n_components=2)
    features_lda = lda.fit_transform(features, labels)

if __name__ == '__main__':
    main()

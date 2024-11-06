import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score, f1_score
from sklearn.metrics.pairwise import pairwise_distances
import os
from train import DCNN


# Step 1: Load your pre-trained DCNN model
net = DCNN()  # Ensure DCNN is defined elsewhere
net.load_parameters("dcnn_trained.params")

# Step 2: Prepare the LFW dataset
lfw_data_path = "ee798r/faces_webface_112x112/lfw"  # Update with your LFW directory

# Create an ImageIter for the LFW dataset
lfw_iter = mx.image.ImageIter(
    batch_size=32,  # Adjust batch size as needed
    data_shape=(3, 100, 100),  # Image shape for LFW (channels, height, width)
    path_imglist=None,  # Not using an image list, direct directory loading
    path_root=lfw_data_path,  # Path to LFW directory
    shuffle=False,  # No need to shuffle for feature extraction
    rand_crop=False,  # No random cropping
    rand_mirror=False  # No random mirroring
)

# Step 3: Feature Extraction Function
def extract_features(data_iter, net):
    features_list = []
    labels_list = []
    num_batch = 0
    data_iter.reset()  # Reset the iterator
    
    for batch in tqdm(data_iter, desc="Extracting Features"):
        # Get data and labels, move to CPU for processing
        data = batch.data[0].as_in_context(mx.cpu())
        label = batch.label[0].as_in_context(mx.cpu())
        
        # Forward pass through the network to get the features
        output = net(data)
        features = output.asnumpy().reshape(output.shape[0], -1)
        
        # Store the features and labels
        features_list.append(features)
        labels_list.append(label.asnumpy())
        
        # Stop after processing 2 batches (can remove this if processing all batches)
        num_batch += 1
        if num_batch >= 2:
            break
    
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    return features, labels

# Step 4: Extract features from the LFW dataset using your pre-trained network
features, labels = extract_features(lfw_iter, net)

# Use a subset for clustering
subset_size = 1000  # Adjust subset size based on your dataset size
features_subset = features[:subset_size]
labels_subset = labels[:subset_size]

print(f"Extracted features shape: {features_subset.shape}")
print(f"Labels shape: {labels_subset.shape}")

# Step 5: Deep Density Clustering (DDC)

# 5.1: Compute the distance matrix
print("Computing pairwise distances...")
distance_matrix = pairwise_distances(features_subset, metric='euclidean')

# 5.2: Define epsilon for neighborhood construction
epsilon = 0.23  # Adjust based on your data distribution

# 5.3: Construct neighborhoods
print("Constructing neighborhoods...")
neighborhoods = [np.where(distance_matrix[i] <= epsilon)[0] for i in range(len(features_subset))]

# 5.4: Train One-Class SVM (SVDD) models
svdd_models = []
gamma = 1.0 / features_subset.shape[1]  # gamma='auto'

print("Training One-Class SVM (SVDD) models...")
for i, neighbors in enumerate(tqdm(neighborhoods, desc="Training SVDD Models")):
    if len(neighbors) < 1:
        svdd_models.append(None)
        continue
    svm_features = features_subset[neighbors]
    model = OneClassSVM(kernel='rbf', gamma=gamma, nu=0.5)
    model.fit(svm_features)
    svdd_models.append(model)

# 5.5: Compute similarity matrix
print("Computing similarity matrix...")
similarity_matrix = np.zeros((len(features_subset), len(features_subset)))

for i in tqdm(range(len(features_subset)), desc="Computing Similarities"):
    for j in range(i + 1, len(features_subset)):
        model_i = svdd_models[i]
        model_j = svdd_models[j]
        if model_i is None or model_j is None:
            similarity = 0
        else:
            score_ij = model_i.decision_function([features_subset[j]])[0]
            score_ji = model_j.decision_function([features_subset[i]])[0]
            similarity = (score_ij + score_ji) / 2.0
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity

# 5.6: Convert similarities to distances
distance_matrix_ddc = -similarity_matrix

# 5.7: Perform Agglomerative Clustering
distance_threshold_percentile = 50  # Adjust as needed
distance_threshold = np.percentile(distance_matrix_ddc, distance_threshold_percentile)
print(f"Clustering with distance threshold (percentile {distance_threshold_percentile}): {distance_threshold}")

clustering = AgglomerativeClustering(
    affinity='precomputed',
    linkage='average',
    distance_threshold=distance_threshold,
    n_clusters=None
)
clustering.fit(distance_matrix_ddc)

# Step 6: Evaluate clustering performance using NMI and F-measure
print("Evaluating clustering performance...")

# Normalized Mutual Information (NMI)
nmi = normalized_mutual_info_score(labels_subset, clustering.labels_)
print(f"NMI: {nmi:.4f}")

# F-measure computation
f_measure = f1_score(labels_subset, clustering.labels_, average='macro')
print(f"F-measure (Macro F1): {f_measure:.4f}")

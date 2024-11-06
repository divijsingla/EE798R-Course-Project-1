from mxnet.gluon.data.vision import ImageRecordDataset
from tqdm import tqdm
import numpy as np
import mxnet as mx
from mxnet import gluon, nd, recordio
from gluoncv.model_zoo import get_model
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics.pairwise import cosine_distances
from sklearn.svm import OneClassSVM
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from scipy.stats import norm
from train import DCNN

net = DCNN()
net.load_parameters("dcnn_trained.params")

# Feature extraction after training
def extract_features(data_iter, net):
    features_list = []
    labels_list = []
    num_batch = 0
    data_iter.reset()  # Reset the iterator to ensure starting from the beginning
    
    for batch in tqdm(data_iter, desc="Extracting Features"):
        # Get data and labels, and move them to CPU for processing
        data = batch.data[0].as_in_context(mx.cpu())
        label = batch.label[0].as_in_context(mx.cpu())
        
        # Forward pass through the network to get the output (features)
        output = net(data)
        
        # Flatten features and convert to numpy array
        features = output.asnumpy().reshape(output.shape[0], -1)
        
        # Store the features and labels
        features_list.append(features)
        labels_list.append(label.asnumpy())
        
        # Stop after processing 2 batches (can remove this if processing all batches)
        num_batch += 1
        if num_batch >= 2:
            break
    
    # Concatenate features and labels from all batches
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    return features, labels

# Use the same ImageIter as before for feature extraction
data_iter = mx.image.ImageIter(
    batch_size=32,  # You can adjust the batch size
    data_shape=(3, 100, 100),  # Input image shape (channels, height, width)
    path_imgrec="ee798r/faces_webface_112x112/train.rec",  # .rec file for the dataset
    path_imgidx="ee798r/faces_webface_112x112/train.idx",  # .idx file for the dataset
    rand_crop=False,  # No random cropping during feature extraction
    rand_mirror=False  # No random mirroring during feature extraction
)

# Extract features from the dataset using the iterator and your pre-trained network
features, labels = extract_features(data_iter, net)

subset_size = 1000
features_subset = features[:subset_size]
labels_subset = labels[:subset_size]
print(features_subset, labels_subset)

print(f"Extracted features shape: {features_subset.shape}")
print(f"Labels shape: {labels_subset.shape}")

# Deep Density Clustering (DDC) Implementation

# # Step 1: Compute the distance matrix
print("Computing pairwise distances...")
distance_matrix = pairwise_distances(features_subset, metric='euclidean')

epsilon = 0.23 # as used in the research paper

# Step 3: Construct neighborhoods
print("Constructing neighborhoods...")
neighborhoods = [np.where(distance_matrix[i] <= epsilon)[0] for i in range(len(features_subset))]

# Step 4: Train One-Class SVM (SVDD) models using LIBSVM
svdd_models = []
gamma = 1.0 / features_subset.shape[1]  # gamma='auto'

print("Training One-Class SVM (SVDD) models...")
for i, neighbors in enumerate(tqdm(neighborhoods, desc="Training SVDD Models")):
    if len(neighbors) < 1:
        # Handle cases with no neighbors
        svdd_models.append(None)
        continue
    # Prepare data for SVM
    svm_features = features_subset[neighbors]
    
    # Create and fit the One-Class SVM model
    model = OneClassSVM(kernel='rbf', gamma=gamma, nu=0.5)
    model.fit(svm_features)
    svdd_models.append(model)

# Step 5: Compute similarity matrix
print("Computing similarity matrix...")
similarity_matrix = np.zeros((len(features_subset), len(features_subset)))

for i in tqdm(range(len(features_subset)), desc="Computing Similarities"):
    for j in range(i + 1, len(features_subset)):
        model_i = svdd_models[i]
        model_j = svdd_models[j]
        if model_i is None or model_j is None:
            # Assign minimal similarity if any model is None
            similarity = 0
        else:
            # Score of j on model i
            score_ij = model_i.decision_function([features_subset[j]])[0]
            # Score of i on model j
            score_ji = model_j.decision_function([features_subset[i]])[0]
            # Average similarity
            similarity = (score_ij + score_ji) / 2.0
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity

# Step 6: Convert similarities to distances
distance_matrix_ddc = -similarity_matrix

# Step 7: Perform Agglomerative Clustering
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

# Step 8: Evaluate clustering performance using NMI and F-measure
print("Evaluating clustering performance...")

from sklearn.metrics import normalized_mutual_info_score

# Normalized Mutual Information (NMI)
nmi = normalized_mutual_info_score(labels_subset, clustering.labels_)
print(f"NMI: {nmi:.4f}")

# F-measure computation
# To compute F-measure for clustering, we need to map clusters to true labels
# One common approach is to use the best matching between clusters and labels
# Here, we'll compute the F1 score for each label and take the average (macro F1)

from sklearn.metrics import f1_score

f_measure = f1_score(labels_subset, clustering.labels_, average='macro')
print(f"F-measure (Macro F1): {f_measure:.4f}")
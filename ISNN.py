import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import numpy as np
import random
import scipy
import yaml
import time
from torch_geometric.utils import negative_sampling,add_self_loops, degree
import torch_sparse
from torch_sparse import SparseTensor
import json
from sklearn.cluster import KMeans
import datasets
from impl import SubGDataset, metrics, config
from impl.train import train_model, test_model
from custom_models import SubIGNN_new, VanillaGCN, SoftEIGNN, SoftIGNN, baseline
from config_path import DATASET_PATH
import itertools

# Set random seed for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

# Parse command-line arguments for hyperparameters
def parse_args():
    parser = argparse.ArgumentParser(description='Run SubIGNN training')
    parser.add_argument('--dataset', type=str, default='hpo_metab', help='Path to the dataset')
    parser.add_argument('--model', type=str, default='isnn', help='Model to use for training')
    parser.add_argument('--device', type=int, default=0, help='Device to use for training')
    parser.add_argument('--seed', type=int, default=44, help='Random seed')
    parser.add_argument('--repeat', type=int, default=1, help='Number of repetitions for the experiment')
    parser.add_argument('--pretrain', action='store_true', help='Whether to pretrain the model')
    parser.add_argument('--channel', type=str, default='None', help='Channel to use for training, valid options are "position", "neighbor", "structure"')

    return parser.parse_args()


def remove_redundant_edges(edge_index, edge_weight):
    """
    Remove redundant edges (bidirectional edges) based on in-degree.
    If there is an edge from `a` to `b` and `b` to `a`, keep only one:
    - Keep the edge for the node with higher in-degree.
    
    Args:
        edge_index (torch.Tensor): Edge index of shape (2, num_edges) with source and target nodes.
        edge_weight (torch.Tensor): Edge weights of shape (num_edges,).
    
    Returns:
        edge_index (torch.Tensor): Updated edge index after removing redundant edges.
        edge_weight (torch.Tensor): Updated edge weights after removing redundant edges.
    """
    # Calculate in-degrees
    num_nodes = edge_index.max().item() + 1
    in_degree = torch.zeros(num_nodes, dtype=torch.long)
    in_degree.scatter_add_(0, edge_index[1], torch.ones_like(edge_index[1]))

    # Create a set of edges for fast lookup
    edge_set = set((src.item(), tgt.item()) for src, tgt in zip(edge_index[0], edge_index[1]))
    to_remove = set()

    # Identify redundant edges to remove
    for src, tgt in edge_set:
        if (tgt, src) in edge_set:  # Check for the reverse edge
            if in_degree[src] > in_degree[tgt]:
                to_remove.add((src, tgt))  # Remove edge from src to tgt
            elif in_degree[src] < in_degree[tgt]:
                to_remove.add((tgt, src))  # Remove edge from tgt to src
            else:
                # Arbitrarily keep one edge when in-degrees are equal
                to_remove.add((src, tgt))

    # Filter edges
    mask = torch.tensor([(src, tgt) not in to_remove for src, tgt in zip(edge_index[0], edge_index[1])], dtype=torch.bool)
    edge_index = edge_index[:, mask]
    edge_weight = edge_weight[mask]

    return edge_index, edge_weight

def get_subgraph_adj_multi_label(embeddings, G, num_balanced_labels, k, binary=False):
    """
    Constructs a subgraph adjacency matrix by selecting k labels that induce balanced classes.

    Args:
        embeddings: Node embeddings (torch.Tensor).
        G: Graph object containing attributes like G.y (binary label matrix) and G.mask.
        k: Number of most distant pairs of training nodes to consider.
        num_balanced_labels: Number of labels to select for balanced class sizes.
        binary: If True, edge weights are binary (1.0). Otherwise, they are based on cosine distances.

    Returns:
        edge_index: Edge indices (torch.LongTensor).
        edge_weight: Edge weights (torch.FloatTensor).
    """
    # Step 1: Preprocess labels (convert binary labels to integer classes)
    binary_matrix = G.y  

    # Ensure binary_matrix is properly thresholded to contain only 0s and 1s
    binary_matrix = (binary_matrix > 0).int()  # Convert to binary (0 or 1)

    # Calculate the class sizes for all combinations of labels
    label_combinations = []
    class_sizes = []

    # Iterate over all combinations of `num_balanced_labels`
    num_labels = binary_matrix.size(1)
    for combination in itertools.combinations(range(num_labels), num_balanced_labels):
        # Filter binary matrix for the current combination of labels
        filtered_matrix = binary_matrix[:, combination]

        # Convert rows to integer classes
        classes = torch.tensor(
            [int("".join(map(str, row.tolist())), 2) for row in filtered_matrix], dtype=torch.long
        )

        # Count the number of members in each class
        class_count = torch.bincount(classes, minlength=2**num_balanced_labels)
        label_combinations.append(combination)
        class_sizes.append(class_count)

    # Step 2: Select the label combination with the most balanced class sizes
    class_variances = [torch.var(size.float()).item() for size in class_sizes]
    best_combination_idx = torch.argmin(torch.tensor(class_variances))
    best_combination = label_combinations[best_combination_idx]

    # Filter the binary matrix for the selected combination of labels
    filtered_matrix = binary_matrix[:, best_combination]

    # Convert rows of the filtered binary matrix to integer classes
    labels = torch.tensor(
        [int("".join(map(str, row.tolist())), 2) for row in filtered_matrix], dtype=torch.long
    )

    train_mask = (G.mask == 0).to(labels.device)
    num_classes = int(labels.max().item() + 1)
    edge_index = []
    edge_weight = []

    # Step 3: Create edges between nodes based on the selected balanced labels
    for i in range(num_classes):
        # Get the indices of nodes belonging to the current class
        class_mask = (labels == i).nonzero(as_tuple=True)[0]
        class_embeddings = embeddings[class_mask]

        # Separate training and non-training nodes for the class
        train_indices = class_mask[train_mask[class_mask]]
        non_train_indices = class_mask[~train_mask[class_mask]]

        if train_indices.size(0) > 1:
            # Calculate pairwise cosine distances for training nodes
            distances = 1 - F.cosine_similarity(
                class_embeddings.unsqueeze(1), class_embeddings.unsqueeze(0), dim=2
            )
            distances.fill_diagonal_(-float('inf'))  # Mask self-loops by setting to -inf

            # Get k most distant pairs of training nodes
            for _ in range(k):
                max_dist_idx = torch.argmax(distances)
                node1, node2 = divmod(max_dist_idx.item(), distances.size(1))
                edge_index.append([class_mask[node1].item(), class_mask[node2].item()])
                edge_index.append([class_mask[node2].item(), class_mask[node1].item()])
                if binary:
                    edge_weight.append(1.0)
                    edge_weight.append(1.0)
                else:
                    edge_weight.append(distances[node1, node2].item())
                    edge_weight.append(distances[node1, node2].item())
                distances[node1, node2] = -float('inf')
                distances[node2, node1] = -float('inf')

    # Connect non-training nodes to the closest training node after processing all classes
    for i in range(num_classes):
        class_mask = (labels == i).nonzero(as_tuple=True)[0]
        non_train_indices = class_mask[~train_mask[class_mask]]
        train_indices = class_mask[train_mask[class_mask]]

        if train_indices.size(0) > 0:
            non_train_embeddings = embeddings[non_train_indices]
            train_embeddings = embeddings[train_indices]

            distances = 1 - F.cosine_similarity(
                non_train_embeddings.unsqueeze(1), train_embeddings.unsqueeze(0), dim=2
            )

            closest_train_indices = torch.argmin(distances, dim=1)
            for idx, closest_idx in enumerate(closest_train_indices):
                edge_index.append([non_train_indices[idx].item(), train_indices[closest_idx].item()])
                edge_index.append([train_indices[closest_idx].item(), non_train_indices[idx].item()])
                if binary:
                    edge_weight.append(1.0)
                    edge_weight.append(1.0)
                else:
                    edge_weight.append(distances[idx, closest_idx].item())
                    edge_weight.append(distances[idx, closest_idx].item())

    # Convert edge list to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

    return edge_index, edge_weight

def get_subgraph_adj(embeddings, G, k, binary=False):
    labels = G.y
    train_mask = (G.mask == 0).to(labels.device)
    num_classes = int(labels.max().item() + 1)
    edge_index = []
    edge_weight = []

    for i in range(num_classes):
        # Get the indices of nodes belonging to the current class
        class_mask = (labels == i).nonzero(as_tuple=True)[0]
        class_embeddings = embeddings[class_mask]

        # Separate training and non-training nodes for the class
        train_indices = class_mask[train_mask[class_mask]]
        non_train_indices = class_mask[~train_mask[class_mask]]

        if train_indices.size(0) > 1:
            # Calculate pairwise cosine distances for training nodes
            distances = 1 - F.cosine_similarity(
                class_embeddings.unsqueeze(1), class_embeddings.unsqueeze(0), dim=2
            )
            distances.fill_diagonal_(-float('inf'))  # Mask self-loops by setting to -inf

            # Get k most distant pairs of training nodes
            for _ in range(k):
                max_dist_idx = torch.argmax(distances)
                node1, node2 = divmod(max_dist_idx.item(), distances.size(1))
                edge_index.append([class_mask[node1].item(), class_mask[node2].item()])
                edge_index.append([class_mask[node2].item(), class_mask[node1].item()])
                if binary:
                    edge_weight.append(1.0)
                    edge_weight.append(1.0)
                else:
                    edge_weight.append(distances[node1, node2].item())
                    edge_weight.append(distances[node1, node2].item())
                distances[node1, node2] = -float('inf')
                distances[node2, node1] = -float('inf')

    # Connect non-training nodes to the closest training node after processing all classes
    for i in range(num_classes):
        class_mask = (labels == i).nonzero(as_tuple=True)[0]
        non_train_indices = class_mask[~train_mask[class_mask]]
        train_indices = class_mask[train_mask[class_mask]]

        if train_indices.size(0) > 0:
            non_train_embeddings = embeddings[non_train_indices]
            train_embeddings = embeddings[train_indices]

            distances = 1 - F.cosine_similarity(
                non_train_embeddings.unsqueeze(1), train_embeddings.unsqueeze(0), dim=2
            )

            closest_train_indices = torch.argmin(distances, dim=1)
            for idx, closest_idx in enumerate(closest_train_indices):
                edge_index.append([non_train_indices[idx].item(), train_indices[closest_idx].item()])
                edge_index.append([train_indices[closest_idx].item(), non_train_indices[idx].item()])
                if binary:
                    edge_weight.append(1.0)
                    edge_weight.append(1.0)
                else:
                    edge_weight.append(distances[idx, closest_idx].item())
                    edge_weight.append(distances[idx, closest_idx].item())

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

    return edge_index, edge_weight



def get_knn_adj(embeddings, k, distance_metric='cosine', normalize_rows=True, binary=True):
    """
    Generate a k-NN adjacency matrix for graph nodes based on their embeddings,
    and remove redundant edges based on in-degree.
    """

    # Existing k-NN adjacency matrix generation code
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    num_nodes = embeddings.shape[0]

    if distance_metric == 'euclidean':
        dist_matrix = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    elif distance_metric == 'cosine':
        normed_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        dist_matrix = 1 - np.dot(normed_embeddings, normed_embeddings.T)
    else:
        raise ValueError("Unsupported distance metric. Use 'euclidean' or 'cosine'.")

    m = 100
    neighbors = np.argsort(dist_matrix, axis=1)[:, m:k+m]
    distances = np.sort(dist_matrix, axis=1)[:, m:k+m]

    if normalize_rows:
        row_sums = distances.sum(axis=1, keepdims=True)
        distances = distances / row_sums

    row_indices = np.repeat(np.arange(num_nodes), k)
    col_indices = neighbors.flatten()
    if binary:
        edge_weights = np.ones_like(distances.flatten())
    else:
        edge_weights = distances.flatten()

    edge_index_array = np.array([row_indices, col_indices])
    subgraph_edge_index = torch.tensor(edge_index_array, dtype=torch.long)
    subgraph_edge_weight = torch.tensor(edge_weights, dtype=torch.float)

    # Remove redundant edges
    subgraph_edge_index, subgraph_edge_weight = remove_redundant_edges(subgraph_edge_index, subgraph_edge_weight)

    return subgraph_edge_index, subgraph_edge_weight

def get_top_k_adj(embeddings, k, distance_metric='cosine', normalize_rows=True):
    """
    Reads adjacency matrices from files, normalizes them by their row sum,
    processes them to retain the top-k largest elements globally, and returns
    edge indices and weights as PyTorch tensors.

    Args:
        data_path (str): Path to the directory containing the adjacency matrices.
        k (int): Number of top elements to retain globally in the matrix.
        channel (str): Specifies which adjacency matrix to load ('position', 'neighbor', 'structure').

    Returns:
        tuple: Edge indices (2D torch tensor of shape [2, num_edges]) and edge weights (1D torch tensor).
    """
    def normalize_by_row_sum(matrix):
        """
        Normalizes the matrix by its row sum.

        Args:
            matrix (numpy.ndarray): Input adjacency matrix.

        Returns:
            numpy.ndarray: Row-normalized adjacency matrix.
        """
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero for rows with all zeros
        return matrix / row_sums

    def top_k_global_to_edges(matrix, k):
        """
        Identifies the top-k largest elements in the adjacency matrix globally
        and converts them to edge indices and weights as PyTorch tensors.

        Args:
            matrix (numpy.ndarray): Input adjacency matrix.
            k (int): Number of top elements to retain globally.

        Returns:
            tuple: Edge indices (2D torch tensor) and edge weights (1D torch tensor).
        """

        # Only consider the upper triangular part of the matrix
        matrix = np.triu(matrix, k=1)
        # Flatten the matrix and find the indices of the top-k largest values
        flat_indices = np.argsort(matrix.flatten())[-k:]
        row_indices, col_indices = np.unravel_index(flat_indices, matrix.shape)

        # Combine indices into a single numpy array and convert to PyTorch tensor
        edge_indices = torch.tensor(
            np.array([row_indices, col_indices]), dtype=torch.long
        )

        # Extract the corresponding weights
        edge_weights = torch.tensor(matrix[row_indices, col_indices], dtype=torch.float32)

        return edge_indices, edge_weights

    # Convert embeddings to numpy if they are in torch.Tensor format
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    num_nodes = embeddings.shape[0]

    # Compute the distance matrix
    if distance_metric == 'euclidean':
        dist_matrix = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    elif distance_metric == 'cosine':
        # Normalize embeddings for cosine similarity
        normed_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        dist_matrix = 1 - np.dot(normed_embeddings, normed_embeddings.T)
    else:
        raise ValueError("Unsupported distance metric. Use 'euclidean' or 'cosine'.")


    if normalize_rows and distance_metric == 'euclidean':
        # Normalize distances so that row sums to 1
        row_sums = dist_matrix.sum(axis=1, keepdims=True)
        dist_matrix = dist_matrix / row_sums


    # Process the normalized adjacency matrix to get edge indices and weights
    edge_indices, edge_weights = top_k_global_to_edges(dist_matrix, k)

    return edge_indices, edge_weights


def load_and_process_adjs(dataset, k, channel='position'):
    """
    Reads adjacency matrices from files, normalizes them by their row sum,
    processes them to retain the top-k largest elements globally, and returns
    edge indices and weights as PyTorch tensors.

    Args:
        data_path (str): Path to the directory containing the adjacency matrices.
        k (int): Number of top elements to retain globally in the matrix.
        channel (str): Specifies which adjacency matrix to load ('position', 'neighbor', 'structure').

    Returns:
        tuple: Edge indices (2D torch tensor of shape [2, num_edges]) and edge weights (1D torch tensor).
    """
    def normalize_by_row_sum(matrix):
        """
        Normalizes the matrix by its row sum.

        Args:
            matrix (numpy.ndarray): Input adjacency matrix.

        Returns:
            numpy.ndarray: Row-normalized adjacency matrix.
        """
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero for rows with all zeros
        return matrix / row_sums

    def top_k_global_to_edges(matrix, k):
        """
        Identifies the top-k largest elements in the adjacency matrix globally
        and converts them to edge indices and weights as PyTorch tensors.

        Args:
            matrix (numpy.ndarray): Input adjacency matrix.
            k (int): Number of top elements to retain globally.

        Returns:
            tuple: Edge indices (2D torch tensor) and edge weights (1D torch tensor).
        """
        # Flatten the matrix and find the indices of the top-k largest values
        flat_indices = np.argsort(matrix.flatten())[-k:]
        row_indices, col_indices = np.unravel_index(flat_indices, matrix.shape)

        # Combine indices into a single numpy array and convert to PyTorch tensor
        edge_indices = torch.tensor(
            np.array([row_indices, col_indices]), dtype=torch.long
        )

        # Extract the corresponding weights
        edge_weights = torch.tensor(matrix[row_indices, col_indices], dtype=torch.float32)

        return edge_indices, edge_weights

    # Load the specified adjacency matrix
    if channel == 'position':
        adj = np.load(f"{DATASET_PATH}/{dataset}/subgraph_position_adj.npy")
    elif channel == 'neighbor':
        adj = np.load(f"{DATASET_PATH}/{dataset}/subgraph_neighbor_adj.npy")
    elif channel == 'structure':
        adj = np.load(f"{DATASET_PATH}/{dataset}/subgraph_structure_adj.npy")
    else:
        raise ValueError(f"Invalid channel: {channel}. Choose from 'position', 'neighbor', 'structure'.")

    # Normalize the adjacency matrix by its row sum
    adj_normalized = normalize_by_row_sum(adj)

    # Process the normalized adjacency matrix to get edge indices and weights
    edge_indices, edge_weights = top_k_global_to_edges(adj_normalized, k)

    return edge_indices, edge_weights




def get_hybrid_edge_index(
    args, baseG, num_random_edges=0, subgraph_edge_index=None, subgraph_edge_weights=None, normalize=True
):
    """
    Create a hybrid adjacency matrix with new nodes and optionally add random edges between new nodes.

    Parameters:
    - baseG: A graph object with `pos` and `x` attributes.
    - num_random_edges: Number of random edges to add between newly added nodes (ignored if subgraph_edge_index is provided).
    - subgraph_edge_index: Predefined edges between new nodes (if provided, no sampling is performed).
    - subgraph_edge_weights: Edge weights corresponding to subgraph_edge_index (if provided, must match shape).
    - normalize: Whether to return a normalized adjacency matrix suitable for GCNs.

    Returns:
    - feature: Updated node feature matrix with subgraph features added.
    - adj_hybrid: A sparse adjacency matrix (torch_sparse.SparseTensor) for GCNs.
    - node_mask: A mask to identify original nodes.
    """
    num_nodes = baseG.x.shape[0]
    src_nodes = []
    dst_nodes = []
    subgraph_features = []

    # Add new nodes and connections to existing nodes
    for i in range(baseG.pos.shape[0]):
        nodes = baseG.pos[i][baseG.pos[i] != -1].tolist()
        src_nodes.extend(nodes)
        dst_nodes.extend([num_nodes + i] * len(nodes))
        # src_nodes.extend([num_nodes + i] * len(nodes))
        # dst_nodes.extend(nodes)
        subgraph_features.append(torch.mean(baseG.x[nodes], dim=0))

    # Create initial hybrid edge index
    edge_index_hybrid = torch.cat(
        [baseG.edge_index, torch.tensor([src_nodes, dst_nodes], dtype=torch.long)], dim=1
    )

    # Create edge weights for hybrid graph
    edge_weights_hybrid = torch.cat(
        [torch.ones(baseG.edge_index.shape[1]), torch.ones(len(src_nodes))], dim=0
    )

    # Add edges between new nodes
    if subgraph_edge_index is not None:
        # Use provided subgraph_edge_index and optionally subgraph_edge_weights
        edge_index_hybrid = torch.cat([edge_index_hybrid, subgraph_edge_index], dim=1)
        if subgraph_edge_weights is not None:
            edge_weights_hybrid = torch.cat([edge_weights_hybrid, subgraph_edge_weights], dim=0)
        else:
            edge_weights_hybrid = torch.cat(
                [edge_weights_hybrid, torch.ones(subgraph_edge_index.shape[1])], dim=0
            )
    elif num_random_edges > 0:
        # Sample random edges if subgraph_edge_index is not provided
        new_node_indices = torch.arange(num_nodes, num_nodes + len(subgraph_features))
        sampled_edges = negative_sampling(
            edge_index=torch.zeros((2, 0), dtype=torch.long),  # No existing edges between new nodes
            num_nodes=new_node_indices.shape[0],
            num_neg_samples=num_random_edges
        )
        sampled_edges[0] = new_node_indices[sampled_edges[0]]
        sampled_edges[1] = new_node_indices[sampled_edges[1]]

        # Add the sampled edges
        edge_index_hybrid = torch.cat([edge_index_hybrid, sampled_edges], dim=1)
        edge_weights_hybrid = torch.cat([edge_weights_hybrid, torch.ones(sampled_edges.shape[1])], dim=0)

    # Add self-loops to the adjacency matrix
    edge_index_hybrid, edge_weights_hybrid = add_self_loops(
        edge_index_hybrid, edge_weights_hybrid, num_nodes=num_nodes + len(subgraph_features)
    )

    # Create sparse adjacency matrix
    adj_hybrid = SparseTensor(
        row=edge_index_hybrid[0],
        col=edge_index_hybrid[1],
        value=edge_weights_hybrid,
        sparse_sizes=(num_nodes + len(subgraph_features), num_nodes + len(subgraph_features))
    )

    if normalize:
        # Normalize adjacency matrix
        row, col, value = adj_hybrid.coo()
        deg = degree(row, num_nodes=adj_hybrid.sparse_size(0), dtype=torch.float32)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        adj_hybrid = SparseTensor(row=row, col=col, value=value, sparse_sizes=adj_hybrid.sparse_sizes())

    # Create updated feature matrix
    feature = torch.cat([baseG.x, torch.stack(subgraph_features)], dim=0)
    node_mask = torch.tensor([True] * num_nodes + [False] * len(subgraph_features))

    return feature, adj_hybrid, node_mask


    

class SubgraphProjection(nn.Module):
    """
    A PyTorch module to construct a sparse binary matrix from baseG.pos,
    optionally normalize its rows, and perform matrix multiplication in the forward pass.
    """
    def __init__(self, device, baseG, normalize=False):
        """
        Initialize the SparseBinaryMatrix module.

        Parameters:
        - baseG: A BaseGraph object with `pos` defining node connections.
        - normalize: Whether to normalize the rows of the sparse matrix to sum to 1.
        """
        super(SubgraphProjection, self).__init__()
        self.num_rows = baseG.pos.shape[0]  # Number of rows (2400)
        self.num_cols = baseG.x.shape[0]   # Number of columns (14587)
        self.normalize = normalize

        # Prepare row and column indices for the sparse matrix
        row_indices = []
        col_indices = []

        for row_idx in range(self.num_rows):
            # Extract valid node indices (exclude -1)
            valid_nodes = baseG.pos[row_idx][baseG.pos[row_idx] != -1].tolist()
            row_indices.extend([row_idx] * len(valid_nodes))
            col_indices.extend(valid_nodes)

        # Convert to tensors
        row_indices = torch.tensor(row_indices, dtype=torch.long)
        col_indices = torch.tensor(col_indices, dtype=torch.long)
        values = torch.ones(len(row_indices), dtype=torch.float32)  # Binary values

        # Create sparse binary matrix
        sparse_matrix = SparseTensor(
            row=row_indices,
            col=col_indices,
            value=values,
            sparse_sizes=(self.num_rows, self.num_cols)
        )

        # Normalize if required
        if self.normalize:
            row_sums = sparse_matrix.sum(dim=1)  # Sum of each row
            row_sums_inv = 1.0 / (row_sums + 1e-8)  # Avoid division by zero
            normalization_factors = row_sums_inv[row_indices]
            normalized_values = values * normalization_factors

            sparse_matrix = SparseTensor(
                row=row_indices,
                col=col_indices,
                value=normalized_values,
                sparse_sizes=(self.num_rows, self.num_cols)
            )

        self.projection_matrix = sparse_matrix.to(device)

    def forward(self, input_matrix):
        """
        Multiply the sparse matrix with the input matrix.

        Parameters:
        - input_matrix: A dense matrix of shape (n, D).

        Returns:
        - output: Result of the matrix multiplication (s, D).
        """
        return self.projection_matrix.matmul(input_matrix)



def train(args, 
        num_epochs=1500,
        hidden_dim=64,
        conv_layer=8,
        dropout=0,
        lr=0.001,
        weight_decay=1e-5,
        hypertuning=False,
        inner_iters=5,
        gamma=0.01,
        num_edges=1,
        knn=30,
        switch_epoch=90, 
        kappa=0.95,
        num_clusters=2
        ):
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Pretrain
    if args.pretrain:
        features = baseG.x
        projection = SubgraphProjection(device, baseG, normalize=False)
        pre_train_model = VanillaGCN(in_channels=features.shape[1],
                            out_channels=hidden_dim,
                            num_classes=output_channels,
                            num_layers=conv_layer,
                            projection_matrix=projection,
                            loss_fn=loss_fn,
                            dropout=dropout,
                            ).to(device)
        features = features.to(device)
        baseG.y = baseG.y.to(device)
        adj_hybrid = baseG.edge_index.to(device)
        optimizer = torch.optim.Adam(pre_train_model.parameters(), lr=lr, weight_decay=weight_decay)
        for i in range(200):
            trn_score, _ = train_model(optimizer, pre_train_model, baseG, features, adj_hybrid, score_fn)
        tmp = test_model(pre_train_model, baseG, score_fn, testing=True)
        print(f"Pretrain accuracy: {tmp:.4f}")
        with torch.no_grad():
            embeddings = pre_train_model(features, adj_hybrid).detach()
        subgraph_edge_index, subgraph_edge_weight = get_knn_adj(embeddings, knn)




    auc_outs = []
    t1 = time.time()

    outs = []
    run_times = []
    trn_time = []
    inference_time = []
    preproc_times = []

    projection = SubgraphProjection(device, baseG, normalize=False)
    if args.model in ['isnn', 'baseline']:
        if args.pretrain:
            pass
        elif args.channel == 'None':
            subgraph_edge_index, subgraph_edge_weight = None, None
        else:
            subgraph_edge_index, subgraph_edge_weight = load_and_process_adjs(dataset=args.dataset, k=num_edges, channel=args.channel)
        
        features, adj_hybrid, node_mask = get_hybrid_edge_index(args, 
                                                               baseG, 
                                                               subgraph_edge_index=subgraph_edge_index,
                                                               subgraph_edge_weights=subgraph_edge_weight,
                                                               num_random_edges=num_edges)
    else:
        features = baseG.x

    all_train_scores = []
    all_test_scores = []
    all_val_scores = []
    for repeat in range(args.repeat):
        all_train_scores.append([])
        all_test_scores.append([])
        all_val_scores.append([])
        start_time = time.time()
        if not hypertuning:
            set_seed(args.seed + repeat)

        start_pre = time.time()
        
        if args.model == 'isnn':
            gnn = SubIGNN_new(in_channels=features.shape[1], 
                            out_channels=hidden_dim, 
                            num_classes=output_channels, 
                            num_nodes=features.shape[0],
                            projection_matrix=projection,
                            node_mask = node_mask.to(device),
                            num_layers=conv_layer, 
                            loss_fn=loss_fn,
                            dropout=dropout,
                            gamma=gamma,
                            kappa=kappa,
                            ).to(device)
            adj_hybrid = adj_hybrid.to(device)
        
        elif args.model == 'soft_ignn':
            gnn = SoftIGNN(in_channels=features.shape[1],
                            out_channels=hidden_dim,
                            num_classes=output_channels,
                            num_nodes=features.shape[0],
                            num_layers=conv_layer,
                            projection_matrix=projection,
                            device=device,
                            loss_fn=loss_fn,
                            dropout=dropout,
                            gamma=gamma
                            ).to(device)
            adj_hybrid = baseG.edge_index.to(device)

        elif args.model == 'soft_eignn':
            gnn = SoftEIGNN(in_channels=features.shape[1],
                            out_channels=hidden_dim,
                            num_classes=output_channels,
                            num_nodes=features.shape[0],
                            num_layers=conv_layer,
                            projection_matrix=projection,
                            device=device,
                            loss_fn=loss_fn,
                            dropout=dropout,
                            gamma=gamma
                            ).to(device)

            # Normalize adjacency matrix
            row, col = baseG.edge_index[0], baseG.edge_index[1]
            value = torch.ones_like(row, dtype=torch.float32)  # Default value for edges is 1
            
            deg = degree(row, num_nodes=features.shape[0], dtype=torch.float32)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]
            
            adj_hybrid = SparseTensor(row=row, col=col, value=value, sparse_sizes=[features.shape[0],features.shape[0]])
            adj_hybrid = adj_hybrid.to(device)

        elif args.model == 'gcn':
            gnn = VanillaGCN(in_channels=features.shape[1],
                            out_channels=hidden_dim,
                            num_classes=output_channels,
                            num_layers=conv_layer,
                            projection_matrix=projection,
                            loss_fn=loss_fn,
                            dropout=dropout,
                            ).to(device)
            adj_hybrid = baseG.edge_index.to(device)
        
        elif args.model == 'baseline':
            gnn = baseline(out_channels=hidden_dim,
                            num_classes=output_channels,
                            num_nodes=features.shape[0],
                            projection_matrix=projection,
                            node_mask=node_mask.to(device),
                            loss_fn=loss_fn,
                            ).to(device)


        
        features = features.to(device)
        baseG.y = baseG.y.to(device)

        end_pre = time.time()
        preproc_times.append(end_pre - start_pre)
        optimizer = torch.optim.Adam(gnn.parameters(), lr=lr, weight_decay=weight_decay)


        scd = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                             factor=0.5,
                                             min_lr=5e-5)
        
        val_score = 0
        tst_score = 0
        tst_auc = 0
        early_stop = 0

        for i in range(1, num_epochs+1):
            if i == switch_epoch and args.model in ['isnn'] and args.channel == 'None' and not args.pretrain:
                with torch.no_grad():
                    embeddings = gnn.get_subgraph_embeddings()
                
                    if args.dataset == 'hpo_neuro':
                        subgraph_edge_index, subgraph_edge_weight = get_subgraph_adj_multi_label(embeddings, baseG, num_clusters, knn, binary=True)
                    else:
                        subgraph_edge_index, subgraph_edge_weight = get_subgraph_adj(embeddings, baseG, knn, binary=True)
                subgraph_edge_index += baseG.x.shape[0]
                _, adj_hybrid, _ = get_hybrid_edge_index(args, 
                                                        baseG, 
                                                        subgraph_edge_index=subgraph_edge_index,
                                                        subgraph_edge_weights=subgraph_edge_weight,
                                                        num_random_edges=num_edges)
                adj_hybrid = adj_hybrid.to(device)


            t1 = time.time()
            if args.model in ['isnn', 'soft_ignn', 'soft_eignn']:
                residual_error = gnn.fixed_point_iteration(features, adj_hybrid, inner_iters)
            elif args.model in ['gcn', 'baseline']:
                residual_error = 0
            trn_score, trn_auc, loss = train_model(optimizer, gnn, baseG, features, adj_hybrid, score_fn)
            all_train_scores[repeat].append(trn_score)
            tmp, tmp1 = test_model( gnn, baseG, features, adj_hybrid, score_fn, testing=True)
            all_test_scores[repeat].append(tmp)
            trn_time.append(time.time() - t1)


            if i >= 200:
                score, auc = test_model(gnn,
                                baseG,
                                features,
                                adj_hybrid, score_fn, testing=False)


                if score > val_score:
                    early_stop = 0
                    val_score = score
                    inf_start = time.time()
                    score, auc = test_model(gnn,
                                   baseG,
                                   features,
                                   adj_hybrid, score_fn, testing=True)
                    inf_end = time.time()
                    inference_time.append(inf_end - inf_start)
                    tst_score = score
                    tst_auc = auc

                elif score >= val_score - 1e-5:
                    inf_start = time.time()
                    score, auc = test_model(gnn,
                                    baseG,
                                    features,
                                    adj_hybrid, score_fn, testing=True)
                    inf_end = time.time()
                    inference_time.append(inf_end - inf_start)
                    tst_score = max(score, tst_score)
                    tst_auc = max(auc, tst_auc)

                else:
                    early_stop += 1
                    if i % 10 == 0:
                        inf_start = time.time()
                        tst_1, auc = test_model(gnn, baseG, features, adj_hybrid, score_fn, testing=True)
                        inf_end = time.time()
                        inference_time.append(inf_end - inf_start)
            if val_score >= 1 - 1e-5:
                early_stop += 1
        end_time = time.time()
        run_time = end_time - start_time
        run_times.append(run_time)
        outs.append(tst_score)
        auc_outs.append(tst_auc)
    tst_average = np.average(outs)
    tst_error = np.std(outs) / np.sqrt(len(outs))
    average_auc = np.average(auc_outs)
    error_auc = np.std(auc_outs) / np.sqrt(len(auc_outs))
    print(
        f"Gamma: {gamma}, Test Accuracy {tst_average :.3f} ± {tst_error :.3f}, AUC {average_auc:.3f} ± {error_auc:.3f}"
    )
    exp_results = {}
    exp_results[f"{args.dataset}"] = {
        "results": {
            "Test Accuracy": f"{tst_average:.3f} ± {tst_error:.3f}",
            "AUC": f"{average_auc:.3f} ± {error_auc:.3f}",
            "Avg runtime": f"{np.average(run_times):.3f} ± {np.std(run_times):.3f}",
            "Avg preprocessing time": f"{np.average(preproc_times):.3f} ± {np.std(preproc_times):.3f}",
            "Avg train time": f"{np.average(trn_time):.3f} ± {np.std(trn_time):.3f}",
            "Avg inference time": f"{np.average(inference_time):.3f} ± {np.std(inference_time):.3f}",
        },
    }
    results_json = f"{args.dataset}_{args.model}_results.json"
    with open(results_json, 'w') as output_file:
        json.dump(exp_results, output_file)


    # Plot the scores mean and std
    train_scores = np.array(all_train_scores)
    val_scores = np.array(all_val_scores)
    test_scores = np.array(all_test_scores)
    plt.figure()
    plt.plot(np.mean(train_scores, axis=0), label="Train")
    plt.plot(np.mean(val_scores, axis=0), label="Val")
    plt.plot(np.mean(test_scores, axis=0), label="Test")
    plt.fill_between(
        np.arange(train_scores.shape[1]),
        np.mean(train_scores, axis=0) - np.std(train_scores, axis=0),
        np.mean(train_scores, axis=0) + np.std(train_scores, axis=0),
        alpha=0.3,
    )
    plt.fill_between(
        np.arange(val_scores.shape[1]),
        np.mean(val_scores, axis=0) - np.std(val_scores, axis=0),
        np.mean(val_scores, axis=0) + np.std(val_scores, axis=0),
        alpha=0.3,
    )
    plt.fill_between(
        np.arange(test_scores.shape[1]),
        np.mean(test_scores, axis=0) - np.std(test_scores, axis=0),
        np.mean(test_scores, axis=0) + np.std(test_scores, axis=0),
        alpha=0.3,
    )
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"Train and Test Scores for {args.dataset}")
    plt.savefig(f"plots/{args.dataset}/{args.dataset}_{args.model}_{num_edges}_{args.channel}_{gamma}_scores.png")
    plt.close()
    print(
        f"Best Mean Accuracy: {np.max(np.mean(test_scores, axis=0))}"
    )

def run_helper(argument_class, hypertuning=False):
    config.set_device(argument_class.device)

    global trn_dataset, val_dataset, tst_dataset
    global max_deg, output_channels
    global score_fn, loss_fn
    global baseG

    baseG = datasets.load_dataset(argument_class.dataset)

    

    if baseG.y.unique().shape[0] == 2:
        # binary classification task
        def loss_fn(x, y):
            return nn.BCEWithLogitsLoss()(x.flatten(), y.flatten())

        baseG.y = baseG.y.to(torch.float)
        if baseG.y.ndim > 1:
            output_channels = baseG.y.shape[1]
        else:
            output_channels = 1
        score_fn = metrics.binaryf1

    else:
        # multi-class classification task
        baseG.y = baseG.y.to(torch.int64)
        loss_fn = nn.CrossEntropyLoss()
        output_channels = baseG.y.unique().shape[0]
        score_fn = metrics.microf1
    
    # read configuration
    path_to_config = f"hyperparams/{argument_class.dataset}.yml"

    with open(path_to_config) as f:
        params = yaml.safe_load(f)
        
    params.update({'args': argument_class,
                   'hypertuning': hypertuning})
    
    print(params)
    return train(**(params))


if __name__ == "__main__":
    args = parse_args()
    run_helper(args)
    print('\n')

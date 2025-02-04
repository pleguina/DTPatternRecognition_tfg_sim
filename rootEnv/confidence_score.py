import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import matplotlib.pyplot as plt

###############################################################################
# 1) CREATE A SMALL, DUMMY GRAPH
###############################################################################
# We will create a graph with:
#   - 5 nodes
#   - each node has 4 random features
#   - a simple chain of edges (0->1->2->3->4)
# Our goal: demonstrate two approaches for node classification GNNs and plot
# their "confidence gaps" to detect "grey zones."

num_nodes = 5
in_channels = 4       # number of features per node
hidden_channels = 8   # hidden dimension for the GCN
out_channels = 3      # number of classes to predict

# Random node features: shape [num_nodes, in_channels]
x = torch.randn((num_nodes, in_channels), dtype=torch.float)

# Simple chain edges: shape [2, num_edges]
# The first row is "source nodes", the second row is "target nodes".
edge_index = torch.tensor([
    [0, 1, 2, 3],
    [1, 2, 3, 4]
], dtype=torch.long)

# Wrap our data in a PyTorch Geometric Data object
data = Data(x=x, edge_index=edge_index)

###############################################################################
# 2) BEST PRACTICE MODEL: Return RAW LOGITS in forward()
###############################################################################
# This is usually best practice if we train with nn.CrossEntropyLoss, because
# CrossEntropyLoss expects unnormalized, raw logits (real-valued scores for each possible class). Then we apply softmax
# outside of the model if we want to inspect probabilities.

class GCNLogits(torch.nn.Module):
    """
    GCNLogits:
    - Returns raw, unnormalized logits from the final layer.
    - Typically we do NOT apply softmax in forward() when training.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNLogits, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Args:
            x (Tensor): Node features [num_nodes, in_channels].
            edge_index (LongTensor): Graph connectivity [2, num_edges].
        
        Returns:
            (Tensor): shape [num_nodes, out_channels], representing raw logits.
        """
        # First graph convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # apply ReLU activation

        # Second graph convolution
        x = self.conv2(x, edge_index)
        # The result x now has shape [num_nodes, out_channels] (raw logits).

        return x  # Return raw logits (no softmax here).

###############################################################################
# 3) INFERENCE WITH THE "BEST PRACTICE" MODEL
###############################################################################
# We'll create an instance of GCNLogits and apply it to our dummy data.

model_logits = GCNLogits(in_channels, hidden_channels, out_channels)
model_logits.eval()  # Set to eval mode (no dropout, etc.)

with torch.no_grad():
    # Get raw logits from the model
    logits = model_logits(data.x, data.edge_index)
    # Convert raw logits to probabilities for inspection/interpretation
    probabilities_logits_model = F.softmax(logits, dim=-1)

# We'll record the "confidence gap" for each node, i.e. the difference
# between the top-1 probability and the top-2 probability for that node.
confidence_gaps_logits = []

# We'll define a threshold below which we consider the prediction to be
# in a "grey zone" (high uncertainty).
threshold = 0.1

print("=== Best Practice Model Output (Logits -> Softmax) ===")
for node_idx, prob_vec in enumerate(probabilities_logits_model):
    # prob_vec is shape [out_channels], each entry is the predicted
    # probability for one class for this node.
    top2_probs, top2_classes = torch.topk(prob_vec, 2)
    confidence_gap = top2_probs[0] - top2_probs[1]
    confidence_gaps_logits.append(confidence_gap.item())

    print(f"Node {node_idx} - Probability distribution: {prob_vec.tolist()}")
    print(f"   Top-2 classes = {top2_classes.tolist()} with probs = {top2_probs.tolist()}")
    print(f"   Confidence gap = {confidence_gap:.3f}")

    # If the gap is small, we say it's a "grey zone."
    if confidence_gap < threshold:
        print("   -> Grey zone (uncertain between these two classes).")

    print()  # Blank line for clarity

###############################################################################
# 4) DIRECT PROBABILITY MODEL: Return SOFTMAX in forward()
###############################################################################
# This approach applies softmax directly in the forward method, returning
# probabilities for each node. This can be convenient for inference if we
# want probabilities immediately, but for training with CrossEntropyLoss we
# generally do not want to do this inside the forward pass.

class GCNProbabilities(torch.nn.Module):
    """
    GCNProbabilities:
    - Returns probability distributions (softmaxed) directly from forward().
    - Convenient for inference, but less common for training with CrossEntropy.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNProbabilities, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Args:
            x (Tensor): Node features [num_nodes, in_channels].
            edge_index (LongTensor): Graph connectivity [2, num_edges].
        
        Returns:
            (Tensor): shape [num_nodes, out_channels], representing
                       probability distributions (rows sum to 1).
        """
        # First graph convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second graph convolution
        x = self.conv2(x, edge_index)
        # Now apply softmax to convert logits to probabilities
        x = F.softmax(x, dim=-1)
        return x

###############################################################################
# 5) INFERENCE WITH THE "DIRECT PROBABILITY" MODEL
###############################################################################
model_probs = GCNProbabilities(in_channels, hidden_channels, out_channels)
model_probs.eval()

with torch.no_grad():
    # Directly get probabilities from the forward pass
    probabilities_direct_model = model_probs(data.x, data.edge_index)

confidence_gaps_probs = []

print("=== Direct Probability Model Output (Softmax in forward) ===")
for node_idx, prob_vec in enumerate(probabilities_direct_model):
    top2_probs, top2_classes = torch.topk(prob_vec, 2) # Get top-2 classes
    confidence_gap = top2_probs[0] - top2_probs[1] # Compute confidence gap
    confidence_gaps_probs.append(confidence_gap.item()) # Record gap

    print(f"Node {node_idx} - Probability distribution: {prob_vec.tolist()}")
    print(f"   Top-2 classes = {top2_classes.tolist()} with probs = {top2_probs.tolist()}")
    print(f"   Confidence gap = {confidence_gap:.3f}")

    if confidence_gap < threshold:
        print("   -> Grey zone (uncertain between these two classes).")

    print()

###############################################################################
# 6) PLOT THE CONFIDENCE GAPS FOR EACH APPROACH
###############################################################################
# We want a visual way to see whether the model is certain or uncertain.
# We define "confidence gap" = top-1 probability - top-2 probability.
# A small gap means the model thinks two classes are almost equally likely
# -> "grey zone."

node_indices = list(range(num_nodes))  # [0, 1, 2, 3, 4]

# We'll create two subplots side-by-side:
#   Left: The confidence gaps from the "logits then softmax" approach
#   Right: The confidence gaps from the "direct probability" approach

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# --- PLOT for GCNLogits approach ---
axes[0].bar(node_indices, confidence_gaps_logits, color='blue', alpha=0.6)
axes[0].axhline(threshold, color='red', linestyle='--', label='Grey zone threshold')
axes[0].set_title("Confidence Gaps (Logits -> Softmax)")
axes[0].set_xlabel("Node Index")
axes[0].set_ylabel("Confidence Gap (Top1 - Top2)")
axes[0].set_xticks(node_indices)
axes[0].legend()

# --- PLOT for GCNProbabilities approach ---
axes[1].bar(node_indices, confidence_gaps_probs, color='green', alpha=0.6)
axes[1].axhline(threshold, color='red', linestyle='--', label='Grey zone threshold')
axes[1].set_title("Confidence Gaps (Softmax in forward)")
axes[1].set_xlabel("Node Index")
axes[1].set_ylabel("Confidence Gap (Top1 - Top2)")
axes[1].set_xticks(node_indices)
axes[1].legend()

plt.tight_layout()
plt.show()


""" Empirical Analysis on a Validation Set

    Collect top-2 probability gaps for all predictions on a validation set.
    Visualize the distribution (e.g., a histogram) of these gaps for correctly classified vs. incorrectly classified examples.
    Look for a natural separation (if any) between correct and incorrect groups in the gap distribution.
        Often, incorrectly classified examples tend to have smaller gaps (more confusion).
    Choose a threshold that maximizes some desired metric on the validation set. For example:
        Maximize accuracy among “confident” predictions, while limiting the fraction of “grey zone” predictions.
        Or minimize misclassification among the subset the model deems “confident.”

This approach is similar to how one chooses a decision threshold in binary classification. Instead of separating positive vs. negative, you separate confident predictions vs. grey zone based on the confidence gap.

 """
# Graph Neural Network–Based Anomaly Detection System

> A research-oriented implementation of node-level anomaly detection on graph-structured data using Graph Neural Networks (GNNs).

---

## Repository Description

Node-level anomaly detection on graph datasets using GCN and GAT architectures built with PyTorch Geometric.

---

## GitHub Topics

```text
graph-neural-network anomaly-detection gcn gat pytorch-geometric
node-classification graph-learning deep-learning python research
```

---

# 1. Project Overview

This project implements a graph-based anomaly detection framework using Graph Neural Networks (GNNs). The system focuses on node-level anomaly detection by learning structural and feature-based representations from graph data.

The project compares Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) on benchmark graph datasets using PyTorch Geometric.

---

# 2. Problem Statement

Traditional anomaly detection techniques operate primarily on tabular or independent data samples and often fail to capture relationships between entities.

Graph Neural Networks enable anomaly detection by incorporating both:

* Node feature information
* Graph topology and neighbourhood relationships

This project explores how graph-based deep learning architectures improve anomaly detection performance on graph-structured datasets.

---

# 3. Objectives

* Implement graph-based anomaly detection using GNN architectures
* Compare GCN and GAT models
* Evaluate model performance on benchmark graph datasets
* Study the impact of graph structure on anomaly detection
* Build a modular and reproducible experimentation pipeline

---

# 4. Methodology

```text
Raw Graph Dataset
        │
        ▼
Graph Construction
(nodes, edges, features)
        │
        ▼
Preprocessing
(normalization, train/test split)
        │
        ▼
Model Training
(GCN / GAT)
        │
        ▼
Node Embedding Generation
        │
        ▼
Classification
(normal vs anomalous)
        │
        ▼
Evaluation
(F1, Precision, Recall, AUC)
```

---

# 5. Models Used

## 5.1 Graph Convolutional Network (GCN)

GCN performs neighbourhood aggregation using normalized graph convolution operations.

```math
H^{(l+1)} = σ( D^{-1/2} A D^{-1/2} H^{(l)} W^{(l)} )
```

### Key Characteristics

* Spectral graph convolution
* Efficient message passing
* Strong baseline for graph learning tasks

---

## 5.2 Graph Attention Network (GAT)

GAT introduces attention mechanisms that allow the model to assign different importance to neighbouring nodes.

```math
α_{ij} = softmax(LeakyReLU(a^T[Wh_i || Wh_j]))
```

### Key Characteristics

* Attention-based neighbour aggregation
* Adaptive weighting of node relationships
* Improved representation learning

---

# 6. Dataset Information

| Property     | Details                             |
| ------------ | ----------------------------------- |
| Dataset Type | Citation / Graph Benchmark Datasets |
| Task         | Node-level anomaly detection        |
| Graph Type   | Undirected graphs                   |
| Labels       | Binary classification               |
| Features     | Node attribute vectors              |

> Update this section with the exact dataset names used in your implementation.

---

# 7. Data Preprocessing

The preprocessing pipeline includes:

1. Graph loading
2. Feature normalization
3. Self-loop addition
4. Label encoding
5. Train/validation/test split generation
6. Class imbalance handling

Example:

```python
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


dataset = Planetoid(
    root='data/',
    name='Cora',
    transform=NormalizeFeatures()
)

data = dataset[0]
```

---

# 8. Model Training Pipeline

```python
model = GAT(in_channels, hidden_channels, out_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()
```

---

# 9. Evaluation Metrics

Due to class imbalance in anomaly detection, multiple evaluation metrics are used.

| Metric    | Purpose                         |
| --------- | ------------------------------- |
| Accuracy  | Overall correctness             |
| Precision | Correct anomaly predictions     |
| Recall    | Ability to detect anomalies     |
| F1-Score  | Balance of precision and recall |
| AUC-ROC   | Classification separability     |

---

# 10. Results

| Model | Accuracy | Precision | Recall | F1-Score |
| ----- | -------- | --------- | ------ | -------- |
| GCN   | TBD      | TBD       | TBD    | TBD      |
| GAT   | TBD      | TBD       | TBD    | TBD      |

> Replace placeholder values with actual experimental results.

---

# 11. Tech Stack

| Component               | Technology        |
| ----------------------- | ----------------- |
| Programming Language    | Python            |
| Deep Learning Framework | PyTorch           |
| Graph Learning Library  | PyTorch Geometric |
| Data Processing         | NumPy, Pandas     |
| Visualization           | Matplotlib        |
| Graph Utilities         | NetworkX          |
| Evaluation              | scikit-learn      |
| Development Environment | Jupyter Notebook  |

---

# 12. Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/gnn-anomaly-detection.git
cd gnn-anomaly-detection
```

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 13. Usage

## Train Model

```bash
python train.py --model gat
```

## Evaluate Model

```bash
python evaluate.py
```

---

# 14. Recommended Folder Structure

```text
gnn-anomaly-detection/
│
├── data/
├── notebooks/
├── models/
├── results/
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 15. Future Improvements

* Implement GraphSAGE and Graph Transformer models
* Add unsupervised anomaly detection methods
* Improve handling of class imbalance
* Support large-scale graphs with mini-batch sampling
* Add hyperparameter optimization
* Extend to edge-level anomaly detection

---

# 16. License

This project is licensed under the MIT License.

---

# References

1. Kipf, T. N., & Welling, M. — Graph Convolutional Networks
2. Veličković, P., et al. — Graph Attention Networks
3. Hamilton, W., et al. — Representation Learning on Graphs

---

## Author

Vidhi Mala

B.Tech CSE (AIML)

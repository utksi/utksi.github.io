
- KANs proposed by [Liu et al.](https://arxiv.org/abs/2404.19756).
- See [Fourier-KAN](https://github.com/GistNoesis/FourierKAN) implementation, replaces splines with fourier coefficients.

## General Message Passing Neural Network (MPNN)

1. **Input Node and Edge Features**:

   - Nodes: $\mathbf{x}_i$ (node features)
   - Edges: $\mathbf{e}_{ij}$ (edge features)

2. **Message Passing Layer** (per layer):

    a. **Edge Feature Transformation**:

      $$\mathbf{e}'_{ij} = f_e(\mathbf{e}_{ij})$$

      where $f_e$ is a transformation function applied to edge features.

    b. **Message Computation**:

      $$\mathbf{m}_{ij} = f_m(\mathbf{x}_i, \mathbf{x}_j, \mathbf{e}'_{ij})$$

      where $f_m$ computes messages using node features $\mathbf{x_i} ,\ \mathbf{x_j}$, and transformed edge features $\mathbf{e}'_{ij}$.

    c. **Message Aggregation**:

      $$\mathbf{m}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}$$

      where $\mathcal{N}(i)$ denotes the set of neighbors of node $i$.

    d. **Node Feature Update**:
   
      $$\mathbf{x}'_i = f_n(\mathbf{x}_i, \mathbf{m}_i)$$

      where $f_n$ updates node features using the aggregated messages $\mathbf{m}_i$.

3. **Output Node and Edge Features**:
   
   - Nodes: $\mathbf{x}'_i$ (updated node features)
   - Edges: $\mathbf{e}'_{ij}$ (updated edge features)

## E3-Equivariant GNN with Learnable Activation Functions on Edges

1. **Input Node and Edge Features**:
   
   - Nodes: $\mathbf{x}_i$ (node features)
   - Edges: $\mathbf{e}_{ij}$ (edge features)

2. **Learnable Edge Feature Transformation**:
    
    - **Fourier-based Edge Transformation**:
   
      $$\mathbf{e}'_{ij} = \text{FourierTransform}(\mathbf{e}_{ij})$$

      where the Fourier transformation is applied to edge features. Specifically, the transformation is defined as:

      $$\mathbf{e}'_{ij} = \sum_{k=1}^{K} a_{ij,k} \cos(k \mathbf{e}_{ij}) + b_{ij,k} \sin(k \mathbf{e}_{ij})$$

      Here, $a_{ij,k}$ and $b_{ij,k}$ are learnable parameters, and $K$ is the number of Fourier terms.

3. **Message Passing and Aggregation**:
   
   a. **Message Computation**:

      $$\mathbf{m}_{ij} = \mathbf{e}'_{ij} \odot \mathbf{x}_j$$
      
      where $\odot$ denotes element-wise multiplication, combining the transformed edge features $\mathbf{e}'_{ij}$ with the neighboring node features $\mathbf{x}_j$.

   b. **Message Aggregation**:
   
      $$\mathbf{m}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}$$

   c. **Simple Node Feature Transformation**:
   
      $$\mathbf{x}'_i = \mathbf{W} (\mathbf{x}_i + \mathbf{m}_i) + \mathbf{b}$$
      
      where $\mathbf{W}$ is a learnable weight matrix and $\mathbf{b}$ is a bias vector.

4. **Output Node and Edge Features**:
   
   - Nodes: $\mathbf{x}'_i$ (updated node features)
   - Edges: $\mathbf{e}'_{ij}$ (updated edge features)

## Full Implementation

```python
import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.transforms import Distance
from torch_geometric.nn import MessagePassing
from torch.optim import Adam
from e3nn import o3
from e3nn.nn import Gate, FullyConnectedNet

  
class LearnableActivationEdge(nn.Module):

"""

Class to define learnable activation functions on edges using Fourier series.

Inspired by Kolmogorov-Arnold Networks (KANs) to capture complex, non-linear transformations on edge features.

"""

def __init__(self, inputdim, outdim, num_terms, addbias=True):

"""

Initialize the LearnableActivationEdge module.

Args:

inputdim (int): Dimension of input edge features.

outdim (int): Dimension of output edge features.

num_terms (int): Number of Fourier terms.

addbias (bool): Whether to add a bias term. Default is True.

"""

super(LearnableActivationEdge, self).__init__()

self.num_terms = num_terms

self.addbias = addbias

self.inputdim = inputdim

self.outdim = outdim

  

# Initialize learnable Fourier coefficients

self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, num_terms) /

(torch.sqrt(torch.tensor(inputdim)) * torch.sqrt(torch.tensor(num_terms))))

if self.addbias:

self.bias = nn.Parameter(torch.zeros(1, outdim))

  

def forward(self, edge_attr):

"""

Forward pass to apply learnable activation functions on edge attributes.

Args:

edge_attr (Tensor): Edge attributes of shape (..., inputdim).

Returns:

Tensor: Transformed edge attributes of shape (..., outdim).

"""

# Reshape edge attributes for Fourier transformation

xshp = edge_attr.shape

outshape = xshp[0:-1] + (self.outdim,)

edge_attr = torch.reshape(edge_attr, (-1, self.inputdim))

  

# Generate Fourier terms

k = torch.reshape(torch.arange(1, self.num_terms + 1, device=edge_attr.device), (1, 1, 1, self.num_terms))

xrshp = torch.reshape(edge_attr, (edge_attr.shape[0], 1, edge_attr.shape[1], 1))

  

# Compute cosine and sine components

c = torch.cos(k * xrshp)

s = torch.sin(k * xrshp)

  

# Apply learnable Fourier coefficients

y = torch.sum(c * self.fouriercoeffs[0:1], (-2, -1))

y += torch.sum(s * self.fouriercoeffs[1:2], (-2, -1))

  

# Add bias if applicable

if self.addbias:

y += self.bias

  

# Reshape to original edge attribute shape

y = torch.reshape(y, outshape)

return y

  

class E3EquivariantGNN(MessagePassing):

"""

E(3)-Equivariant Graph Neural Network (GNN) that focuses on learnable activation functions on edges.

"""

def __init__(self, in_features, out_features, hidden_dim, num_layers, num_terms):

"""

Initialize the E3EquivariantGNN module.

Args:

in_features (int): Dimension of input node features.

out_features (int): Dimension of output node features.

hidden_dim (int): Dimension of hidden layers.

num_layers (int): Number of layers in the network.

num_terms (int): Number of Fourier terms for learnable activation functions.

"""

super(E3EquivariantGNN, self).__init__(aggr='add')

self.num_layers = num_layers

# Define the input and output irreps (representations)

self.input_irrep = o3.Irreps.spherical_harmonics(lmax=1) # Example irreps, adjust as needed

self.output_irrep = o3.Irreps([(out_features, (0, 1))]) # Scalar output

# Define the hidden irreps

hidden_irreps = [o3.Irreps.spherical_harmonics(lmax=1) for _ in range(num_layers)] # Adjust as needed

# Create the equivariant layers and learnable activation functions on edges

self.fourier_layers = nn.ModuleList([

LearnableActivationEdge(in_features if i == 0 else hidden_dim, hidden_dim, num_terms)

for i in range(num_layers)

])

self.layers = nn.ModuleList([

Gate(self.input_irrep, hidden_irreps[0], kernel_size=num_terms),

*[Gate(hidden_irreps[i], hidden_irreps[i+1], kernel_size=num_terms) for i in range(num_layers-1)],

Gate(hidden_irreps[-1], self.output_irrep, kernel_size=num_terms)

])

# Output layer

self.output_layer = nn.Linear(hidden_dim, out_features)

  

def forward(self, x, edge_index, edge_attr):

"""

Forward pass to propagate node features through the GNN.

Args:

x (Tensor): Node features of shape (num_nodes, in_features).

edge_index (Tensor): Edge indices of shape (2, num_edges).

edge_attr (Tensor): Edge attributes of shape (num_edges, edge_dim).

Returns:

Tensor: Output node features of shape (num_nodes, out_features).

"""

row, col = edge_index

  

# Apply Fourier-based message passing and equivariant transformations

for i in range(self.num_layers):

# Transform edge features with Fourier series

fourier_messages = self.fourier_layers[i](edge_attr)

# Apply equivariant transformations to node features

x = self.layers[i](x, fourier_messages)

# Compute messages

m_ij = fourier_messages[col] * x[row]

# Aggregate messages

m_i = scatter_add(m_ij, row, dim=0, dim_size=x.size(0))

# Update node features

x = m_i

  

# Apply the final linear layer

x = self.output_layer(x)

return x

# Load and prepare the QM9 dataset
dataset = QM9(root='data/QM9')
dataset.transform = Distance(norm=False)

  

# Split dataset into training, validation, and test sets

train_dataset = dataset[:110000]

val_dataset = dataset[110000:120000]

test_dataset = dataset[120000:]

  

# Data loaders for training, validation, and test sets

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

  

# Define the loss function and optimizer

criterion = nn.MSELoss()

model = E3EquivariantGNN(in_features=16, out_features=1, hidden_dim=32, num_layers=3, num_terms=5)

optimizer = Adam(model.parameters(), lr=1e-3)

  

def train_step(model, optimizer, criterion, data):

"""

Perform a single training step.

Args:

model (nn.Module): The neural network model.

optimizer (Optimizer): The optimizer.

criterion (Loss): The loss function.

data (Data): The input data batch.

Returns:

float: The loss value.

"""

model.train()

optimizer.zero_grad()

out = model(data.x, data.edge_index, data.edge_attr)

loss = criterion(out, data.y)

loss.backward()

optimizer.step()

return loss.item()

  

# Training loop

num_epochs = 100

for epoch in range(num_epochs):

train_loss = 0

for data in train_loader:

train_loss += train_step(model, optimizer, criterion, data)

train_loss /= len(train_loader)

  

val_loss = 0

model.eval()

with torch.no_grad():

for data in val_loader:

out = model(data.x, data.edge_index, data.edge_attr)

loss = criterion(out, data.y)

val_loss += loss.item()

val_loss /= len(val_loader)

  

print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
```
## Detailed Explanation of Mathematical Formulations

### Learnable Edge Feature Transformation
For each edge $(i, j)$ with feature $\mathbf{e}_{ij}$:

$$
\mathbf{e}'_{ij} = \sum_{k=1}^{K} a_{ij,k} \cos(k \mathbf{e}_{ij}) + b_{ij,k} \sin(k \mathbf{e}_{ij})
$$

where $a_{ij,k}$ and $b_{ij,k}$ are learnable parameters, and $K$ is the number of terms.

### Message Computation
For each edge $(i, j)$:

$$
\mathbf{m}_{ij} = \mathbf{e}'_{ij} \odot \mathbf{x}_j
$$

where $\odot$ denotes element-wise multiplication.

### Message Aggregation
For each node $i$:

$$
\mathbf{m}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}
$$

where $\mathcal{N}(i)$ denotes the set of neighbors of node $i$.

### Node Feature Update
For each node $i$:

$$
\mathbf{x}'_i = \mathbf{W} (\mathbf{x}_i + \mathbf{m}_i) + \mathbf{b}
$$

where $\mathbf{W}$ is a learnable weight matrix and $\mathbf{b}$ is a bias vector.

## Summary

This implementation combines the learnable activation functions on edges with E(3) equivariant transformations on node features. The detailed mathematical formulations provided in the comments explain each step of the process, making it suitable for a physicist audience familiar with these concepts.

#Idea #TODO: KANs for learnable edge activations in MACE - to have it as an option. Train on the same set.

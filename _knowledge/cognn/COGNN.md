## Detailed Analysis of Cooperative Graph Neural Networks (Co-GNNs)

- Proposed by [Finkelstein et al.](https://arxiv.org/abs/2310.01267)

### **Framework Overview**

Co-GNNs introduce a novel, flexible message-passing mechanism where each node in the graph dynamically selects from the actions: `listen`, `broadcast`, `listen and broadcast`, or `isolate`. This is facilitated by two cooperating networks:

1. **Action Network ($\large{\pi}$)**: Determines the optimal action for each node.
2. **Environment Network ($\large{\eta}$)**: Updates the node states based on the chosen actions.

## **Mathematical Formulation**

1. **Action Selection (Action Network π)**:
    - For each node $\large{v}$ , the action network predicts a probability distribution  $\large{p^{(\ell)}_v}$  over the actions {S, L, B, I} at layer  $\ell$ .
    
    $$
      p^{(\ell)}_v = \pi \left( h^{(\ell)}_v, \{ h^{(\ell)}_u \mid u \in N_v \} \right)
    $$
    
    - Actions are sampled using the Straight-through Gumbel-softmax estimator.

2. **State Update (Environment Network η)**:
    - The environment network updates the node states based on the selected actions.
    
    $$
      h^{(\ell+1)}_v = \begin{cases} 
      \eta^{(\ell)} \left( h^{(\ell)}_v, \{ \} \right) & \text{if } a^{(\ell)}_v = \text{I or B} \\
      \eta^{(\ell)} \left( h^{(\ell)}_v, \{ h^{(\ell)}_u \mid u \in N_v, a^{(\ell)}_u = \text{S or B} \} \right) & \text{if } a^{(\ell)}_v = \text{L or S}
      \end{cases}
    $$

3. **Layer-wise Update**:
    - A Co-GNN layer involves predicting actions, sampling them, and updating node states.
    - Repeated for  L  layers to obtain final node representations  $\large{h^{(L)}_v}$ .

### **Environment Network η Details**

The environment network updates node states using a message-passing scheme based on the selected actions. Let’s consider the standard GCN layer and how it adapts to Co-GNN concepts:

1. **Message Aggregation**:
    - For each node  v , aggregate messages from its neighbors  u  that are broadcasting or using the standard action.
    $$
      m_v^{(\ell)} = \sum_{u \in N_v, a_u^{(\ell)}\ =\  \text{S or B}} h_u^{(\ell)}
    $$


2. **Node Update**:
    - The node updates its state based on the aggregated messages and its current state.
    $$
      h_v^{(\ell+1)} = \sigma \left( W^{(\ell)}_s h_v^{(\ell)} + W^{(\ell)}_n m_v^{(\ell)} \right)
    $$


### **Properties and Benefits**

- **Task-specific**: Nodes learn to focus on relevant neighbors based on the task.
- **Directed**: Edges can become directed, influencing directional information flow.
- **Dynamic and Feature-based**: Adapt to changing graph structures and node features.
- **Asynchronous Updates**: Nodes can be updated independently.
- **Expressive Power**: More expressive than traditional GNNs, capable of handling long-range dependencies and reducing over-squashing and over-smoothing.

### **Example Implementation**

Consider a GCN (Graph Convolutional Network) adapted with Co-GNN concepts:

1. **GCN Layer (Traditional)**:
    
    $$
      h^{(\ell+1)}_v = \sigma \left( W^{(\ell)}_s h^{(\ell)}_v + W^{(\ell)}_n \sum_{u \in N_v} h^{(\ell)}_u \right)
    $$


2. **Co-GNN Layer**:
    - **Action Network**: Predicts action probabilities for each node.
    
    $$
      p^{(\ell)}_v = \text{Softmax} \left( W_a h^{(\ell)}_v + b_a \right)
    
    $$

    - **Action Sampling**: Gumbel-softmax to select actions.
    $$
      a^{(\ell)}_v \sim \text{Gumbel-Softmax}(p^{(\ell)}_v)
    $$
    
    - **State Update (Environment Network)**: 
    
    $$
      h^{(\ell+1)}_v = \begin{cases} 
      \sigma \left( W^{(\ell)}_s h^{(\ell)}_v \right) & \text{if } a^{(\ell)}_v = \text{I or B} \\
      \sigma \left( W^{(\ell)}_s h^{(\ell)}_v + W^{(\ell)}_n \sum_{u \in N_v, a^{(\ell)}_u = \text{S or B}} h^{(\ell)}_u \right) & \text{if } a^{(\ell)}_v = \text{L or S}
      \end{cases}
    $$

## Conclusion

Co-GNNs represent a significant advancement in GNN architectures, offering a dynamic and adaptive message-passing framework that improves the handling of complex graph structures and long-range dependencies. The introduction of the Action Network and Environment Network provides a more flexible and task-specific approach to node state updates, leading to superior performance on various graph-related tasks.

For further details, refer to the [manuscript](https://arxiv.org/abs/2310.01267).


# Integrating Co-GNN Concepts into MACE

#### **1. Node Representation**
Each node i is represented by:

$$
\large{\sigma_i^{(t)} = (r_i, z_i, h_i^{(t)})}
$$

where  $r_i \in \mathbb{R}^3$  is the position,  $z_i$  is the chemical element, and  $h_i^{(t)}$  are the learnable features at layer $t$.

#### **2. Action Network (π)**
For each atom i at layer t, the Action Network $\pi$ predicts a probability distribution over actions {S, L, B, I}:

$$
\large{p_i^{(t)} = \pi(\sigma_i^{(t)}, \{\sigma_j^{(t)} | j \in N(i)\})}
$$

#### **3. Action Sampling**
Actions are sampled using the Straight-through Gumbel-softmax estimator:

$$
\large{a_i^{(t)} \sim \text{Gumbel-Softmax}(p_i^{(t)})}
$$

#### **4. Message Construction**
Messages are constructed hierarchically using body order expansion, modified to consider only neighbors that are broadcasting (B) or using the standard action (S):

$$
\large{m_i^{(t)} = \sum_{j \in N(i), a_j^{(t)} \in \{S, B\}} u_1(\sigma_i^{(t)}, \sigma_j^{(t)}) + \sum_{j_1, j_2 \in N(i), a_{j_1}^{(t)} \in \{S, B\}, a_{j_2}^{(t)} \in \{S, B\}} u_2(\sigma_i^{(t)}, \sigma_{j_1}^{(t)}, \sigma_{j_2}^{(t)}) + \cdots + \sum_{j_1, \ldots, j_\nu \in N(i), a_{j_1}^{(t)} \in \{S, B\}, \ldots, a_{j_\nu}^{(t)} \in \{S, B\}} u_\nu(\sigma_i^{(t)}, \sigma_{j_1}^{(t)}, \ldots, \sigma_{j_\nu}^{(t)})}
$$


For the two-body interactions:

$$
\large{A_i^{(t)} = \sum_{j \in N(i), a_j^{(t)} \in \{S, B\}} R_{kl_1l_2l_3}^{(t)}(r_{ij}) Y_{l_1}^{m_1}(\hat{r}_{ij}) W_{kk_2l_2}^{(t)} h_{j,k_2l_2m_2}^{(t)}}
$$

where  R  is a learnable radial basis,  Y  are spherical harmonics, and  W  are learnable weights.

#### **5. Higher-order Feature Construction**
Higher-order features are constructed using tensor products and symmetrization, modified to consider the actions of neighboring atoms:

$$
\large{B_{i, \eta \nu k LM}^{(t)} = \sum_{lm} C_{LM \eta \nu, lm} \prod_{\xi=1}^\nu \sum_{k_\xi} w_{kk_\xi l_\xi}^{(t)} A_{i, k_\xi l_\xi m_\xi}^{(t)}}
$$

where  C  are generalized Clebsch-Gordan coefficients.

#### **6. State Update (Environment Network η)**
The state update is modified based on the sampled actions:
- If $a_i^{(t)} \in \{L, S\}$:

$$
\large{h_i^{(t+1)} = \eta^{(t)}(h_i^{(t)}, \{h_j^{(t)} | j \in N(i), a_j^{(t)} \in \{S, B\}\})}
$$

- If $a_i^{(t)} \in \{I, B\}$:

$$
\large{h_i^{(t+1)} = \eta^{(t)}(h_i^{(t)}, \{\})}
$$

#### **7. Readout Phase**
In the readout phase, invariant features are mapped to site energies:

$$
\large{E_i = E_i^{(0)} + E_i^{(1)} + \cdots + E_i^{(T)}}
$$

where:

$$
\large{E_i^{(t)} = R_t(h_i^{(t)}) = \sum_{k'} W_{\text{readout}, k'}^{(t)} h_{i, k' 00}^{(t)} \quad \text{for } t < T}
$$

$$
\large{E_i^{(T)} = \text{MLP}_{\text{readout}}^{(t)}(\{h_{i, k 00}^{(t)}\})}
$$

#### **8. Equivariance**
The model ensures equivariance under rotation  $Q \in O(3)$ :

$$
\large{h_i^{(t)}(Q \cdot (r_1, \ldots, r_N)) = D(Q) h_i^{(t)}(r_1, \ldots, r_N)}
$$

where  $D(Q)$  is a Wigner D-matrix. For feature  $\large{h_{i, k LM}^{(t)}}$ , it transforms as:

$$
\large{h_{i, k LM}^{(t)}(Q \cdot (r_1, \ldots, r_N)) = \sum_{M'} D_L(Q)_{M'M} h_{i, k LM'}^{(t)}(r_1, \ldots, r_N)}
$$

### Conclusion

By incorporating the dynamic message-passing strategy of Co-GNNs into the MACE framework, we can enhance its flexibility and adaptability. This involves using an Action Network to determine the message-passing strategy for each atom, modifying the message construction and state update equations accordingly. This integration retains the equivariance properties of MACE while potentially improving its expressiveness and ability to capture complex interactions(?).



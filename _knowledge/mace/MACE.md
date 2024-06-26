### **Introduction**
MACE (Message Passing Atomic Cluster Expansion) is an equivariant message passing neural network that uses higher-order messages to enhance the accuracy and efficiency of force fields in computational chemistry.

### **Node Representation**
Each node $\large{i}$ is represented by:

$$
\large{\sigma_i^{(t)} = (r_i, z_i, h_i^{(t)})}
$$

where $r_i \in \mathbb{R}^3$ is the position, $\large{z_i}$ is the chemical element, and $\large{h_i^{(t)}}$ are the learnable features at layer $\large{t}$.

### **Message Construction**
Messages are constructed hierarchically using a body order expansion:

$$
m_i^{(t)} = \sum_j u_1(\sigma_i^{(t)}, \sigma_j^{(t)}) + \sum_{j_1, j_2} u_2(\sigma_i^{(t)}, \sigma_{j_1}^{(t)}, \sigma_{j_2}^{(t)}) + \cdots + \sum_{j_1, \ldots, j_\nu} u_\nu(\sigma_i^{(t)}, \sigma_{j_1}^{(t)}, \ldots, \sigma_{j_\nu}^{(t)})
$$

### **Two-body Message Construction**
For two-body interactions, the message  $m_i^{(t)}$  is:

$$
A_i^{(t)} = \sum_{j \in N(i)} R_{kl_1l_2l_3}^{(t)}(r_{ij}) Y_{l_1}^{m_1}(\hat{r}_{ij}) W_{kk_2l_2}^{(t)} h_{j,k_2l_2m_2}^{(t)}
$$

where  $\large{R}$  is a learnable radial basis,  $\large{Y}$  are spherical harmonics, and  $\large{W}$  are learnable weights.  $\large{C}$  are Clebsch-Gordan coefficients ensuring equivariance.

### **Higher-order Feature Construction**
Higher-order features are constructed using tensor products and symmetrization:

$$
\large{B_{i, \eta \nu k LM}^{(t)} = \sum_{lm} C_{LM \eta \nu, lm} \prod_{\xi=1}^\nu \sum_{k_\xi} w_{kk_\xi l_\xi}^{(t)} A_{i, k_\xi l_\xi m_\xi}^{(t)}}
$$

where  $\large{C}$  are generalized Clebsch-Gordan coefficients.

### **Message Passing**
The message passing updates the node features by aggregating messages:

$$
\large{h_i^{(t+1)} = U_{kL}^{(t)}(\sigma_i^{(t)}, m_i^{(t)}) = \sum_{k'} W_{kL, k'}^{(t)} m_{i, k' LM} + \sum_{k'} W_{z_i kL, k'}^{(t)} h_{i, k' LM}^{(t)}}
$$

### **Readout Phase**
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

### **Equivariance**
The model ensures equivariance under rotation  $\large{Q \in O(3)}$ :

$$
\large{h_i^{(t)}(Q \cdot (r_1, \ldots, r_N)) = D(Q) h_i^{(t)}(r_1, \ldots, r_N)}
$$

where $\large{D(Q)}$ is a Wigner D-matrix. For feature $\large{h_{i, k LM}^{(t)}}$, it transforms as:

$$
\large{h_{i, k LM}^{(t)}(Q \cdot (r_1, \ldots, r_N)) = \sum_{M'} D_L(Q)_{M'M} h_{i, k LM'}^{(t)}(r_1, \ldots, r_N)}
$$

## Properties and Computational Efficiency

1. **Body Order Expansion**:
   - MACE constructs messages using higher body order expansions, enabling rich representations of atomic environments.

2. **Computational Efficiency**:
   - The use of higher-order messages reduces the required number of message-passing layers to two, enhancing computational efficiency and scalability.

3. **Receptive Field**:
   - MACE maintains a small receptive field by decoupling correlation order increase from the number of message-passing iterations, facilitating parallelization.

4. **State-of-the-Art Performance**:
   - MACE achieves state-of-the-art accuracy on benchmark tasks (rMD17, 3BPA, AcAc), demonstrating its effectiveness in modeling complex atomic interactions.

For further details, refer to the [Batatia et al.](https://arxiv.org/abs/2206.07697).


## Necessary math to know:


### 1. **Spherical Harmonics**

**Concept:**
- Spherical harmonics $Y^L_M$ are functions defined on the surface of a sphere. They are used in many areas of physics, including quantum mechanics and electrodynamics, to describe the angular part of a system.

**Role in MACE:**
- Spherical harmonics are used to decompose the angular dependency of the atomic environment. This helps in capturing the rotational properties of the features in a systematic way.

**Mathematically:**
- The spherical harmonics $Y^L_M(\theta, \phi)$ are given by:

$$
  Y^L_M(\theta, \phi) = \sqrt{\frac{(2L+1)}{4\pi} \frac{(L-M)!}{(L+M)!}} P^M_L(\cos \theta) e^{iM\phi}
$$

where $P^M_L$ are the associated Legendre polynomials.

### 2. **Clebsch-Gordan Coefficients**

**Concept:**
- Clebsch-Gordan coefficients are used in quantum mechanics to combine angular momenta. They arise in the coupling of two angular momentum states to form a new angular momentum state.

**Role in MACE:**
- In MACE, Clebsch-Gordan coefficients are used to combine features from different atoms while maintaining rotational invariance. They ensure that the resulting features transform correctly under rotations, preserving the physical symmetry of the system.

**Mathematically:**
- When combining two angular momentum states  $\vert l_1, m_1\rangle$  and  $\vert l_2, m_2\rangle$, the resulting state  $\vert L, M\rangle$  is given by:

$$

|L, M\rangle = \sum_{m_1, m_2} C_{L, M}^{l_1, m_1; l_2, m_2} |l_1, m_1\rangle |l_2, m_2\rangle

$$

where  $C_{L, M}^{l_1, m_1; l_2, m_2}$  are the Clebsch-Gordan coefficients.

### 3. **$O(3)$ Rotations**

**Concept:**
- The group $O(3)$ consists of all rotations and reflections in three-dimensional space. It represents the symmetries of a 3D system, including operations that preserve the distance between points.

**Role in MACE:**
- Ensuring that the neural network respects $O(3)$ symmetry is crucial for modeling physical systems accurately. MACE achieves this by using operations that are invariant or equivariant under these rotations and reflections.

**Mathematically:**
- A rotation in $O(3)$ can be represented by a 3x3 orthogonal matrix $Q$ such that:

$$
  Q^T Q = I \quad \text{and} \quad \det(Q) = \pm 1
$$

where $I$ is the identity matrix.

### 4. **Wigner D-matrix**

**Concept:**
- The Wigner D-matrix $D^L(Q)$ represents the action of a rotation $Q$ on spherical harmonics. It provides a way to transform the components of a tensor under rotation.

**Role in MACE:**
- Wigner D-matrices are used to ensure that the feature vectors in the neural network transform correctly under rotations. This is essential for maintaining the rotational equivariance of the model.

**Mathematically:**
- For a rotation $Q \in O(3)$ and a spherical harmonic of degree $L$, the Wigner D-matrix $D^L(Q)$ is a $(2L+1) \times (2L+1)$ matrix. If $Y^L_M$ is a spherical harmonic, then under rotation $Q$, it transforms as:

$$
  Y^L_M(Q \cdot \mathbf{r}) = \sum_{M'=-L}^{L} D^L_{M'M}(Q) Y^L_{M'}(\mathbf{r})
$$



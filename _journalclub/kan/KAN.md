- Proposed by Max Tegmark's group. See [Liu et al](https://arxiv.org/abs/2404.19756)
- A hot topic on twitter - a matter of a lot of debate.

The paper "KAN: Kolmogorov–Arnold Networks" proposes Kolmogorov-Arnold Networks (KANs) as an alternative to Multi-Layer Perceptrons (MLPs). The core idea behind KANs is inspired by the Kolmogorov-Arnold representation theorem, which states that any multivariate continuous function can be represented as a sum of continuous functions of one variable. This section will summarize the technical details of the paper, focusing on the mathematical formulations.

### 1. Introduction

The motivation for KANs stems from the limitations of MLPs, such as fixed activation functions on nodes and linear weights. MLPs rely heavily on the universal approximation theorem, but their structure can be less efficient and interpretable. KANs, on the other hand, utilize learnable activation functions on edges and replace linear weights with univariate functions parametrized as splines.

### 2. Kolmogorov-Arnold Representation Theorem

The Kolmogorov-Arnold representation theorem states:

$$
f(x) = \sum_{q=1}^{2n+1} \Phi_q \left( \sum_{p=1}^n \varphi_{q,p}(x_p) \right)
$$

where $\varphi_{q,p} : [0, 1] \to \mathbb{R}$ and $\Phi_q : \mathbb{R} \to \mathbb{R}$.

### 3. KAN Architecture

KANs generalize the representation theorem to arbitrary depths and widths. Each weight parameter in KANs is replaced by a learnable 1D function (spline).

#### 3.1. Mathematical Formulation of KANs

Define a KAN layer with $n_{\text{in}}$-dimensional inputs and $n_{\text{out}}$-dimensional outputs as a matrix of 1D functions:

$$
\Phi = \{ \varphi_{q,p} \}, \quad p = 1, 2, \ldots, n_{\text{in}}, \quad q = 1, 2, \ldots, n_{\text{out}}
$$

Activation function on edge $\varphi_{l,j,i}$ between layer $l$ and $l+1$ is given by:

$$
\varphi_{l,j,i}(x) = w (b(x) + \text{spline}(x))
$$

where $b(x) = \text{silu}(x) = \frac{x}{1 + e^{-x}}$.

The output of each layer is computed as:

$$
x_{l+1, j} = \sum_{i=1}^{n_l} \varphi_{l,j,i}(x_{l,i})
$$

in matrix form:

$$
x_{l+1} = \Phi_l x_l
$$

where $\Phi_l$ is the function matrix of layer $l$.

### 4. Approximation Abilities and Scaling Laws

KANs can approximate functions by decomposing high-dimensional problems into several 1D problems, effectively avoiding the curse of dimensionality.

#### Theorem 2.1: Approximation Bound

Let $f(x)$ be represented as:

$$
f = (\Phi_{L-1} \circ \Phi_{L-2} \circ \cdots \circ \Phi_1 \circ \Phi_0)x
$$

For each $\Phi_{l,i,j}$, there exist $k$-th order B-spline functions $\Phi_{l,i,j}^G$ such that:

$$
\| f - (\Phi_{L-1}^G \circ \Phi_{L-2}^G \circ \cdots \circ \Phi_1^G \circ \Phi_0^G)x \|_{C^m} \leq C G^{-k-1+m}
$$

where $G$ is the grid size and $C$ depends on $f$ and its representation.

### 5. Grid Extension Technique

KANs can increase accuracy by refining the grid used in splines:

$$
\{c'_j\} = \arg\min_{\{c'_j\}} E_{x \sim p(x)} \left( \sum_{j=0}^{G2+k-1} c'_j B'_j(x) - \sum_{i=0}^{G1+k-1} c_i B_i(x) \right)^2
$$

### 6. Simplification Techniques

KANs can be made more interpretable by sparsification, pruning, and symbolification. The L1 norm and entropy regularization can be used to sparsify the network.

### 7. Toy Examples and Empirical Results

KANs were shown to have better scaling laws than MLPs, achieving lower test losses with fewer parameters in various toy datasets and special functions.

#### Example Functions:

1. Bessel function: $f(x) = J_0(20x)$
2. High-dimensional function: $f(x_1, \ldots, x_{100}) = \exp\left( \frac{1}{100} \sum_{i=1}^{100} \sin^2(\pi x_i / 2) \right)$

KANs can achieve near-theoretical scaling exponents $\alpha = 4$, outperforming MLPs in accuracy and parameter efficiency.

### Conclusion

KANs provide a novel approach to neural network design, leveraging the Kolmogorov-Arnold representation theorem to achieve better performance and interpretability compared to traditional MLPs. The use of learnable activation functions on edges and splines allows for greater flexibility and efficiency in function approximation.

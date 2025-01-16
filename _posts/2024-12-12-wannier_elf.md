---
layout: post
title: "ELF, with wannier functions"
date: 2024-12-12 02:14
description: "A worklog implementing calculation of Electron Localization Function (ELF) with wannier functions."
tags: wannier DFT ELF physics
categories: worklog
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

- This is not complete - in the sense that it is something, but no one yet knows if it is useful to have.

- For a primer on wannier functions: please see [this](https://doi.org/10.1103/RevModPhys.96.045008).

- In the event reading the following is not really needed, the implementation is available here:
  [Wannier-ELF](https://github.com/utksi/wannier_elf)

---

### **What, and Why?**

The Electron Localization Function (ELF) is a valuable tool in computational chemistry and condensed matter physics for visualizing and understanding electron localization in atoms, molecules, and solids.

Speaking informally, if one knows the degree of electron localization, a more generalized (as opposed to localized :D) perspective of bonding in the material of interest should be obtained.
It is somewhat easier to think of this when looking at the math:

The standard expression for ELF involves the kinetic energy densities:

$$
\text{ELF}(\mathbf{r}) = \dfrac{1}{1 + \left( \dfrac{ t_P(\mathbf{r}) }{ t_h(\mathbf{r}) } \right)^2}
$$

- $$t_P(\mathbf{r}) = t(\mathbf{r}) - t_W(\mathbf{r})$$: Pauli kinetic energy density.
- $$t(\mathbf{r})$$: Total kinetic energy density.
- $$t_W(\mathbf{r})$$: von Weizsäcker kinetic energy density.
- $$t_h(\mathbf{r})$$: Kinetic energy density of a homogeneous electron gas.

Essentially, it is a three-dimensional scalar field that **tracks the variation in KE density** (Total - Von-Weizsäcker term) at some point $$\mathbf{\vec{r}}$$ in the cell, compared to if one had a homogeneous electron gas with the same electron density, for which the gradient term: $$\nabla n(\mathbf{r})$$ should be exactly zero.

This ratio: $$\dfrac{ t_P(\mathbf{r}) }{ t_h(\mathbf{r}) }$$ is also sometimes called the **localization index** : $$\chi (r)$$.

Given $$n(\mathbf{r})$$: the electron density.

1. **von Weizsäcker Kinetic Energy Density ($t_W(\mathbf{r})$)**

 $$
 t_W(\mathbf{r}) = \dfrac{1}{8} \dfrac{|\nabla n(\mathbf{r})|^2}{n(\mathbf{r})}
 $$

2. **Pauli Kinetic Energy Density ($t_P(\mathbf{r})$)**

 $$
 t_P(\mathbf{r}) = t(\mathbf{r}) - t_W(\mathbf{r})
 $$

3. **Homogeneous Electron Gas Kinetic Energy Density ($t_h(\mathbf{r})$)**

 $$
 t_h(\mathbf{r}) = \dfrac{3}{5} (3\pi^2)^{2/3} [n(\mathbf{r})]^{5/3}
 $$

---

### VASP/CASTEP expressions for ELF

If you're reading this, then it's highly likely that you already know that ELF fields can be written very easily after a obtaining charge density in VASP/CASTEP.

And,
In CASTEP and VASP, the ELF is calculated using an expression involving the Laplacian of the Kohn-Sham orbitals and electron density:

$$
D(\mathbf{r}) = -2A \sum_i \psi_i^*(\mathbf{r}) \nabla^2 \psi_i(\mathbf{r}) + \dfrac{A}{2} \nabla^2 n(\mathbf{r}) - \dfrac{A}{4n(\mathbf{r})} \left( \nabla n(\mathbf{r}) \right)^2
$$

- $$A = \dfrac{\hbar^2}{2m}$$: Constant involving Planck's constant $$\hbar$$ and electron mass $m$.
- The terms represent:
  - **First Term**: Kinetic energy density of the non-interacting Kohn-Sham system.
  - **Second Term**: "Correlation correction."
  - **Third Term**: Kinetic energy density of an ideal Bose gas at the same density.

The ELF is then calculated as:

$$
\text{ELF}(\mathbf{r}) = \dfrac{1}{1 + \left( \dfrac{ D(\mathbf{r}) }{ D_0(\mathbf{r}) } \right)^2}
$$

In the $$\texttt{VASP}$$ source code, $$D(r) = T + T_{corr.} - T_{bos.}$$ could be found, i.e. the same thing.

The numerator in the localization index now looks a bit different here, and the first guess should be that this of course looks like this because the normal expression has been broken down into direct, cross and divergence terms - and one would be correct!

I still find it a tiny bit cathartic to show this explicitly (doing my part against the entropy of the universe).

#### **Reconciling the Expressions**

##### **Step 1: Relate the First Term to Total Kinetic Energy Density**

The first term in the CASTEP/VASP expression:

$$
-2A \sum_i \psi_i^*(\mathbf{r}) \nabla^2 \psi_i(\mathbf{r})
$$

Using the identity:

$$
\psi_i^*(\mathbf{r}) \nabla^2 \psi_i(\mathbf{r}) = \nabla \cdot \left( \psi_i^*(\mathbf{r}) \nabla \psi_i(\mathbf{r}) \right) - |\nabla \psi_i(\mathbf{r})|^2
$$

Substitute back:

$$
-2A \sum_i \psi_i^*(\mathbf{r}) \nabla^2 \psi_i(\mathbf{r}) = 2A \sum_i |\nabla \psi_i(\mathbf{r})|^2 - 2A \sum_i \nabla \cdot \left( \psi_i^*(\mathbf{r}) \nabla \psi_i(\mathbf{r}) \right)
$$

Recognizing that $$A = \dfrac{\hbar^2}{2m}$, the term $$2A \sum_i |\nabla \psi_i(\mathbf{r})|^2$$ corresponds to twice the total kinetic energy density:

$$
2 t(\mathbf{r}) = 2A \sum_i |\nabla \psi_i(\mathbf{r})|^2
$$

##### **Step 2: Relate the Second and Third Terms to von Weizsäcker Kinetic Energy Density**

The second term:

$$
\dfrac{A}{2} \nabla^2 n(\mathbf{r}) = A \nabla \cdot \left( \dfrac{1}{2} \nabla n(\mathbf{r}) \right)
$$

The third term:

$$
- \dfrac{A}{4n(\mathbf{r})} \left( \nabla n(\mathbf{r}) \right)^2 = -2 t_W(\mathbf{r})
$$

since:

$$
t_W(\mathbf{r}) = \dfrac{1}{8} \dfrac{|\nabla n(\mathbf{r})|^2}{n(\mathbf{r})} = \dfrac{A}{4n(\mathbf{r})} |\nabla n(\mathbf{r})|^2
$$

##### **Step 3: Combine Terms**

The total expression becomes:

$$
D(\mathbf{r}) = 2 t(\mathbf{r}) - 2 t_W(\mathbf{r}) - 2A \sum_i \nabla \cdot \left( \psi_i^*(\mathbf{r}) \nabla \psi_i(\mathbf{r}) \right) + A \nabla \cdot \left( \dfrac{1}{2} \nabla n(\mathbf{r}) \right)
$$

Group divergence terms:

$$
D(\mathbf{r}) = 2 \left[ t(\mathbf{r}) - t_W(\mathbf{r}) \right] + \text{Divergence Terms}
$$

Thus, we see that:

$$
D(\mathbf{r}) = 2 t_P(\mathbf{r}) + \text{Divergence Terms}
$$

#### **Equivalence**

- The CASTEP/VASP expression for $D(\mathbf{r})$ essentially represents twice the Pauli kinetic energy density $2 t_P(\mathbf{r})$, up to divergence terms.
- The divergence terms may cancel out or integrate to zero under appropriate boundary conditions but can be significant locally.

---

### Scalar fields with Wannier functions

The first step would be to recognize that we're working with *Maximally localized* wannier functions. As as result, the phase is consistent.

To that end, the solution should be simple.

Given a scalar field $$w_n(r)$$:

$$
n(\mathbf{r}) = \sum_n^{\text{occ}} |\tilde{w}_n(\mathbf{r})|^2
$$

And, calculate the kinetic energy density terms in the same way.

Note that, at the end, we need $$D_h(r)$$ and $$D(r) = \tau - \tau_w(r)$$.

---

### Density and density-gradient from $$w_n(r)$$

In terms of implementation, the electron density and its gradient can be constructed as:

```python
# Process the wannier function.
for i, wf in enumerate(self.wannier_data):
    self.logger.info(
        f"Processing Wannier function {i+1}/{len(self.wannier_data)}"
    )

    if smooth_sigma is not None:
        wf = gaussian_filter(wf, smooth_sigma)

    # No normalization - Wannier functions should already be normalized

    # Accumulate density (e/Å³)
    density += wf**2

    # Calculate gradients (Å^-4)
    grad_wf = self.compute_gradient(wf)

    # Accumulate kinetic energy density (eV/Å³)
    # Using same prefactor as VASP for consistency
    tau += np.sum(grad_wf**2, axis=-1)

    # Accumulate density gradient (e/Å⁴)
    grad_density += 2 * wf[..., np.newaxis] * grad_wf

# Double density for non-spin-polarized system
density *= 2.0
```

---

### Symmetrization of scalar fields $$F(r)$$

**Strong** emphasis needs to be laid on the importance of symmetrization of the charge density and kinetic energy scalar fields derived from wannier functions.
Since the wannier functions are not **symmetry-adapted**, but **maximally-localized**, it matters quite a bit.

```markdown
Hint: Try to disable symmetrization in the code, or relax the constraints from 1e-5 to something higher; say, 1e-1; and see what happens.

See the `symmetrize_field()` function.
```

The following symmetrizations are therefore essential.

```python
# Symmetrize fields
density = self.symmetrize_field(density, "density")
tau = self.symmetrize_field(tau, "kinetic energy density")
grad_density = self.symmetrize_field(grad_density, "density gradient")
```

Here is the symmetrization utility, which can do this both in real and reciprocal space.
`spglib` is used for detecting the lattice symmetry.
It should be obvious that unless one has really dense 3D-scalar fields, accurate symmetrization in real space would be a bad idea.

Symmetrization method based on argument; default is `reciprocal`.

```python
def symmetrize_field(self, field: np.ndarray, field_name: str) -> np.ndarray:
    """
    Symmetrize a field according to crystal symmetry.
    Uses either real space or reciprocal space symmetrization based on initialization.

    Args:
        field: Scalar field with shape (nx, ny, nz) or vector field with shape (nx, ny, nz, 3)
        field_name: Name of field for logging

    Returns:
        symmetrized_field: Field with same shape as input but symmetrized
    """
    if self.symmetrization_method == SymmetrizationMethod.RECIPROCAL:
        return self._reciprocal_symmetrize(field, field_name)
    else:
        return self._real_symmetrize(field, field_name)
```

If we select real space symmetrization:

```python
def _real_symmetrize(self, field: np.ndarray, field_name: str) -> np.ndarray:
    """
    Traditional symmetrization using averaging of symmetry-equivalent points in real space.
    """
    self.logger.info(f"Starting real-space symmetrization of {field_name}")

    # Store original field for validation
    original_field = field.copy()

    # Get the shape and dimensions
    field_shape = field.shape
    spatial_dims = field_shape[:3]
    component_dims = field_shape[3:]  # Empty for scalar field, (3,) for vector
    nx, ny, nz = spatial_dims
    n_ops = len(self.rotations)

    # Create fractional grid coordinates
    grid_points = np.indices(spatial_dims).reshape(3, -1).T / np.array([nx, ny, nz])

    # Initialize array to accumulate field values
    num_points = len(grid_points)
    if component_dims:
        sym_field_values = np.zeros((num_points, n_ops) + component_dims)
    else:
        sym_field_values = np.zeros((num_points, n_ops))

    # Grid for interpolation
    grid = (np.arange(nx), np.arange(ny), np.arange(nz))

    # Reshape field for interpolation
    field_reshaped = field.reshape(spatial_dims + (-1,))

    # Apply symmetry operations
    for i, (rot, trans) in enumerate(zip(self.rotations, self.translations)):
        if (i + 1) % 10 == 0 or i == n_ops - 1:
            self.logger.info(f"Processing symmetry operation {i+1}/{n_ops}")

        # Apply rotation and translation to fractional coordinates
        transformed_coords = (grid_points @ rot.T + trans) % 1.0

        # Convert fractional coordinates to grid indices
        transformed_indices = transformed_coords * np.array([nx, ny, nz])

        # Ensure indices are within the grid
        transformed_indices %= np.array([nx, ny, nz])

        # Interpolate field values at transformed positions
        field_values = []
        for comp in range(field_reshaped.shape[-1]):
            values = interpn(
                grid,
                field_reshaped[..., comp],
                transformed_indices,
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            field_values.append(values)

        # Stack component values
        field_values = np.stack(field_values, axis=-1)

        # If scalar field, squeeze the last dimension
        if field_values.shape[-1] == 1:
            field_values = field_values.squeeze(-1)

        sym_field_values[:, i, ...] = field_values

    # Average over symmetry operations
    sym_field = np.mean(sym_field_values, axis=1)

    # Reshape symmetrized field back to original shape
    sym_field = sym_field.reshape(field_shape)

    # Validate the symmetrized field
    self.validate_field_properties(sym_field, field_name, original_field)

    self.logger.info(f"Completed real-space symmetrization of {field_name}")
    return sym_field
```

If we select reciprocal space symmetrization:

```python
def _reciprocal_symmetrize(self, field: np.ndarray, field_name: str) -> np.ndarray:
    """
    Symmetrization in reciprocal space, similar to VASP's approach.
    """
    self.logger.info(f"Starting reciprocal-space symmetrization of {field_name}")

    # Store original field for validation
    original_field = field.copy()

    # Transform to reciprocal space
    field_reciprocal = self._to_reciprocal_space(field)

    # Get reciprocal lattice vectors
    recip_vecs = self._get_reciprocal_vectors()

    # Get grid dimensions
    nx, ny, nz = field.shape[:3]

    # Create reciprocal space grid
    kx = np.fft.fftfreq(nx) * nx  # Scaled to match grid points
    ky = np.fft.fftfreq(ny) * ny
    kz = np.fft.fftfreq(nz) * nz

    # Create meshgrid of k-points
    kgrid = np.array(np.meshgrid(kx, ky, kz, indexing="ij"))

    # Initialize symmetrized field in reciprocal space
    sym_field_reciprocal = np.zeros_like(field_reciprocal)
    weights = np.zeros_like(field_reciprocal, dtype=float)

    # Apply symmetry operations in reciprocal space
    for i, (rot, trans) in enumerate(zip(self.rotations, self.translations)):
        if (i + 1) % 10 == 0 or i == len(self.rotations) - 1:
            self.logger.info(
                f"Processing symmetry operation {i+1}/{len(self.rotations)}"
            )

        # Rotate k-points
        rot_kgrid = np.einsum("ij,jpqr->ipqr", rot, kgrid)

        # Find corresponding indices in the FFT grid
        indices = np.round(rot_kgrid).astype(int)
        # Apply periodic boundary conditions
        indices = indices % np.array([nx, ny, nz])[:, None, None, None]

        # Compute phase factors from translations
        phase = np.exp(
            -2j * np.pi * np.sum(trans[:, None, None, None] * kgrid, axis=0)
        )

        # Accumulate symmetrized components
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    idx = (
                        indices[0, ix, iy, iz],
                        indices[1, ix, iy, iz],
                        indices[2, ix, iy, iz],
                    )
                    sym_field_reciprocal[ix, iy, iz] += (
                        field_reciprocal[idx] * phase[ix, iy, iz]
                    )
                    weights[ix, iy, iz] += 1.0

    # Average by weights
    mask = weights > 0
    sym_field_reciprocal[mask] /= weights[mask]

    # Transform back to real space
    sym_field = self._to_real_space(sym_field_reciprocal)

    # Ensure result is real
    if not np.allclose(sym_field.imag, 0, atol=1e-10):
        self.logger.warning("Symmetrized field has non-zero imaginary components")
    sym_field = sym_field.real

    # Validate the symmetrized field
    self.validate_field_properties(sym_field, field_name, original_field)

    self.logger.info(f"Completed reciprocal-space symmetrization of {field_name}")
    return sym_field
```

It should be made sure that the integral quantities are conserved before and after symmetrization, in addition to whether the scalar field obey ,symmetrization in different regions: `core`, `bonding`, `interstitial`, because even if the scalar field is sampled uniformly, the constituing wavefunctions/wannier-functions are most definitely not.

So,

```python
# Validate symmetry with spatial analysis
self.validate_symmetry(density, "density")
self.validate_symmetry(tau, "kinetic energy density")
self.validate_symmetry(grad_density, "density gradient")
```

which calls upon the functions that validate symmetry and field properties

```python
def validate_symmetry(self, field: np.ndarray, label: str) -> None:
    """
    Check if field obeys crystal symmetry, with spatial analysis relative to atomic positions.
    """
    self.logger.info(f"Validating symmetry of {label}")

    # Get field shape and dimensions
    field_shape = field.shape
    spatial_dims = field_shape[:3]
    nx, ny, nz = spatial_dims

    # Create fractional grid coordinates
    grid_points = np.indices(spatial_dims).reshape(3, -1).T / np.array([nx, ny, nz])

    # Sample subset of points for validation
    num_points = 1000
    indices = np.random.choice(len(grid_points), size=num_points, replace=False)
    sampled_points = grid_points[indices]

    # For scalar or vector fields, reshape as needed
    field_reshaped = field.reshape(spatial_dims + (-1,))

    # Grid for interpolation
    grid = (np.arange(nx), np.arange(ny), np.arange(nz))

    # Track violations by region
    violations = {
        "core": [],  # Within 1Å of nuclei
        "bonding": [],  # 1-2Å from nuclei
        "interstitial": [],  # >2Å from nuclei
    }
    max_violation = 0.0

    # For each sampled point, check symmetry
    for idx, point in enumerate(sampled_points):
        # Calculate distance to nearest atom
        dist_to_atom = self._compute_distance_to_atoms(
            point * [nx, ny, nz], (nx, ny, nz)
        )

        # Check symmetry violation
        field_values = []
        for rot, trans in zip(self.rotations, self.translations):
            transformed_point = (rot @ point + trans) % 1.0
            transformed_indices = transformed_point * np.array([nx, ny, nz])
            transformed_indices %= np.array([nx, ny, nz])

            values = []
            for comp in range(field_reshaped.shape[-1]):
                value = interpn(
                    grid,
                    field_reshaped[..., comp],
                    transformed_indices[np.newaxis, :],
                    method="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )[0]
                values.append(value)
            field_values.append(values)

        field_values = np.array(field_values)
        max_diff = np.max(np.ptp(field_values, axis=0))
        max_violation = max(max_violation, max_diff)
```

**AND** validating the fields were symmetrized correctly

```python
def validate_field_properties(
    self,
    field: np.ndarray,
    field_name: str,
    original_field: Optional[np.ndarray] = None,
) -> None:
    """
    Validate physical properties of a field.
    """
    # Check for NaN or infinite values
    if np.any(~np.isfinite(field)):
        raise ValueError(f"{field_name} contains NaN or infinite values")

    # Add unit-aware validation
    if field_name == "density":
        if np.any(field < 0):
            self.logger.error(f"Negative values found in {field_name}")
        self.logger.info(
            f"{field_name} range: [{field.min():.6e}, {field.max():.6e}] e/Å³"
        )
    elif field_name == "kinetic energy density":
        if np.any(field < 0):
            self.logger.error(f"Negative values found in {field_name}")
        self.logger.info(
            f"{field_name} range: [{field.min():.6e}, {field.max():.6e}] eV/Å³"
        )

    # Check total integral conservation
    if original_field is not None:
        volume = np.abs(np.linalg.det(self.atoms.cell))
        nx, ny, nz = field.shape[:3]
        dV = volume / (nx * ny * nz)

        total_orig = np.sum(original_field) * dV
        total_new = np.sum(field) * dV

        relative_diff = (
            abs(total_orig - total_new) / abs(total_orig)
            if abs(total_orig) > 1e-10
            else 0.0
        )

        if relative_diff > 1e-6:
            self.logger.warning(
                f"Total {field_name} not conserved after symmetrization. "
                f"Relative difference: {relative_diff:.2e}"
            )
        else:
            self.logger.info(
                f"Total {field_name} conserved after symmetrization. "
                f"Relative difference: {relative_diff:.2e}"
            )
```

---

### Calculating $$\mathrm{ELF(r)}$$

Finally, calculating ELF, which is straightforward:

```python
self.logger.info("Computing ELF...")
# Apply density threshold to avoid numerical issues
density_threshold = 1e-6  # e/Å³
mask = density > density_threshold

# Calculate uniform electron gas kinetic energy density
# Following VASP's approach with same prefactors
D_h = np.zeros_like(density)
D_h[mask] = density[mask] ** (5.0 / 3.0)

# Calculate Pauli kinetic energy term
grad_density_norm = np.sum(grad_density**2, axis=-1)
tau_w = np.zeros_like(density)
tau_w[mask] = grad_density_norm[mask] / (8.0 * density[mask])

# Calculate D = τ - τ_w
D = np.maximum(tau - tau_w, 0.0)

# Initialize ELF array (starting from 0.0, not 0.5)
elf = np.zeros_like(density)

# Compute dimensionless χ = D/D_h
chi = np.zeros_like(density)
chi[mask] = D[mask] / D_h[mask]

# Compute ELF
elf[mask] = 1.0 / (1.0 + chi[mask] ** 2)
```

Using a `threshold`, masking values with `mask` seems to be important for stable values.

---

### Writing scalar fields

```python
self.write_field_xsf("density.xsf", density)
self.write_field_xsf("tau.xsf", tau)
self.write_field_xsf("tau_w.xsf", tau_w)
self.write_field_xsf("D_h.xsf", D_h)
self.write_field_xsf("ELF.xsf", elf)
```

We make use of ASE's write function:

```python
def write_field_xsf(self, filename: str, field: np.ndarray) -> None:
    """Write a field to an XSF file for visualization."""
    self.logger.info(f"Writing field to file: {filename}")
    write(
        filename,
        self.atoms,
        format="xsf",
        data=field,
        origin=self.origin,
        span_vectors=self.span_vectors,
    )
```

---

### ELF obtained from $$w_n(r)$$

A good example is $$\mathrm{CeO_2}$$ where Cerium is supposed to have +4 and not +3 formal oxidation state. So the $$ELF(r)$$ field value near Cerium across all cross sections should be minimal.

<div class="l-page">
<iframe src="/assets/plotly/elf_plot.html" frameborder='0' scrolling='no' height="100%" width="75%" style="border: 1px dashed grey;"></iframe>
</div>

---

### **Conclusion**

Calculating the Electron Localization Function (ELF) using Wannier functions provides a localized perspective on electron localization, anc can potentially offer new insights into chemical bonding and electron pairing.
Normalization, phase alignment, and inclusion of cross terms need to be carefully addressed.
.. *Ongoing* ...

---

### **References**

1. Becke, A. D., & Edgecombe, K. E. (1990). A simple measure of electron localization in atomic and molecular systems. *The Journal of Chemical Physics*, 92(9), 5397–5403.
2. Silvi, B., & Savin, A. (1994). Classification of chemical bonds based on topological analysis of electron localization functions. *Nature*, 371(6499), 683–686.
3. Kohn, W., & Sham, L. J. (1965). Self-consistent equations including exchange and correlation effects. *Physical Review*, 140(4A), A1133.
4. Marzari, N., Mostofi, A. A., Yates, J. R., Souza, I., & Vanderbilt, D. (2012). Maximally localized Wannier functions: Theory and applications. *Reviews of Modern Physics*, 84(4), 1419–1475.
5. Bader, R. F. W. (1990). *Atoms in Molecules: A Quantum Theory*. Oxford University Press.
6. Yang, W., & Parr, R. G. (1985). Hardness, softness, and the Fukui function in the electronic theory of metals and catalysis. *Proceedings of the National Academy of Sciences*, 82(20), 6723–6726.
7. Silvi, B., & Gatti, C. (2000). Electron localization function along a bond and atomic shell structure in solids. *The Journal of Physical Chemistry A*, 104(13), 2627–2635.
8. Henkelman, G., Arnaldsson, A., & Jónsson, H. (2006). A fast and robust algorithm for Bader decomposition of charge density. *Computational Materials Science*, 36(3), 354–360.

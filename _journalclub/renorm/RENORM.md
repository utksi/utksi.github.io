- Via [Badrtdinov, Katsnelson & Rudenko](https://arxiv.org/abs/2406.05229)
- I find this very interesting. There have been recent (relatively) works on introducing spin-lattice interaction in practical implementation(s) of ASD solvers [See the work from Anders Bergman and Anna Delin (2022-2023)], but renormalization is not taken care of in those things explicitly. To get a measure of how much the exchange interactions are affected as a function of temperature independent of explicit electron-phonon interactions, this could be interesting.

## First, a one line (or two :D) introduction to key concepts

**Magnetism in 2D Materials**:
- **2D Magnets** are materials with magnetic properties confined to two dimensions, influenced significantly by quantum effects. They are promising for applications in spintronics, where electronic spins are used to store, process, and transfer information.

**Heisenberg Model**:
- Describes magnetic interactions through pairwise exchange interactions.
- The Hamiltonian:
  $$ H_0 = \sum_{i > j} J_{ij} \mathbf{S}_i \cdot \mathbf{S}_j $$
  where $$ J_{ij} $$ is the exchange interaction between spins $$ \mathbf{S}_i $$ and $$ \mathbf{S}_j $$.

**Electron-Phonon Coupling**:
- Refers to interactions between electrons and lattice vibrations (phonons).
- These interactions affect various electronic properties, including magnetic exchange interactions.

**Green’s Functions and Self-Energy**:
- **Green’s Functions** $$ G^{\sigma}_{ij}(i\omega_n) $$: Describe electron propagation with spin $$ \sigma $$.
- **Self-Energy** $$ \Sigma^{\sigma}_k(i\omega_n) $$: Represents interaction effects on electrons due to phonons and other electrons.

## 1. Theory and Model

The paper extends the Heisenberg model to include electron-phonon interactions, recalculating the exchange interaction $$ J_{ij} $$ using the magnetic force theorem:

$$
J_{ij} = 2 \text{Tr}_{\omega L} \left[ \Delta_i G^{\uparrow}_{ij}(i\omega_n) \Delta_j G^{\downarrow}_{ji}(i\omega_n) \right] S^{-2}
$$

where:
- $$ \Delta_i $$ is the exchange splitting at lattice site $$ i $$.
- $$ G^{\sigma}_{ij}(i\omega_n) $$ is the spin-polarized electron propagator.
- $$ \text{Tr}_{\omega L} $$ denotes the trace over Matsubara frequencies $$ i\omega_n $$ and orbital indices $$ L $$.

Incorporating electron-phonon interactions, the Green’s function is renormalized using the Dyson equation:

$$
G^{-1}_k(i\omega_n) \rightarrow \tilde{G}^{-1}_k(i\omega_n) = G^{-1}_k(i\omega_n) - \Sigma_k(i\omega_n)
$$

This leads to a renormalized exchange splitting:

$$
\Delta \rightarrow \tilde{\Delta}_k(i\omega_n) = \Delta + \Sigma^{\uparrow}_k(i\omega_n) - \Sigma^{\downarrow}_k(i\omega_n)
$$

where the self-energy $$ \Sigma^{\sigma}_k(i\omega_n) $$ is given by:

$$
\Sigma^{\sigma}_k(i\omega_n) = -T \sum_{k' \nu m} G^{\sigma}_{k'}(i\omega_n - i\omega_m) |g^{\nu \sigma}_{kk'}|^2 D_{k-k'}(i\omega_n - i\omega_m)
$$

Here, $$ g^{\nu \sigma}_{kk'} $$ is the electron-phonon coupling vertex, and $$ D_q(i\omega_n) $$ is the phonon propagator.

**Context**: This theoretical framework allows the authors to predict how the electron-phonon interactions influence the magnetic properties of 2D materials by renormalizing the exchange interactions between spins.

## 2. Square Lattice Model

To illustrate the effect, the authors use a square lattice model at half-filling with the Hamiltonian:

$$
H = t \sum_{\langle ij \rangle \sigma} c^{\dagger}_{i\sigma} c_{j\sigma} + \frac{\Delta}{2} \sum_i (n^{\uparrow}_i - n^{\downarrow}_i) + \sum_q \omega_q b^{\dagger}_q b_q + \sum_{q, \langle ij \rangle \sigma} g_q (b^{\dagger}_q + b_{-q}) c^{\dagger}_{i\sigma} c_{j\sigma}
$$

where:
- $$ t $$ is the nearest-neighbor hopping.
- $$ \Delta $$ is the on-site exchange splitting.
- $$ \omega_q $$ is the phonon frequency.
- $$ g_q $$ is the electron-phonon coupling constant.

The self-energy in the high-temperature limit simplifies to:

$$
\Sigma^{\sigma}_k(\omega, T) = 2\lambda \frac{k_BT}{N^{\sigma}_F} \sum_q G^{\sigma}_{k+q}(\omega)
$$

where $$ \lambda $$ is the dimensionless electron-phonon coupling constant.

**Context**: The square lattice model serves as a simple yet effective system to understand the temperature dependence of exchange interactions due to electron-phonon coupling.

## 3. Renormalization of Exchange Interactions

The main result shows that the exchange interaction is renormalized linearly with temperature due to electron-phonon coupling:

$$
J(T) = J(0) - c\lambda T
$$

where $$ c $$ is a renormalization constant.

**Derivation**:
- The linear temperature dependence arises from the self-energy correction, which modifies the exchange interaction strength $$ J_{ij} $$.
- The renormalization constant $$ c $$ is determined by the specific electronic structure of the material and the strength of the electron-phonon coupling.

## 4. Application to $$ \mathrm{Fe_3GeTe_2} $$

For the metallic 2D ferromagnet $$ \mathrm{Fe_3GeTe_2} $$, the authors use first-principles calculations to determine the electronic and phononic structures. The temperature dependence of the exchange interactions is calculated, showing a reduction of the Curie temperature by about 10% due to electron-phonon interactions.

**First-Principles Calculation**:
- **Density Functional Theory (DFT)** is employed to calculate the electronic structure.
- **Density Functional Perturbation Theory (DFPT)** is used for phonon calculations.
- The electronic structure in the vicinity of the Fermi level is parameterized using maximally localized Wannier functions.

**Context**: Wannierization is an invaluable tool for simply writing the hamiltonian in a local basis in the vicinity of the Fermi level, such a model is easier to solve exactly at lower (read: more rigorous) levels of theory as well, which is essential for evaluating the temperature-dependent exchange interactions.

## 5. Spin-Wave Renormalization

**Spin-Wave Theory**:
- Spin waves, or magnons, are collective excitations in a magnetically ordered system.
- The stability of magnetic order is influenced by spin-wave spectra, which can be calculated by diagonalizing the spin-wave Hamiltonian.

The Heisenberg model with single-ion anisotropy (SIA) is used to describe the spin waves:

$$
H = H_0 + A \sum_i (S^z_i)^2
$$

where $$ A $$ is the anisotropy parameter.

**Magnon Eigenvectors and Spectra**:
- Magnon frequencies $$ \Omega_{q \nu} $$ are obtained by diagonalizing the spin-wave Hamiltonian:

$$
H^{\text{SW}}_{\mu \nu}(q) = \left[ \delta_{\mu \nu} \left( 2A \Phi + \sum_{\chi} J_{\mu \chi}(0) \right) - J_{\mu \nu}(q) \right] \langle S^z \rangle
$$

where:
- $$ J_{\mu \nu}(q) $$ are the Fourier transforms of the exchange interaction matrix.
- $$ \Phi = 1 - \left( 1 - \langle S^2_z \rangle / 2 \right) $$ is the Anderson-Callen decoupling factor for $$ S = 1 $$.

The magnon spectra exhibit optical and acoustic branches. Near the $$ \Gamma $$ point, the acoustic branch disperses quadratically:

$$
\Omega_q \approx \Omega_0 + Dq^2
$$

where $$ D $$ is the spin-stiffness constant and $$ \Omega_0 $$ is the gap due to single-ion anisotropy.

**Temperature-Dependent Magnetization**:
- Magnetization $$ \langle S^z \rangle $$ is determined by spin-wave excitations, using the Tyablikov decoupling (RPA):

$$
\langle (S^z_i)^n S^-_i S^+_i \rangle = \langle [S^+_i, (S^z_i)^n S^-_i] \rangle \sum_{q \nu} \langle b^{\dagger}_{q \nu} b_{q \nu} \rangle
$$

where $$ \langle b^{\dagger}_{q \nu} b_{q \nu} \rangle = [\exp(\Omega_{q \nu} / k_BT) - 1]^{-1} $$ is the equilibrium magnon distribution.

By solving these equations self-consistently, the renormalized exchange interactions $$ J(T) $$ are used to calculate the Curie temperature $$ T_C $$. The renormalized interactions lead to a reduced $$ T_C $$ compared to the non-renormalized case.

**Context**: The detailed calculation of spin-wave spectra and their temperature dependence provides insight into the stability of magnetic order and how it is influenced by electron-phonon interactions.

## Discussion

The discussion highlights several key points:

1. **Adiabatic vs. Antiadiabatic Electron-Phonon Coupling**:
   - Most systems can be treated adiabatically, where phonon energies are much smaller than electron energies.
   - For stronger renormalization effects, systems with narrow electron bands or high phonon energies relative to electron energies need to be considered.

2. **Out-of-Equilibrium Effects**:
   - Non-equilibrium distributions, such as those induced by charge currents or laser fields, can enhance electron-phonon coupling.
   - This can lead to significant changes in exchange interactions and magnetic properties.

3. **Anisotropic Magnetic Interactions**:
   - The study focuses on isotropic exchange interactions, but anisotropic interactions, such as Dzyaloshinskii-Moriya interaction (DMI), can exhibit stronger renormalization effects.

**Conclusion**: The study demonstrates that electron-phonon coupling significantly affects the magnetic properties of 2D metallic magnets by renormalizing the exchange interactions. This renormalization leads to a suppression of magnetic ordering temperatures and modifies the magnon spectra, which has implications for the design and application of 2D magnetic materials in technology.

### One (or two :D) line mathematical context

**Magnetic Force Theorem**:
- The exchange interaction $$ J_{ij} $$ is derived from the magnetic force theorem, which involves calculating the energy cost of rotating spins $$ \mathbf{S}_i $$ and $$ \mathbf{S}_j $$.
- The exchange interaction is given by the integral over the Brillouin zone of the product of the spin-resolved Green's functions and exchange splitting.

**Dyson Equation**:
- The renormalization of Green’s functions due to self-energy $$ \Sigma_k(i\omega_n) $$ is given by the Dyson equation.
- This renormalizes both the propagators and the exchange splitting.

**Electron-Phonon Self-Energy**:
- The self-energy $$ \Sigma^{\sigma}_k(i\omega_n) $$ represents the correction to the electron’s energy due to its interaction with phonons.
- The self-energy is calculated using second-order perturbation theory in the electron-phonon coupling.

**Spin-Wave Theory**:
- The spin-wave Hamiltonian is derived by expanding the Heisenberg Hamiltonian to second order in spin deviations.
- Diagonalizing the resulting Hamiltonian gives the magnon eigenvalues (frequencies) and eigenvectors.

**Temperature-Dependent Magnetization**:
- The magnetization $$ \langle S^z \rangle $$ is obtained using the Tyablikov decoupling method, which approximates the thermal averages of spin operators.
- The self-consistent solution of the magnetization equations provides the temperature dependence of $$ \langle S^z \rangle $$ and $$ J(T) $$.

### Conclusion

I haven't really thought of more implications, not further than what the authors imply.


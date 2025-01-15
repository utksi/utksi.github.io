---
layout: post
title: "[Stuff I read] Dark states of electrons in a quantum system with two pairs of sublattices"
date: 2024-09-13 12:59:44
description: "A high-level summary of main ideas from Chung et al."
tags: physics spectroscopy materials-science
categories: stuff-i-read
giscus_comments: false
related_posts: false
toc: true
---

- A slightly different way of looking at dark bands in the BZ. See [Chung et al](https://doi.org/10.1038/s41567-024-02586-x)

This review discusses the discovery of **condensed-matter dark states** in the material **Palladium Diselenide (PdSe₂)**, where entire bands of quantum states in the Brillouin zone remain undetectable via angle-resolved photoemission spectroscopy (ARPES). The dark states arise due to interference effects between sublattices, a novel feature in quantum materials.

---

## 1. Introduction to Dark States

In quantum mechanics, a **dark state** refers to a state that cannot absorb or emit photons and is thus undetectable through typical spectroscopic methods. These states are typically well-known in atomic and molecular systems where they arise due to **quantum interference** or **conservation of angular momentum**. 

In the condensed matter context, dark states have been less explored, especially when caused by interference between **sublattices**. This paper expands the concept to solid-state systems, where dark states emerge due to destructive interference within the crystal’s sublattices. These states remain hidden from ARPES measurements because their transition matrix elements vanish.

### Key Definitions:
- **Dark State**: A quantum state that does not interact with light and is therefore undetectable by traditional spectroscopic methods.
- **Quantum Interference**: The phenomenon where the probability amplitudes of quantum states add or cancel out, affecting the visibility of quantum transitions.

---

## 2. Crystal Structure and Sublattice Symmetry

PdSe₂ is chosen for its crystal structure, which consists of two pairs of **sublattices** labeled A, B, C, and D. These sublattices are related by **glide-mirror symmetries**, which leads to specific **quantum phases** that control the interference patterns of electronic wavefunctions in the Brillouin zone.

Mathematically, the electronic structure of PdSe₂ is described using the **tight-binding Hamiltonian** model. The dominant states arise from **Pd 4d orbitals**, and the relative phases $ \varphi_{AB}, \varphi_{AC}, \varphi_{AD} $ between sublattices dictate whether the interference is constructive or destructive.

$$
H_{PdSe_2} = 
\begin{pmatrix}
f_{AA} & f_{AB} & f_{AC} & f_{AD} \\
f_{AB} & f_{AA} & f_{AC} & f_{AD} \\
f_{AC} & f_{AD} & f_{AA} & f_{AB} \\
f_{AD} & f_{AC} & f_{AB} & f_{AA}
\end{pmatrix}
$$

The key discovery here is that PdSe₂ has a unique sublattice arrangement where **multiple glide-mirror symmetries** connect these sublattices, resulting in **double destructive interference** under certain conditions, leading to the appearance of **dark states**. 

### Key Definitions:
- **Sublattice**: A subset of atoms within a crystal lattice that repeats in a regular pattern.
- **Glide-Mirror Symmetry**: A symmetry operation combining a reflection with a translation.
- **Tight-Binding Hamiltonian**: A mathematical model used to describe the movement of electrons in a material by considering the hopping between atoms.

---

## 3. Angle-Resolved Photoemission Spectroscopy (ARPES), Light Polarization, and Dark States

The experimental technique used in the paper, **ARPES**, allows researchers to probe the electronic band structure of a material. However, in PdSe₂, an entire band of electronic states in the Brillouin zone is **invisible** regardless of the photon energy or light polarization used. This is a clear indicator of **dark states** resulting from **sublattice interference**.

The transition probability in ARPES is governed by **Fermi’s Golden Rule**:

$$
M_k = \int \psi_f^* (\mathbf{A} \cdot \mathbf{p}) \psi_i \, dV
$$

Where:
- $ \mathbf{A} $ is the electromagnetic vector potential.
- $ \mathbf{p} $ is the momentum operator.
- $ \psi_i $ and $ \psi_f $ are the initial and final electronic states.

The critical point in PdSe₂ is that the **interference between sublattices** can lead to **destructive interference** when certain relative quantum phases $ \varphi_{AB}, \varphi_{AC}, \varphi_{AD} $ cancel out the matrix elements $ M_k $. This makes some states completely **undetectable by ARPES**.

The experimental data shows that with **p-polarized light**, only one of the **nine cuboidal Brillouin zones** exhibits detectable valence bands (centered at Γ₆). However, when using **s-polarized light**, even this valence band disappears, as shown in the data taken under identical experimental conditions.

In summary:
- With **p-polarized light**, constructive interference allows detection of the 000 state.
- With **s-polarized light**, all pseudospin states vanish due to destructive interference.
- Bands centered at $ \Gamma_{106}, \Gamma_{101}, \Gamma_{016} $ in the kx and ky directions are **not observed** regardless of the polarization.

This clearly indicates the existence of **dark states** in PdSe₂, which are **undetectable** at any photon energy or light polarization due to **double destructive interference** in these sublattices.

### Key Definitions:
- **ARPES**: A technique used to observe the energy and momentum distribution of electrons in a material, providing insights into its electronic structure.
- **Fermi’s Golden Rule**: A formula that calculates the transition probability per unit time for a quantum system interacting with an external perturbation.
- **p-polarized light**: Light in which the electric field oscillates parallel to the plane of incidence.
- **s-polarized light**: Light in which the electric field oscillates perpendicular to the plane of incidence.
  
---

## 4. Phase Polarization and Quantum States in the Brillouin Zone

A major finding in this paper is the identification of **phase polarization** in the Brillouin zone of PdSe₂. The electronic wavefunctions in PdSe₂ are fully polarized to one of four possible states: **000, 0ππ, π0π, ππ0**, depending on the relative quantum phases $ \varphi_{AB}, \varphi_{AC}, \varphi_{AD} $.

- The **000 state** (blue pseudospin) is **visible** in ARPES under **p-polarized light**, because constructive interference ensures a non-zero matrix element.
- The other states, **0ππ, π0π, and ππ0** (red, yellow, and green pseudospins), are **dark states** because two of the three quantum phases are $ \pi $, leading to **double destructive interference**. These states are completely undetectable by ARPES under **any light polarization**.

The phase polarization forms a **checkerboard pattern** in momentum space, where each region of the Brillouin zone is polarized to one of these four states. The **dark states** correspond to areas where double destructive interference occurs, making the electronic states invisible in ARPES measurements.

### Key Definitions:
- **Pseudospin**: An abstract concept used to describe two-level quantum systems, often associated with sublattices or quantum states.
- **Brillouin Zone**: The fundamental region of reciprocal space in a crystal, within which the electronic wavefunctions are defined.

---

## 5. Generalization to Other Materials

The paper also generalizes the findings on dark states to other material systems with similar sublattice structures. This includes:
- **Cuprates**: In high-temperature superconductors such as **Bi2201**, shadow bands have been observed in ARPES that cannot be explained by typical band theory. The paper demonstrates that these shadow bands are **dark states**, undetectable due to sublattice interference.
    - For cuprates, the two nearly degenerate Fermi surfaces (FS1 and FS2) show that segments of the Fermi surface polarized to **000 states** are visible in ARPES with **p-polarized light**, while segments polarized to **0ππ states** are visible with **s-polarized light**.
    - However, parts of the Fermi surface polarized to **π0π** and **ππ0** remain undetectable due to dark states.
  
- **Lead Halide Perovskites**: In **CsPbBr₃**, ARPES measurements show that two distinct valence bands (VB1 and VB2) are observed depending on the photon energy used (kz = $ \Gamma_7 $ for VB1, kz = $ \Gamma_8 $ for VB2). However, only specific bands appear under **p-polarized light**, and both bands vanish under **s-polarized light**. This behavior is explained as a result of **dark states** in the perovskite's sublattice structure.

These findings demonstrate that the phenomenon of **dark states** is not limited to PdSe₂ but is a **universal feature** in materials with two pairs of sublattices connected by **glide-mirror symmetries**.

### Key Definitions:
- **Shadow Bands**: Bands in the ARPES data of cuprates that appear with lower intensity or are completely undetectable due to their sublattice interference.
- **Band Folding**: A phenomenon in ARPES where multiple bands appear due to the periodicity of the crystal lattice, often related to structural distortions or superlattice formations.

---

## 6. Novel Contributions and Impact of the Paper

The paper makes several important novel contributions to the field of condensed matter physics:

1. **Discovery of Condensed-Matter Dark States**: This study is the first to report the observation of dark states in a solid-state material (PdSe₂). These dark states are completely undetectable in ARPES due to destructive interference between wavefunctions from different sublattices. This represents a new class of quantum states in condensed matter physics that had previously only been studied in atomic and molecular systems.

2. **Mechanism of Sublattice Interference**: The authors introduce a rigorous theoretical framework that explains how the interference between sublattices, connected by **glide-mirror symmetries**, leads to the emergence of dark states. The framework is supported by first-principles calculations and the **tight-binding Hamiltonian** model, which accurately describes the band structure and phase relationships in PdSe₂.

3. **Generalization to Other Systems**: The study extends the concept of dark states to other materials, such as **cuprates** and **lead halide perovskites**, providing a unified explanation for previously unexplained ARPES data in these systems. For example, the **shadow bands** in cuprates and the **vanishing bands** in perovskites can now be understood as the result of sublattice interference and dark states.

4. **Light Polarization Effects**: One of the most striking aspects of the paper is the detailed discussion of how **light polarization** affects the visibility of electronic states in ARPES. The authors show that:
   - **p-polarized light** can detect certain quantum states (e.g., the 000 state in PdSe₂).
   - **s-polarized light** renders all states invisible due to complete destructive interference.
   This effect has profound implications for future experimental studies using ARPES, as it highlights the need to carefully control light polarization to detect or suppress specific quantum states.

### Why This Paper is Impressive

This paper is highly impressive for several reasons:
- **Novel Concept**: The generalization of **dark states** to condensed matter systems is a significant breakthrough. It opens up a new avenue of research for understanding hidden quantum states in materials that were previously inaccessible to experimental probes like ARPES.
- **Comprehensive Approach**: The combination of sophisticated experimental techniques (e.g., ARPES) with detailed theoretical modeling (tight-binding calculations and symmetry analysis) makes this paper a comprehensive and authoritative study on the topic.
- **Broad Implications**: The findings are not limited to PdSe₂ but extend to a wide range of materials with sublattice structures. This could have far-reaching implications for the study of high-temperature superconductors, optoelectronic materials, and other quantum systems.
- **Resolution of Long-Standing Puzzles**: By explaining phenomena such as the shadow bands in cuprates and the vanishing bands in perovskites, the paper resolves several long-standing puzzles in the field of condensed matter physics.

In summary, the discovery of condensed-matter dark states, the detailed analysis of sublattice interference, and the generalization to other material systems make this paper a landmark contribution to the study of quantum materials.

---

### Key Definitions (Recap):
- **Dark State**: A quantum state that does not interact with light, making it undetectable in spectroscopic experiments.
- **Sublattice Interference**: The interaction between wavefunctions from different sublattices, leading to constructive or destructive interference.
- **p-polarized light**: Light whose electric field oscillates parallel to the plane of incidence, often used to detect quantum states in ARPES.
- **s-polarized light**: Light whose electric field oscillates perpendicular to the plane of incidence, which can cause destructive interference in certain quantum states.
- **Fermi’s Golden Rule**: A quantum mechanical formula used to calculate the transition probability of electrons between states when interacting with light.


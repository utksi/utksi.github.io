---
layout: post
title: "[Stuff I read] Dark states of electrons in a quantum system with two pairs of sublattices"
date: 2024-09-13 12:59:44
description: "A high-level summary of main ideas from Chung et al."
tags: physics spectroscopy materials-science
categories: journal-club
giscus_comments: true
related_posts: false
toc: true
---

- A slightly different way of looking at dark bands in the Brillouin Zone (BZ). See [Chung et al.](https://doi.org/10.1038/s41567-024-02586-x).

This review discusses the discovery of **condensed-matter dark states** in the material **Palladium Diselenide (PdSe₂)**, where entire bands of quantum states in the Brillouin zone remain undetectable via angle-resolved photoemission spectroscopy (ARPES). The dark states arise due to interference effects between sublattices, a novel feature in quantum materials.

---

## 1. Introduction to Dark States

In quantum mechanics, a **dark state** refers to a state that cannot absorb or emit photons and is thus undetectable through typical spectroscopic methods. These states are typically well-known in atomic and molecular systems where they arise due to **quantum interference** or **conservation of angular momentum**.

In the condensed-matter context, dark states have been less explored, especially when caused by interference between **sublattices**. This paper expands the concept to solid-state systems, where dark states emerge due to destructive interference within the crystal’s sublattices. These states remain hidden from ARPES measurements because their transition matrix elements vanish.

### Key Definitions:

- **Dark State**: A quantum state that does not interact with light and is therefore undetectable by traditional spectroscopic methods.
- **Quantum Interference**: The phenomenon where the probability amplitudes of quantum states add or cancel out, affecting the visibility of quantum transitions.

---

## 2. Crystal Structure and Sublattice Symmetry

PdSe₂ is chosen for its crystal structure, which consists of two pairs of **sublattices** labeled \( A \), \( B \), \( C \), and \( D \). These sublattices are related by **glide-mirror symmetries**, which leads to specific **quantum phases** that control the interference patterns of electronic wavefunctions in the Brillouin zone.

Mathematically, the electronic structure of PdSe₂ is described using the **tight-binding Hamiltonian** model. The dominant states arise from **Pd 4d orbitals**, and the relative phases \( \varphi*{AB}, \varphi*{AC}, \varphi\_{AD} \) between sublattices dictate whether the interference is constructive or destructive:

\[
H*{PdSe_2} =
\begin{pmatrix}
f*{AA} & f*{AB} & f*{AC} & f*{AD} \\
f*{AB} & f*{AA} & f*{AC} & f*{AD} \\
f*{AC} & f*{AD} & f*{AA} & f*{AB} \\
f*{AD} & f*{AC} & f*{AB} & f\_{AA}
\end{pmatrix}
\]

The key discovery is that PdSe₂ has a unique sublattice arrangement where **multiple glide-mirror symmetries** connect these sublattices, resulting in **double destructive interference** under certain conditions, leading to the appearance of **dark states**.

### Key Definitions:

- **Sublattice**: A subset of atoms within a crystal lattice that repeats in a regular pattern.
- **Glide-Mirror Symmetry**: A symmetry operation combining a reflection with a translation.
- **Tight-Binding Hamiltonian**: A mathematical model used to describe the movement of electrons in a material by considering the hopping between atoms.

---

## 3. ARPES, Light Polarization, and Dark States

The experimental technique used in the paper, **ARPES**, allows researchers to probe the electronic band structure of a material. However, in PdSe₂, an entire band of electronic states in the Brillouin zone is **invisible** regardless of the photon energy or light polarization used. This is a clear indicator of **dark states** resulting from **sublattice interference**.

The transition probability in ARPES is governed by **Fermi’s Golden Rule**:

\[
M_k = \int \psi_f^\* (\mathbf{A} \cdot \mathbf{p}) \psi_i \, dV
\]

Where:

- \( \mathbf{A} \) is the electromagnetic vector potential.
- \( \mathbf{p} \) is the momentum operator.
- \( \psi_i \) and \( \psi_f \) are the initial and final electronic states.

The critical point in PdSe₂ is that the **interference between sublattices** can lead to **destructive interference** when certain relative quantum phases \( \varphi*{AB}, \varphi*{AC}, \varphi\_{AD} \) cancel out the matrix elements \( M_k \). This makes some states completely **undetectable by ARPES**.

---

## 4. Phase Polarization and Brillouin Zone Quantum States

A major finding in this paper is the identification of **phase polarization** in the Brillouin zone of PdSe₂. The electronic wavefunctions in PdSe₂ are fully polarized to one of four possible states: \( 000, 0\pi\pi, \pi0\pi, \pi\pi0 \), depending on the relative quantum phases \( \varphi*{AB}, \varphi*{AC}, \varphi\_{AD} \).

- The **000 state** (blue pseudospin) is **visible** in ARPES under **p-polarized light**, because constructive interference ensures a non-zero matrix element.
- The other states, \( 0\pi\pi, \pi0\pi, \pi\pi0 \) (red, yellow, and green pseudospins), are **dark states** because two of the three quantum phases are \( \pi \), leading to **double destructive interference**. These states are completely undetectable by ARPES under **any light polarization**.

---

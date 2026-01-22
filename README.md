# Inverse Problems in Disordered Systems (PhD Thesis)

Author: Shardul S. Mukim  
Institution: School of Physics, Trinity College Dublin  
Year: 2023  
Document: `shardul_mukim_thesis.pdf`

## Abstract (short)
This thesis studies quantum inverse problems: inferring the Hamiltonian or disorder profile of a device from electronic transport signatures. The work develops inversion techniques that estimate disorder concentration and, in a spatially resolved form, map where scatterers are located. The methods are aimed at disordered quantum devices and nanowire/nanosheet networks and are designed to be computationally tractable compared to brute-force searches.

## Thesis outline
1. **Introduction**
   - Motivation, quantum devices, nanomaterial networks, and thesis structure.
2. **Mathematical Methods**
   - Green functions, density of states, Dyson equation, recursive algorithms, transport, and tight-binding models (including graphene nanoribbons).
3. **Disorder information from conductance signature**
   - Forward modelling, inversion via misfit functions, applications to graphene and hBN, DFT-based tight-binding, and error analysis.
4. **Sudoku Problem**
   - Multi-terminal setup, multi-terminal misfit, spatial mapping (4/9/16-cell partitions), generalized partitions, geometry effects, and error analysis.
5. **Junction properties of nanomaterial networks**
   - Graph/network models (adjacency matrix, Kirchhoff circuits), AC impedance, forward modelling, and inversion via misfit functions.
6. **Summary and Future work**
   - Thesis summary, ongoing projects (e.g., zigzag GNR edge disorder), and future directions (magnetic/thermal properties).

## Publications resulting from this work
- S. Mukim, F. P. Amorim, A. R. Rocha, R. B. Muniz, C. Lewenkopf, M. S. Ferreira, "Disorder information from conductance: a quantum inverse problem," Phys. Rev. B 102, 0755409 (2020).
- F. R. Duarte, S. Mukim, A. Molina-Sanchez, T. G. Rappoport, M. S. Ferreira, "Decoding the DC and optical conductivities of disordered MoS2 films: an inverse problem," New Journal of Physics 23 (2021).
- S. Mukim, J. O'Brien, M. Abarashi, M. S. Ferreira, C. G. Rocha, "Decoding the conductance of disordered nanostructures: a quantum inverse problem," J. Phys.: Condens. Matter 34, 085901 (2022).
- S. Mukim, C. Lewenkopf, M. S. Ferreira, "Spatial mapping of disordered 2D materials: the conductance Sudoku," Carbon 188, 360 (2022).
- S. Mukim, M. Kucukbas, S. R. Power, M. S. Ferreira, "Characterization of zigzag edged Graphene nanoribbons using spin current," in preparation.

## Repository structure
- `projects/`: per-project workspaces. Within subfolders, files are organized into `data/` (csv/dat), `scripts/` (py/m/nb), `pdfs/`, and `figures/`.
- `code/`: shared or standalone source code.
- `data/`: shared datasets.
- `figures/`: shared figures and plots.
- `notebooks/`: analysis notebooks.
- `pdfs/`: papers and reference PDFs.
- `presentations/`: slides and talks.
- `archives/`: older or archived material.
- `cleanup_report.txt`: summary report of the organization pass.

## Appendix
- Interpolation method for configurational averaging.

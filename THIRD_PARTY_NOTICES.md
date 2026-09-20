# Third-Party Notices

This repository contains both project-original code and third-party material. The top-level Apache License 2.0 applies to project-original material unless otherwise noted. Third-party material retains its upstream copyright and license terms.

## Paulihedral

The `Paulihedral/` directory originates from the artifact accompanying:

Gushu Li et al., **"Paulihedral: A Generalized Block-Wise Compiler Optimization Framework For Quantum Simulation Kernels,"** ASPLOS 2022.

- Paper: https://arxiv.org/abs/2109.03371
- Artifact DOI: https://doi.org/10.5281/zenodo.5780204
- Upstream license: **Apache License 2.0**

The upstream artifact documentation identifies the Paulihedral implementation as Apache-2.0 licensed. Files in `Paulihedral/` are not originally authored by Yi Hu and remain subject to the upstream copyright, attribution, and license requirements.

This includes, among other files:

- `Paulihedral/synthesis_FT.py`
- `Paulihedral/synthesis_SC.py`
- `Paulihedral/synthesis_sd.py`
- `Paulihedral/parallel_bl.py`
- `Paulihedral/qubit_place.py`
- `Paulihedral/comparision.py`
- `Paulihedral/benchmark/*`
- associated Paulihedral benchmark assets

## Duplicated Paulihedral benchmark material

Some benchmark utilities and data outside `Paulihedral/` are identical to files in the Paulihedral artifact. Confirmed examples include:

- `benchmark/mypauli.py`
- selected files under `data/`, including molecular and UCCSD benchmark data

These files should be treated as Paulihedral-derived material under Apache License 2.0, rather than as project-original code.

## Project-original material

Unless otherwise documented, project-specific implementation such as `CircuitConstruct.py`, `CircuitOptimize.py`, project-specific portions of `HamiltonianGenerator.py`, and `main.py` is licensed under the repository's Apache License 2.0.

External software imported as a dependency remains subject to its own upstream license.

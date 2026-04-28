# Cantilever

Python software to conduct bridged comparison analysis. Direct comparisons between different actions (eg, exposures, 
treatments, interventions) are not always feasible due to practical or ethical constraints. Bridged comparisons offer
an alternative where different actions can be compared across two different studies through an action that is shared
by both trials. More broadly, bridged comparisons are an individual-level data method for indirect comparisons with
either randomized trials or observational studies.

NOTE: this is an initial alpha release. The provided functions are still undergoing active development. Thus results 
and API may change. Currently, this software should only be used for exploratory purposes.


## Installation

Currently, `cantilever` must be installed manually from GitHub. It is not currently available on PyPI. To install,
first clone the library to your computer, open your terminal, navigate to the downloaded `Cantilever/` folder, and 
then run

```commandline
python -m pip install . 
```

Dependencies: `numpy`, `scipy`, `pandas`, `delicatessen`, `matplotlib`, `formulaic`


## Getting Started

See `docs/Example/` for illustrative applications with publicly-available data.


## Acknowledgements

Development of this software was supported by the National Institute of Allergy and Infectious Diseases (NIAID) through
K01-AI177102 (PI: Paul Zivich). The contents of this software is solely the responsibility of the author and does not 
necessarily represent the official views of the National Institutes of Health.


## References

Shook‐Sa BE, Zivich PN, Rosin SP, Edwards JK, Adimora AA, Hudgens MG, & Cole SR. (2024). Fusing trial data for 
treatment comparisons: Single vs multi‐span bridging. *Statistics in Medicine*, 43(4), 793-815.

Zivich PN, Cole SR, Edwards JK, Shook-Sa BE, Breskin A, & Hudgens MG. (2025). Bridged treatment comparisons: an 
illustrative application in HIV treatment. *American Journal of Epidemiology*, 194(6), 1687-1694.

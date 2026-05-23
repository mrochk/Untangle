![Logo](logo.png)

# Untangle

Fast tensor decoupling in Jax. Collection of algorithms for decoupling multivariate functions using tensor decompositions.

This project was built using `uv` (https://docs.astral.sh/uv). 

### Installation

You can easily get `untangle` from PyPI:
```bash
pip install tensor-decoupling # (untangle was already taken...)
```
Otherwise, for a local `uv` installation:
```bash
git clone git@github.com:mrochk/untangle.git
uv add untangle
```

### Algorithms

- Basic Tensor Decoupling: `untangle/algorithm/basic`
- Polynomial Constrained Tensor Decoupling: `untangle/algorithm/ctd_polynomial`
- CMTF-BSD: `untangle/algorithm/cmtf_bspline`
- CMTF-PSD: `untangle/algorithm/cmtf_pspline`

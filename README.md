![Logo](logo.png)

# Untangle

Fast tensor decoupling in Jax. Collection of algorithms for decoupling multivariate functions using tensor decompositions.

This project was built using `uv` (https://docs.astral.sh/uv). 

### Installation

You can easily get `untangle` from PyPI:
```bash
pip install decoupling # ("untangle" was already taken...)
```
Otherwise, for a local `uv` installation:
```bash
git clone git@github.com:mrochk/untangle.git
uv add ./untangle # or pip install ./untangle
```

### Algorithms

- Polynomial Tensor Decoupling: `untangle/algorithm/basic` [Dreesen, Ishteva & Schoukens (2015)]
- Constrained Polynomial TD: `untangle/algorithm/ctd_polynomial` [Hollander, (2017)]
- CMTF B-Spline Decoupling: `untangle/algorithm/cmtf_bspline` [De Jonghe & Ishteva (2025)]
- CMTF P-Spline Decoupling: `untangle/algorithm/cmtf_pspline`

### Testing

```bash
uv run -m unittest discover testing -v # or ./test.sh
```
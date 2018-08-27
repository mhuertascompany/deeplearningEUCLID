# Estimating galaxy sizes with deep learning

The goal of this exercice is to set up a CNN to estimate the effective radii of galaxies. This is based on the [Tuccillo+18](http://adsabs.harvard.edu/abs/2018MNRAS.475..894T). 

## Available Material

You have two different datasets to play with:
- A set of simulated Sersic profiles with GalSim (including noise and PSF effects - H band). There is one galaxy per stamp (128x128 pixels). For each galaxy the effective radius is known by definition and it is included in the associated catalog (one entry per galaxy).
- A set of real galaxies observed with HST (CANDELS survey). For these galaxies the true size is obcvioulsy not known but we do have measurements done with an independent software (GALFIT - [van der Wel+13](http://adsabs.harvard.edu/abs/2012ApJS..203...24V)) 

Both datasets can be downloaded [here](https://wetransfer.com/downloads/2d600b62c6e885014037817a700637c020180827014433/e7dc2fbff7d1785c919473f7acc9afdc20180827014433/7cd5b8).

In order to make things easier, a jupyter notebook including the basic steps is provided. You should use this as a starting seed and modify it to your needs.

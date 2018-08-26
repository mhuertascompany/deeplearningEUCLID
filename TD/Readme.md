# Estimating galaxy sizes with deep learning

The goal of this exercice is to set up a CNN to estimate the effective radii of galaxies. This is based on the [this paper](http://adsabs.harvard.edu/abs/2018MNRAS.475..894T). 

## Available Material

You have two different datasets to play with:
- A set of simulated Sersic profiles with GalSim (including noise and PSF effects - H band). There is one galaxy per stamp (128x128 pixels). For each galaxy the effective radius is known by definition and it is included in the associated catalog (one entry per galaxy).
- A set of real galaxies observed with HST (CANDELS survey). For these galaxies the true size is obcvioulsy not known but we do have measurements done with an independent software (GALFIT) 

Both datasets can be downloaded [here](https://wetransfer.com/downloads/ff968cee9fb2561a75272cf3bb9a412020180826210439/94abb1d971b792f31e482a2a302a0e2320180826210439/ccb794?utm_campaign=WT_email_tracking&utm_content=general&utm_medium=download_button&utm_source=notify_recipient_email).

In order to make things easier, a jupyter notebook including the basic steps is provided. You should use this as a starting seed and modify it to your needs.

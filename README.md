# SSjags

### Automatically Construct and Fit a State-Space Model in 'JAGS' through 'jagsUI'

Tools are provided for constructing model code in the JAGS syntax corresponding 
to a State-space model, and optionally running that model with a user-supplied 
vector of time series data.  

Currently an integrated random walk will be constructed, with options to include
deterministic or stochastic cyclic component(s) as well as an AR(1) disturbance.

### Commonly-used functions

* `runSS()` constructs and optionally runs the JAGS model, and contains inputs 
for the data set used, model specification, and MCMC arguments.

* `plot_components()` produces a plot of the respective components of a State-space 
model.  Possible components include Trend, Cycle (stochastic and/or deterministic), 
AR(1), and Irregular.  Model components will be plotted as posterior envelopes, 
with default credible interval widths of 50 percent and 95 percent and a line 
corresponding to the posterior medians.

* `psd_components()` produces a plot of the respective components of a State-space 
model, according to the Lomb-Scargle Power Spectral Density (PSD).  Possible 
components include Trend, Cycle (stochastic and/or deterministic), AR(1), and 
Irregular.




### Installation

The development version is currently available on Github, and can be installed in R with the following code:

`install.packages("devtools",dependencies=T)`

`devtools::install_github("mbtyers/SSjags")`

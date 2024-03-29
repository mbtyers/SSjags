% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/SSjags1.R
\name{runSS}
\alias{runSS}
\title{Construct and Run a State-Space Model}
\usage{
runSS(
  y,
  x = NULL,
  runmodel = TRUE,
  printmodel = TRUE,
  niter = 2000,
  ncores = NULL,
  parallel = TRUE,
  outlength = 1000,
  stochasticPeriods = NULL,
  deterministicPeriods = NULL,
  AR1 = FALSE,
  sig_eps_prior = NULL,
  sig_xi_prior = NULL,
  sig_omega_prior = NULL,
  sig_psi_prior = NULL,
  phi_prior = "dunif(0,1)",
  sigeps_breaks = NULL,
  sigxi_breaks = NULL,
  normalizedRate = FALSE
)
}
\arguments{
\item{y}{The input time series, expressed as a numeric vector.}

\item{x}{The corresponding time measurements, expressed as a numeric vector.
If the default \code{NULL} is used, the time measurements will be assumed to be
equally-spaced integer values.}

\item{runmodel}{Whether to run the model in 'JAGS', as the user may intend to
print the model to the console instead.  Defaults to \code{TRUE}.}

\item{printmodel}{Whether to print the model in 'JAGS' syntax to the console, providing a check to
validate that the model is constructed as intended.  Defaults to \code{TRUE}.}

\item{niter}{The number of MCMC iterations to run  Defaults to \code{2000}, which will
almost certainly not be enough to achieve convergence! However, this default value
will serve as a starting point for the user to test model performance and alter
the number of iterations as necessary.}

\item{ncores}{The number of CPU cores to use, assuming MCMC chains are run in
parallel.  If the default \code{NULL} is accepted, \code{parallel::detectCores} will attempt
to detect the number of available CPU cores, and the function will use the number
of available cores minus one.}

\item{parallel}{Whether to run chains in parallel, which should save considerable
processing time.  Defaults to \code{TRUE}.}

\item{outlength}{Desired number of MCMC iterations to save in output.  The function
will automatically thin the output to the desired length.  Defaults to \code{1000}.}

\item{stochasticPeriods}{An optional vector of stochastic cycle periods, in
units of time.  Defaults to \code{NULL}, indicating no stochastic cycle present.}

\item{deterministicPeriods}{An optional vector of deterministic cycle periods, in
units of time.  Defaults to \code{NULL}, indicating no deterministic cycle present.}

\item{AR1}{Whether to include an AR(1) autoregressive process.
Defaults to \code{FALSE}.}

\item{sig_eps_prior}{The prior used for the IRREGULAR standard deviation(s),
expressed in 'JAGS' syntax.  Note that providing a number here will set this value
to a constant, instead of treating it as a modeled quantity.  If the default \code{NULL}
is used, an exponential prior will be used (\code{dexp()} in 'JAGS' syntax), with
a rate parameter determined by the approximate scale of the data.}

\item{sig_xi_prior}{The prior used for the RATE DISTURBANCE standard deviation(s),
expressed in 'JAGS' syntax.  Note that providing a number here will set this value
to a constant, instead of treating it as a modeled quantity.  If the default \code{NULL}
is used, an exponential prior will be used (\code{dexp()} in 'JAGS' syntax), with
a rate parameter determined by the approximate scale of the data.}

\item{sig_omega_prior}{The prior used for the CYCLIC DISTURBANCE standard deviation(s),
expressed in 'JAGS' syntax.  Note that providing a number here will set this value
to a constant, instead of treating it as a modeled quantity.  Note that this argument
will be ignored if \code{stochasticPeriods} is set to \code{NULL}.   If the default \code{NULL}
is used, an exponential prior will be used (\code{dexp()} in 'JAGS' syntax), with
a rate parameter determined by the approximate scale of the data.}

\item{sig_psi_prior}{The prior used for the AR(1) PROCESS DISTURBANCE standard deviation,
expressed in 'JAGS' syntax.  Note that providing a number here will set this value
to a constant, instead of treating it as a modeled quantity.  Note that this argument
will be ignored unless \code{AR1} is set to \code{process}.  If the default \code{NULL}
is used, an exponential prior will be used (\code{dexp()} in 'JAGS' syntax), with
a rate parameter determined by the approximate scale of the data.}

\item{phi_prior}{The prior used for the AR(1) autoregressive parameter,
expressed in 'JAGS' syntax.  Note that providing a number here will set this value
to a constant, instead of treating it as a modeled quantity.  Note that this argument
will be ignored if \code{AR1==FALSE}.  Defaults to \code{"dunif(0,1)"}}

\item{sigeps_breaks}{An optional vector of structural breakpoints in the irregular
component, which may be interpreted as different irregular standard deviations
in different time periods.  Breakpoints should be expressed in the units of the
time measurements supplied to argument \code{x}.  A single breakpoint may be used if
there are two time periods, etc.  Defaults to \code{NULL}, indicating no breaks.}

\item{sigxi_breaks}{An optional vector of structural breakpoints in the RATE
DISTURBANCE component, which may be interpreted as different RATE
DISTURBANCE standard deviations
in different time periods.  Breakpoints should be expressed in the units of the
time measurements supplied to argument \code{x}.  A single breakpoint may be used if
there are two time periods, etc.  Defaults to \code{NULL}, indicating no breaks.}

\item{normalizedRate}{Whether to express treat the RATE DISTURBANCES as occurring
on the normalized time scale, as opposed to the data time scale.  If this is set
to \code{TRUE}, the rates and rate disturbance standard deviation may be interpreted
as rate per normalized time step; If this is set
to \code{FALSE}, the rates and rate disturbance standard deviation may be interpreted
as rate per unit time with respect to the input data units.  This functionality
is retained in order to facilitate comparison with output from other software.
Defaults to \code{FALSE}.}
}
\value{
An output object from \code{jagsUI::jags()}.  This will have the following
parameters, in which n_t denotes the length of the input time series and
n_s and n_d denote the number of stochastic and deterministic cycle periods, respectively:
\itemize{
\item \strong{trend: vector of length n_t} Trend \eqn{\mu_i} at each epoch \eqn{i}
\item \strong{rate: vector of length n_t} Rate \eqn{\nu_i} at each epoch \eqn{i}
\item \strong{cycle (possibly): vector of length n_t} Full sum of cycle components
\eqn{\sum_jc_{i,j}} at each epoch \eqn{i}
\item \strong{cycle_s (possibly): matrix of dimensions n_t x n_s} Cycle components
\eqn{c_{i,j}} at each epoch \eqn{i} and stochastic cycle period \eqn{j}
\item \strong{cycle_d (possibly): matrix of dimensions n_t x n_d} Cycle components
\eqn{c_{i,j}} at each epoch \eqn{i} and deterministic cycle period \eqn{j}
\item \strong{ar1 (possibly): vector of length n_t} AR(1) autoregressive component
\eqn{\zeta_i} at each epoch \eqn{i}
\item \strong{fit: vector of length n_t} Fitted value at each epoch \eqn{i}, equivalent
to the sum of the trend, cycle, and autoregressive components and excluding
the irregular component \eqn{\epsilon_i}
\item \strong{ypp: vector of length n_t} Posterior-predicted value at each epoch \eqn{i},
equivalent to sampling a predicted value for each MCMC sample given that sample's
values of fit and irregular standard deviation
\item \strong{sig_eps: vector} Irregular standard deviations \eqn{\sigma_\epsilon}
for each time period (if there are structural breaks, otherwise a single parameter)
\item \strong{sig_xi: vector} Rate disturbance standard deviations \eqn{\sigma_\xi}
for each time period (if there are structural breaks, otherwise a single parameter)
\item \strong{sig_omega (possibly): vector of length n_s} Cycle disturbance standard
deviations \eqn{\sigma_{\omega,j}} for each stochastic period \eqn{j}
\item \strong{phi (possibly): single value} Autoregressive parameter \eqn{\phi}
}
}
\description{
Automatically constructs and optionally runs a State-Space time series
model, given an input time series and optional model components.

This is a wrapper function, which constructs a model using the JAGS syntax according
to model arguments provided by the user.  The model is written to a temporary text file,
which is called by 'JAGS' using \code{jagsUI::jags()}.
\subsection{Data model & state model}{

Observation \eqn{y_i} at epoch \eqn{i} is treated as the sum of
trend \eqn{\mu_i}, cycle (seasonal harmonic) terms \eqn{\sum_jc_{i,j}}, AR(1)
process \eqn{\zeta_i}, and an irregular term \eqn{\epsilon_i}.  The irregular
term may be interpreted as consisting of
observation error plus un-modeled signal.

\deqn{
y_i = \mu_i+\sum_jc_{i,j}+\zeta_i+\epsilon_i
}

The trend is defined as the following, with a stochastic disturbance \eqn{\xi_i}
acting on the rate \eqn{\nu_i}.  With stochastic disturbances acting on the rate,
this may be termed an Integrated Random Walk model.

\deqn{
\mu_{i+1}=\mu_i+\nu_idt_i
}

\deqn{
\nu_{i+1}=\nu_i+\xi_i
}

Cycle (seasonal harmonic) terms \eqn{c_{i,j}} and \eqn{c^*_{i,j}} at epoch \eqn{i}
and frequency \eqn{\lambda_j} are defined
using the recursive formulation (Harvey, 1990) which allow frequency and
amplitude to vary stochastically through the inclusion of stochastic terms
\eqn{\omega_{i,j}} and \eqn{\omega^*_{i,j}}.  Note that the function arguments
express seasonal harmonic frequencies in terms of annual period \eqn{p_j}, in
which \eqn{\lambda_j=\frac{2\pi}{p_j}}.

\deqn{
c_{i+1,j}=c_{i,j}cos(\lambda_jdt_i)+c^*_{i,j}sin(\lambda_jdt_i)+\omega_{i,j}
}

\deqn{
c^*_{i+1,j}=-c_{i,j}sin(\lambda_jdt_i)+c^*_{i,j}cos(\lambda_jdt_i)+\omega^*_{i,j}
}

The AR(1) process is modeled as below, with autoregressive parameter \eqn{\phi}
for each dataset bounded on the interval \eqn{[0,1]}, and normalized time step
\eqn{dt_i} calculated as shown below.

\deqn{
\zeta_{i+1}=\zeta_i\phi^{dt_i} + \psi_i
}

\deqn{
dt_i=\frac{t_i-t_{i-1}}{\frac{1}{n}\sum_{i=1}^{n-1}(t_{i+1}-t_i)}
}
}

\subsection{Prior distributions for initial states}{

Trend and rate at the initial epoch are not set to constant values, but are
given diffuse univariate Gaussian prior distributions, of the form below, in
which \eqn{\hat{V}(.)} denotes the sample variance of a vector of data.

The prior distribution for the mass or volume trend at the initial epoch is
centered on the corresponding observation, with variance given by the sample
variance of the full mass or volume data vector, thus providing a relatively
diffuse distribution.

\deqn{
\mu_1 \sim N(y_1, \hat{V}(y_.))
}

The prior distribution for the rate at the initial epoch is
centered on zero, with variance given by the sample variance of the pairwise
differences between data values divided by the normalized time step, thus
providing a range of empirical rates.  The variance of these empirical rates
will be much greater than that of any smoothed rate, therefore this provides
a relatively diffuse prior distribution.

\deqn{
\nu_1 \sim N(0, \hat{V}(\frac{\Delta y_.}{dt_.}))
}

The AR(1) term at the initial time step \eqn{\zeta_i} for each element is set to zero.

Initial states of cycle terms \eqn{c_{i,j}} and \eqn{c^*_{i,j}} for frequency
\eqn{j} are given diffuse univariate Gaussian priors.

\deqn{
c_{1,j} \sim N(0,\frac{1}{0.001})
}

\deqn{
c^*_{1,j} \sim N(0,\frac{1}{0.001})
}
}

\subsection{Disturbance distributions and (default) hyperpriors}{

The irregular component \eqn{\epsilon_i}, rate disturbance component \eqn{\xi_i},
cyclic component disturbances \eqn{\omega_{i,j}} and \eqn{\omega^*_{i,j}}
associated with frequency \eqn{j}, and AR(1) disturbance \eqn{psi_i} are treated as Gaussian.

\deqn{
\epsilon_i \sim N(0,\sigma_\epsilon dt_i)
}

\deqn{
\xi_i \sim N(0,\sigma_\xi dt_i)
}

\deqn{
\omega_{i,j} \sim N(0,\sigma_{\omega,j} dt_i)
}

\deqn{
\omega^*_{i,j} \sim N(0,\sigma_{\omega,j} dt_i)
}

\deqn{
\psi_i \sim N(0,\sigma_\psi dt_i)
}

Standard deviation hyperparameters \eqn{\sigma_\epsilon}, \eqn{\sigma_\xi},\eqn{\sigma_{\omega,j}},
and \eqn{\sigma_\psi}
are given (by default) exponential priors with rate parameter 0.2, giving an expected value
of 5.  The exponential distribution gives a natural lower bound of zero, and
is intended to account for scale-dependence.  Note that priors are defined on
the standard deviation scale, as opposed to variance.

All (hyper)prior distributions are given defaults shown below, but can be specified
by the user according to JAGS syntax.

\deqn{
\sigma_\epsilon \sim Exp(0.2)
}

\deqn{
\sigma_\xi \sim Exp(0.2)
}

\deqn{
\sigma_{\omega,j} \sim Exp(0.2)
}

\deqn{
\sigma_\psi \sim Exp(0.2)
}

The AR(1) autoregressive parameters \eqn{\phi} is bounded on the interval \eqn{[0,1]}
and is given by default a Uniform(0,1) prior.

\deqn{
\phi \sim Unif(0,1)
}

Note that different prior distributions may be used instead for hyperparameters,
as well as fixed values.
}
}
\examples{
jagsout <- runSS(y = SS_data$y,                # vector of observations
               x = SS_data$x,                  # associated time measurements
               stochasticPeriods = c(1, 0.5),  # stochastic cycle periods
               niter = 10000,                  # number of MCMC iterations
               runmodel = FALSE)               # just print to console

\dontrun{
jagsout <- runSS(y = SS_data$y,                # vector of observations
               x = SS_data$x,                  # associated time measurements
               stochasticPeriods = c(1, 0.5),  # stochastic cycle periods
               niter = 100000)                 # number of MCMC iterations
}
}
\author{
Matt Tyers
}

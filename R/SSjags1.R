#' Construct and Run a State-Space Model
#' @description Automatically constructs and optionally runs a State-Space time series
#' model, given an input time series and optional model components.
#'
#' This is a wrapper function, which constructs a model using the JAGS syntax according
#' to model arguments provided by the user.  This is written to a temporary text file,
#' which is called by 'JAGS' using `jagsUI::jags()`.
#'
#' ## Data model & state model
#'
#' Observation \eqn{y_i} at epoch \eqn{i} is treated as the sum of
#' trend \eqn{\mu_i}, cycle (seasonal harmonic) terms \eqn{\sum_jc_{i,j}}, AR(1)
#' process \eqn{\zeta_i}, and an irregular term \eqn{\epsilon_i}.  The irregular
#' term may be interpreted as consisting of
#' observation error plus un-modeled signal.
#'
#' \deqn{
#' y_i = \mu_i+\sum_jc_{i,j}+\zeta_i+\epsilon_i
#' }
#'
#' The trend is defined as the following, with a stochastic disturbance \eqn{\xi_i}
#' acting on the rate \eqn{\nu_i}.  With stochastic disturbances acting on the rate,
#' this may be termed an Integrated Random Walk model.
#'
#' \deqn{
#' \mu_{i+1}=\mu_i+\nu_idt_i
#' }
#'
#' \deqn{
#' \nu_{i+1}=\nu_i+\xi_i
#' }
#'
#' Cycle (seasonal harmonic) terms \eqn{c_{i,j}} and \eqn{c^*_{i,j}} at epoch \eqn{i}
#' and frequency \eqn{\lambda_j} are defined
#' using the recursive formulation (Harvey, 1990) which allow frequency and
#' amplitude to vary stochastically through the inclusion of stochastic terms
#' \eqn{\omega_{i,j}} and \eqn{\omega^*_{i,j}}.  Note that the function arguments
#' express seasonal harmonic frequencies in terms of annual period \eqn{p_j}, in
#' which \eqn{\lambda_j=\frac{2\pi}{p_j}}.
#'
#' \deqn{
#' c_{i+1,j}=c_{i,j}cos(\lambda_jdt_i)+c^*_{i,j}sin(\lambda_jdt_i)+\omega_{i,j}
#' }
#'
#' \deqn{
#' c^*_{i+1,j}=-c_{i,j}sin(\lambda_jdt_i)+c^*_{i,j}cos(\lambda_jdt_i)+\omega^*_{i,j}
#' }
#'
#' The AR(1) process is modeled as below, with autoregressive parameter \eqn{\phi}
#' for each dataset bounded on the interval \eqn{[0,1]}, and normalized time step
#' \eqn{dt_i} calculated as shown below.
#'
#' \deqn{
#' \zeta_{i+1}=\zeta_i\phi^{dt_i} + \psi_i
#' }
#'
#' \deqn{
#' dt_i=\frac{t_i-t_{i-1}}{\frac{1}{n}\sum_{i=1}^{n-1}(t_{i+1}-t_i)}
#' }
#'
#' ## Prior distributions for initial states
#'
#' Trend and rate at the initial epoch are not set to constant values, but are
#' given diffuse univariate Gaussian prior distributions, of the form below, in
#' which \eqn{\hat{V}(.)} denotes the sample variance of a vector of data.
#'
#' The prior distribution for the mass or volume trend at the initial epoch is
#' centered on the corresponding observation, with variance given by the sample
#' variance of the full mass or volume data vector, thus providing a relatively
#' diffuse distribution.
#'
#' \deqn{
#' \mu_1 \sim N(y_1, \hat{V}(y_.))
#' }
#'
#' The prior distribution for the rate at the initial epoch is
#' centered on zero, with variance given by the sample variance of the pairwise
#' differences between data values divided by the normalized time step, thus
#' providing a range of empirical rates.  The variance of these empirical rates
#' will be much greater than that of any smoothed rate, therefore this provides
#' a relatively diffuse prior distribution.
#'
#' \deqn{
#' \nu_1 \sim N(0, \hat{V}(\frac{\Delta y_.}{dt_.}))
#' }
#'
#' The AR(1) term at the initial time step \eqn{\zeta_i} for each element is set to zero.
#'
#' Initial states of cycle terms \eqn{c_{i,j}} and \eqn{c^*_{i,j}} for frequency
#' \eqn{j} are given diffuse univariate Gaussian priors.
#'
#' \deqn{
#' c_{1,j} \sim N(0,\frac{1}{0.001})
#' }
#'
#' \deqn{
#' c^*_{1,j} \sim N(0,\frac{1}{0.001})
#' }
#'
#' ## Disturbance distributions and (default) hyperpriors
#'
#' The irregular component \eqn{\epsilon_i}, rate disturbance component \eqn{\xi_i},
#' cyclic component disturbances \eqn{\omega_{i,j}} and \eqn{\omega^*_{i,j}}
#' associated with frequency \eqn{j}, and AR(1) disturbance \eqn{psi_i} are treated as Gaussian.
#'
#' \deqn{
#' \epsilon_i \sim N(0,\sigma_\epsilon dt_i)
#' }
#'
#' \deqn{
#' \xi_i \sim N(0,\sigma_\xi dt_i)
#' }
#'
#' \deqn{
#' \omega_{i,j} \sim N(0,\sigma_{\omega,j} dt_i)
#' }
#'
#' \deqn{
#' \omega^*_{i,j} \sim N(0,\sigma_{\omega,j} dt_i)
#' }
#'
#' \deqn{
#' \psi_i \sim N(0,\sigma_\psi dt_i)
#' }
#'
#' Standard deviation hyperparameters \eqn{\sigma_\epsilon}, \eqn{\sigma_\xi},\eqn{\sigma_{\omega,j}},
#' and \eqn{\sigma_\psi}
#' are given (by default) exponential priors with rate parameter 0.2, giving an expected value
#' of 5.  The exponential distribution gives a natural lower bound of zero, and
#' is intended to account for scale-dependence.  Note that priors are defined on
#' the standard deviation scale, as opposed to variance.
#'
#' All (hyper)prior distributions are given defaults shown below, but can be specified
#' by the user according to JAGS syntax.
#'
#' \deqn{
#' \sigma_\epsilon \sim Exp(0.2)
#' }
#'
#' \deqn{
#' \sigma_\xi \sim Exp(0.2)
#' }
#'
#' \deqn{
#' \sigma_{\omega,j} \sim Exp(0.2)
#' }
#'
#' \deqn{
#' \sigma_\psi \sim Exp(0.2)
#' }
#'
#' The AR(1) autoregressive parameters \eqn{\phi} is bounded on the interval \eqn{[0,1]}
#' and is given by default a Uniform(0,1) prior.
#'
#' \deqn{
#' \phi \sim Unif(0,1)
#' }
#'
#' Note that different prior distributions may be used instead for hyperparameters,
#' as well as fixed values.
#'
#' @param y The input time series, expressed as a numeric vector.
#' @param x The corresponding time measurements, expressed as a numeric vector.
#' If the default `NULL` is used, the time measurements will be assumed to be
#' equally-spaced integer values.
#' @param runmodel Whether to run the model in 'JAGS'.  If `FALSE`, the function will
#' instead print the model in 'JAGS' syntax to the console, providing a check to
#' validate that the model is constructed as intended.  Defaults to `TRUE`.
#' @param niter The number of MCMC iterations to run  Defaults to `2000`, which will
#' almost certainly not be enough to achieve convergence! However, this default value
#' will serve as a starting point for the user to test model performance and alter
#' the number of iterations as necessary.
#' @param ncores The number of CPU cores to use, assuming MCMC chains are run in
#' parallel.  If the default `NULL` is accepted, `parallel::detectCores` will attempt
#' to detect the number of available CPU cores, and the function will use the number
#' of available cores minus one.
#' @param parallel Whether to run chains in parallel, which should save considerable
#' processing time.  Defaults to `TRUE`.
#' @param outlength Desired number of MCMC iterations to save in output.  The function
#' will automatically thin the output to the desired length.  Defaults to `1000`.
#' @param stochasticPeriods An optional vector of stochastic cycle periods, in
#' units of time.  Defaults to `NULL`, indicating no stochastic cycle present.
#' @param deterministicPeriods An optional vector of deterministic cycle periods, in
#' units of time.  Defaults to `NULL`, indicating no deterministic cycle present.
#' @param AR1 Whether to include an AR(1) autoregressive process.
#' Defaults to `FALSE`.
#' @param sig_eps_prior The prior used for the IRREGULAR standard deviation(s),
#' expressed in 'JAGS' syntax.  Note that providing a number here will set this value
#' to a constant, instead of treating it as a modeled quantity.  Defaults to `"dexp(0.2)"`
#' @param sig_xi_prior The prior used for the RATE DISTURBANCE standard deviation(s),
#' expressed in 'JAGS' syntax.  Note that providing a number here will set this value
#' to a constant, instead of treating it as a modeled quantity.  Defaults to `"dexp(0.2)"`
#' @param sig_omega_prior The prior used for the CYCLIC DISTURBANCE standard deviation(s),
#' expressed in 'JAGS' syntax.  Note that providing a number here will set this value
#' to a constant, instead of treating it as a modeled quantity.  Note that this argument
#' will be ignored if `stochasticPeriods` is set to `NULL`.  Defaults to `"dexp(0.2)"`
#' @param sig_psi_prior The prior used for the AR(1) PROCESS DISTURBANCE standard deviation,
#' expressed in 'JAGS' syntax.  Note that providing a number here will set this value
#' to a constant, instead of treating it as a modeled quantity.  Note that this argument
#' will be ignored unless `AR1` is set to `process`.  Defaults to `"dexp(0.2)"`
#' @param phi_prior The prior used for the AR(1) autoregressive parameter,
#' expressed in 'JAGS' syntax.  Note that providing a number here will set this value
#' to a constant, instead of treating it as a modeled quantity.  Note that this argument
#' will be ignored if `AR1==FALSE`.  Defaults to `"dunif(0,1)"`
#' @param sigeps_breaks An optional vector of structural breakpoints in the irregular
#' component, which may be interpreted as different irregular standard deviations
#' in different time periods.  Defaults to `NULL`, indicating no breaks.
#' @param normalizedRate Whether to express treat the RATE DISTURBANCES as occurring
#' on the normalized time scale, as opposed to the data time scale.  If this is set
#' to `TRUE`, the rates and rate disturbance standard deviation may be interpreted
#' as rate per normalized time step; If this is set
#' to `FALSE`, the rates and rate disturbance standard deviation may be interpreted
#' as rate per unit time with respect to the input data units.  This functionality
#' is retained in order to facilitate comparison with output from other software.
#' Defaults to `TRUE`.
#' @return An output object from `jagsUI::jags()`.  This will have the following
#' parameters, in which n_t denotes the length of the input time series and
#' n_s and n_d denote the number of stochastic and deterministic cycle periods, respectively:
#' * **trend: vector of length n_t** Trend \eqn{\mu_i} at each epoch \eqn{i}
#' * **rate: vector of length n_t** Rate \eqn{\nu_i} at each epoch \eqn{i}
#' * **cycle (possibly): vector of length n_t** Full sum of cycle components
#' \eqn{\sum_jc_{i,j}} at each epoch \eqn{i}
#' * **cycle_s (possibly): matrix of dimensions n_t x n_s** Cycle components
#' \eqn{c_{i,j}} at each epoch \eqn{i} and stochastic cycle period \eqn{j}
#' * **cycle_d (possibly): matrix of dimensions n_t x n_d** Cycle components
#' \eqn{c_{i,j}} at each epoch \eqn{i} and deterministic cycle period \eqn{j}
#' * **ar1 (possibly): vector of length n_t** AR(1) autoregressive component
#' \eqn{\zeta_i} at each epoch \eqn{i}
#' * **fit: vector of length n_t** Fitted value at each epoch \eqn{i}, equivalent
#' to the sum of the trend, cycle, and autoregressive components and excluding
#' the irregular component \eqn{\epsilon_i}
#' * **ypp: vector of length n_t** Posterior-predicted value at each epoch \eqn{i},
#' equivalent to sampling a predicted value for each MCMC sample given that sample's
#' values of fit and irregular standard deviation
#' * **sig_eps: single value** Irregular standard deviation \eqn{\sigma_\epsilon}
#' * **sig_xi: single value** Rate disturbance standard deviation \eqn{\sigma_\xi}
#' * **sig_omega (possibly): vector of length n_s** Cycle disturbance standard
#' deviations \eqn{\sigma_{\omega,j}} for each stochastic period \eqn{j}
#' * **phi (possibly): single value** Autoregressive parameter \eqn{\phi}
#' @note DO I WANT A NOTE HERE??
#' @author Matt Tyers
#' @importFrom parallel detectCores
#' @importFrom jagsUI jags
#' @importFrom grDevices adjustcolor
#' @importFrom graphics axis lines abline
#' @importFrom stats median var quantile
#' @examples
#' ## FILL THIS IN
#' @export
runSS <- function(y, x=NULL, runmodel=T,
                  niter=2000, ncores=NULL, parallel=TRUE,outlength=1000,
                  stochasticPeriods=NULL, deterministicPeriods=NULL, AR1=FALSE,
                  sig_eps_prior="dexp(0.2)",
                  sig_xi_prior="dexp(0.2)",
                  sig_omega_prior="dexp(0.2)",
                  sig_psi_prior="dexp(0.2)",
                  phi_prior="dunif(0,1)",
                  sigeps_breaks=NULL,
                  normalizedRate=TRUE) {

  if(is.null(ncores)) ncores <- parallel::detectCores()-1
  if(is.na(ncores)) stop("Unable to detect number of cores, please set ncores= manually.")
  tmp <- tempfile()

  isAR1noise <- AR1=="noise"
  isAR1process <- (AR1=="process")|(AR1==T)
  if(AR1==F) {
    isAR1noise <- isAR1process <- F
  }
  isDetcycle <- !is.null(deterministicPeriods)
  isStochcycle <- !is.null(stochasticPeriods)
  isSigeps_split <- !is.null(sigeps_breaks)

  cat('model {
  for(i in 1:n) {
    fit[i] <- trend[i]',file=tmp)
  if(isDetcycle | isStochcycle) cat(' + cycle[i]',file=tmp, append=T)
  if(isAR1noise | isAR1process) cat(' + ar1[i]',file=tmp, append=T)
  cat('
    y[i] ~ dnorm(fit[i], tau_eps',file=tmp, append=T)      ###### y[i] ~ dnorm(trend[i], tau_eps'
  if(isSigeps_split) cat('[sigeps_split[i]]', file=tmp, append=T)
  cat(')
    ypp[i] ~ dnorm(fit[i], tau_eps',file=tmp, append=T)    ##### ypp[i] ~ dnorm(trend[i], tau_eps'
  if(isSigeps_split) cat('[sigeps_split[i]]', file=tmp, append=T)
  cat(')
  }
  for(i in 2:n) {',file=tmp, append=T)
  if(normalizedRate) cat('
    trend[i] <- trend[i-1] + rate[i-1]*dt[i]',file=tmp, append=T)
  if(!normalizedRate) cat('
    trend[i] <- trend[i-1] + rate[i-1]*dt2[i]',file=tmp, append=T)
  cat('
    rate[i] ~ dnorm(rate[i-1], tau_xi)
',file=tmp, append=T)
  if(isAR1noise) cat('
    res[i] <- y[i]-fit[i]
    ar1[i] <- res[i-1]*(phi^dt[i])
',file=tmp, append=T)
  if(isAR1process) cat('
    ar1[i] ~ dnorm(ar1[i-1]*(phi^dt[i]), tau_psi)
',file=tmp, append=T)
  if(isStochcycle) cat('
    for(i_ps in 1:n_ps) {
      cc[i,i_ps] ~ dnorm(cc[i-1,i_ps]*cos(2*pi*dt2[i]/p_s[i_ps]) + c_star[i-1,i_ps]*sin(2*pi*dt2[i]/p_s[i_ps]), tau_omega[i_ps])
      c_star[i,i_ps] ~ dnorm(-cc[i-1,i_ps]*sin(2*pi*dt2[i]/p_s[i_ps]) + c_star[i-1,i_ps]*cos(2*pi*dt2[i]/p_s[i_ps]), tau_omega[i_ps])
      cycle_s[i,i_ps] <- cc[i,i_ps]
    }
', file=tmp, append=T)
  if(isDetcycle) cat('
    for(i_pd in 1:n_pd) {
      cycle_d[i,i_pd] <- betaS[i_pd]*sin(2*pi*tt[i]/p_d[i_pd]) + betaC[i_pd]*cos(2*pi*tt[i]/p_d[i_pd])
    }
', file=tmp, append=T)
  if(isStochcycle & isDetcycle) cat('
    cycle[i] <- sum(cycle_s[i,1:n_ps]) + sum(cycle_d[i,1:n_ps])
', file=tmp, append=T)
  if(isStochcycle & !isDetcycle) cat('
    cycle[i] <- sum(cycle_s[i,1:n_ps])
', file=tmp, append=T)
  if(!isStochcycle & isDetcycle) cat('
    cycle[i] <- sum(cycle_d[i,1:n_pd])
', file=tmp, append=T)
  cat('  }

  # initialize trend and rate
  trend[1] ~ dnorm(0, trendprecinit[1])  ###y[1]
  rate[1] ~ dnorm(0, rateprecinit[1])
', file=tmp, append=T)
  if(isAR1noise) cat('
  res[1] <- 0
  ar1[1] <- 0
', file=tmp, append=T)
  if(isAR1process) cat('
  ar1[1] <- 0
', file=tmp, append=T)
  if(isStochcycle) cat('
  for(i_ps in 1:n_ps) {
    cc[1,i_ps] ~ dnorm(0, 0.001)
    c_star[1,i_ps] ~ dnorm(0, 0.001)
    cycle_s[1,i_ps] <- cc[1,i_ps]
    cc_kf[i_ps] <- cc[n,i_ps]*cos(2*pi*dt2[n]/p_s[i_ps]) + c_star[n,i_ps]*sin(2*pi*dt2[n]/p_s[i_ps])
  }
', file=tmp, append=T)
  if(isDetcycle) cat('
  for(i_pd in 1:n_pd) {
    cycle_d[1,i_pd] <- betaS[i_pd]*sin(2*pi*tt[1]/p_d[i_pd]) + betaC[i_pd]*cos(2*pi*tt[1]/p_d[i_pd])
  }
', file=tmp, append=T)
  if(isStochcycle & isDetcycle) cat('
  cycle[1] <- sum(cycle_s[1,1:n_ps]) + sum(cycle_d[1,1:n_ps])
', file=tmp, append=T)
  if(isStochcycle & !isDetcycle) cat('
  cycle[1] <- sum(cycle_s[1,1:n_ps])
', file=tmp, append=T)
  if(!isStochcycle & isDetcycle) cat('
  cycle[1] <- sum(cycle_d[1,1:n_pd])
', file=tmp, append=T)

  cat('
  for(i_eps in 1:n_sigeps_split) {
    tau_eps[i_eps] <- pow(sig_eps[i_eps], -2)
    sig_eps[i_eps]', file=tmp, append=T)
  if(!is.numeric(sig_eps_prior)) cat(' ~',sig_eps_prior, file=tmp, append=T)
  if(is.numeric(sig_eps_prior)) {
    cat(' <- ',sig_eps_prior, file=tmp, append=T)
    if(length(sig_eps_prior)>1) cat('[i_eps]', file=tmp, append=T)
  }
  cat('
  }
  tau_xi <- pow(sig_xi, -2)
  sig_xi', file=tmp, append=T)
  if(!is.numeric(sig_xi_prior)) cat(' ~', sig_xi_prior, file=tmp, append=T)
  if(is.numeric(sig_xi_prior)) cat(' <-', sig_xi_prior, file=tmp, append=T)
  if(isStochcycle) {
    cat('
  for(i_ps in 1:n_ps) {
    tau_omega[i_ps] <- pow(sig_omega[i_ps], -2)
    sig_omega[i_ps]', file=tmp, append=T)
    if(!is.numeric(sig_omega_prior)) cat(' ~',sig_omega_prior, file=tmp, append=T)
    if(is.numeric(sig_omega_prior)) {
      cat(' <-',sig_omega_prior, file=tmp, append=T)
      if(length(sig_omega_prior)>1) cat('[i_ps]', file=tmp, append=T)
    }
    cat('
  }', file=tmp, append=T)
  }
  if(isDetcycle) {
    cat('
  for(i_pd in 1:n_pd) {
    betaS[i_pd] ~ dnorm(0, 0.0001)
    betaC[i_pd] ~ dnorm(0, 0.0001)
  }
', file=tmp, append=T)
  }
  if(isAR1process) {
    cat('
  tau_psi <- pow(sig_psi, -2)
  sig_psi', file=tmp, append=T)
    if(!is.numeric(sig_psi_prior)) cat(' ~',sig_psi_prior, file=tmp, append=T)
    if(is.numeric(sig_psi_prior)) {
      cat(' <-',sig_psi_prior, file=tmp, append=T)
    }
  }
  if(isAR1process | isAR1noise) {
    cat('
  phi', file=tmp, append=T)
    if(!is.numeric(phi_prior)) cat(' ~', phi_prior, file=tmp, append=T)
    if(is.numeric(phi_prior)) cat(' <-', phi_prior, file=tmp, append=T)
  }
  cat('
}
', file=tmp, append=T)

  if(!runmodel) {
    aschar <- readLines(tmp)
    for(i in 1:length(aschar)) cat(aschar[i],"\n")
  } else {
    ## bundle data
    if(is.null(x)) x <- 1:length(y)
    dt1 <- c(NA, diff(x))
    SS_data <- list(y=y,
                    n=length(y),
                    dt=dt1/mean(dt1,na.rm=T),
                    dt2=dt1,
                    tt=x-x[1], pi=pi, x=x)
    SS_data$y[is.nan(SS_data$y)] <- NA
    SS_data$trendprecinit <- 1/var(SS_data$y, na.rm=T)
    SS_data$rateprecinit <- 1/var(diff(SS_data$y)/diff(SS_data$tt), na.rm=T)  ### make this normalized time step!!

    SS_data$sigeps_split <- as.numeric(cut(SS_data$x, c(min(SS_data$x),sigeps_breaks,max(SS_data$x)), include.lowest=T))
    SS_data$n_sigeps_split <- max(SS_data$sigeps_split)

    if(is.numeric(phi_prior)) SS_data$phi_prior <- phi_prior
    SS_data$p_s <- stochasticPeriods
    SS_data$p_d <- deterministicPeriods
    SS_data$n_ps <- length(stochasticPeriods)
    SS_data$n_pd <- length(deterministicPeriods)

    ## run JAGS
    tstart <- Sys.time()
    cat("started at ")
    print(tstart)
    jagsout <- jagsUI::jags(model.file=tmp, data=SS_data,
                            parameters.to.save=c("trend","rate",
                                                 "cycle","cycle_s","cycle_d",
                                                 "ar1",
                                                 "fit","ypp",
                                                 "sig_eps","sig_xi","sig_omega",
                                                 "sig_psi",   ### added sig_psi
                                                 "phi"),
                            n.chains=ncores, parallel=T, n.iter=niter,
                            n.burnin=niter/2, n.thin=niter/outlength/2)

    now <- Sys.time()
    totaltime <- now - tstart
    print(totaltime)
    cat("finished at ")
    print(Sys.time())

    cat("\n","Max Rhat:",max(unlist(jagsout$Rhat), na.rm=T))
    cat("\n","Min n.eff:",min(unlist(jagsout$n.eff), na.rm=T))

    return(jagsout)
  }
}


envelope_separate <- function(y,x,col=NA,xlab="",ylab="",main="",...) {  # x is list of envelope things
  ranges <- sapply(y, function(x) diff(range(x,na.rm=T)))
  maxes <- sapply(y, function(x) max(apply(x,2,quantile, p=.95, na.rm=T),na.rm=T))
  mins <- sapply(y, function(x) min(apply(x,2,quantile, p=.05, na.rm=T),na.rm=T))
  ranges <- maxes-mins
  plot(NA, xlim=range(x), ylim=c(mins[1]-sum(ranges[2:length(y)]),maxes[1]), yaxt="n",ylab=ylab,xlab=xlab,main=main)
  abline(h=0,lty=3)
  if(all(is.na(col))) col<-rep(4,length(y))
  jagshelper::envelope(y[[1]], x=x, add=T,col=col[1],...=...)
  prettything <- pretty(c(maxes,mins),n=10)
  axis(side=2, at=prettything[prettything>mins[1] & prettything<maxes[1]],col=col[1],col.axis = col[1], las=2)
  for(i in 2:length(y)) {
    jagshelper::envelope(y[[i]]+maxes[1]-sum(ranges[1:(i-1)])-maxes[i], x=x, add=T,col=col[i],...=...)
    abline(h=maxes[1]-sum(ranges[1:(i-1)])-maxes[i], lty=3)
    axis(side=2, at=prettything[prettything>mins[i] & prettything<maxes[i]] + maxes[1]-sum(ranges[1:(i-1)])-maxes[i],
         labels=prettything[prettything>mins[i] & prettything<maxes[i]],
         col=col[i],col.axis = col[i], las=2)
  }
}
# envelope_separate(list(tryit$sims.list$trend, tryit$sims.list$cycle, tryit$sims.list$ar1), x=xss,col=1:3)



#' Plot Components of State-space Model
#' @description Produces a plot of the respective components of a State-space model.
#'
#' Possible components include Trend, Cycle (stochastic and/or deterministic), AR(1),
#' and Irregular.
#'
#' Model components will be plotted as posterior envelopes, using \link[jagshelper]{envelope},
#' with default credible interval widths of 50 percent and 95 percent and a line
#' corresponding to the posterior medians.
#' @param jagsout Output object returned from \link{runSS}.  Note that this will
#' be an object of class `jagsUI`.
#' @param y Input time series used by \link{runSS}.  Used by function to calculate
#' the Irregular component.  Accepting the default value `NULL` will omit plotting
#' the Irregular component.
#' @param x Time measurements associated with time series `y`.  If
#' default value `NULL` is accepted, integer-valued time steps will be plotted.
#' @param collapsecycle Whether to collapse all stochastic and/or deterministic
#' cycle components as one single Cycle component
#' @param ... additional arguments to \link[jagshelper]{envelope}
#' @return NULL
#' @author Matt Tyers
#' @importFrom jagshelper envelope
#' @examples
#' plot_components(jagsout=SS_out, y=SS_data$y, x=SS_data$x)
#' plot_components(jagsout=SS_out, y=SS_data$y, x=SS_data$x, collapsecycle=TRUE)
#' @export
plot_components <- function(jagsout, y=NULL, x=NULL, collapsecycle=FALSE, ...) {
  ylist <- list()
  ilist <- 1
  ylist[[ilist]] <- jagsout$sims.list$trend
  cols <- 4
  if(is.null(x)) x <- 1:ncol(jagsout$sims.list$trend)

  p_d_dim <- dim(jagsout$sims.list$cycle_d)
  p_s_dim <- dim(jagsout$sims.list$cycle_s)
  p_dim <- dim(jagsout$sims.list$cycle)
  ar_dim <- dim(jagsout$sims.list$ar1)

  if(collapsecycle) {
    if(!is.null(p_dim)) {
      ilist <- ilist+1
      ylist[[ilist]] <- jagsout$sims.list$cycle
      cols[ilist] <- 3
    }
  } else {
    if(!is.null(p_d_dim)) {
      for(i in 1:(p_d_dim[3])) {
        ilist <- ilist+1
        ylist[[ilist]] <- jagsout$sims.list$cycle_d[,,i]
        cols[ilist] <- 3
      }
    }
    if(!is.null(p_s_dim)) {
      for(i in 1:(p_s_dim[3])) {
        ilist <- ilist+1
        ylist[[ilist]] <- jagsout$sims.list$cycle_s[,,i]
        cols[ilist] <- 3
      }
    }
  }
  if(!is.null(ar_dim)) {
    ilist <- ilist+1
    ylist[[ilist]] <- jagsout$sims.list$ar1
    cols[ilist] <- 2
  }

  if(!is.null(y)) {
    totalfit <- ylist[[1]]
    for(i in 2:length(ylist)) totalfit <- totalfit+ylist[[i]]
    ylist[[ilist+1]] <- matrix(y, byrow=T, nrow=nrow(totalfit), ncol=ncol(totalfit))-totalfit
    cols[ilist+1] <- 1
  }

  envelope_separate(y=ylist, x=x, col=cols, ...=...)
  if(!is.null(y)) lines(x=x, y=y, col=adjustcolor(1,alpha.f=.5))
}
# components(jagsout=tryit, x=xss, y=y)
# components(jagsout=tryit, x=xss, y=y, ci=c(.1,.9), main="title")
# components(jagsout=tryit, x=xss)
# components(jagsout=tryit, y=y)
# components(jagsout=tryit, x=xss, y=y, collapsecycle = T)



#' Lomb-Scargle Power Spectral Density of State-space Model Components
#' @description Produces a plot of the respective components of a State-space model,
#' according to the Lomb-Scargle Power Spectral Density (PSD).
#'
#' Possible components include Trend, Cycle (stochastic and/or deterministic), AR(1),
#' and Irregular.
#' @param jagsout Output object returned from \link{runSS}.  Note that this will
#' be an object of class `jagsUI`.
#' @param y Input time series used by \link{runSS}.  Used by function to calculate
#' the Irregular component.  Accepting the default value `NULL` will omit plotting
#' the Irregular component.
#' @param x Time measurements associated with time series `y`.  If
#' default value `NULL` is accepted, integer-valued time steps will be plotted.
#' @param alpha Significance threshold for reference lines.  Defaults to `0.01`.
#' @param collapsecycle Whether to collapse all stochastic and/or deterministic
#' cycle components as one single Cycle component
#' @param ... additional plotting arguments
#' @return NULL
#' @author Matt Tyers
#' @importFrom lomb lsp
#' @examples
#' psd_components(jagsout=SS_out, y=SS_data$y, x=SS_data$x)
#' psd_components(jagsout=SS_out, y=SS_data$y, x=SS_data$x, collapsecycle=TRUE)
#' @export
psd_components <- function(jagsout, y=NULL, x=NULL, alpha=0.01, collapsecycle=FALSE,...) {
  ylist <- list()
  ilist <- 1
  ylist[[ilist]] <- jagsout$sims.list$trend
  cols <- 4
  axnames <- "Trend"

  p_d_dim <- dim(jagsout$sims.list$cycle_d)
  p_s_dim <- dim(jagsout$sims.list$cycle_s)
  p_dim <- dim(jagsout$sims.list$cycle)
  ar_dim <- dim(jagsout$sims.list$ar1)

  if(collapsecycle) {
    if(!is.null(p_dim)) {
      ilist <- ilist+1
      ylist[[ilist]] <- jagsout$sims.list$cycle
      cols[ilist] <- 3
      axnames[ilist] <- "Cycle"
    }
  } else {
    if(!is.null(p_d_dim)) {
      for(i in 1:(p_d_dim[3])) {
        ilist <- ilist+1
        ylist[[ilist]] <- jagsout$sims.list$cycle_d[,,i]
        cols[ilist] <- 3
        axnames[ilist] <- paste0("Cycle[",i,"]")
      }
    }
    if(!is.null(p_s_dim)) {
      for(i in 1:(p_s_dim[3])) {
        ilist <- ilist+1
        ylist[[ilist]] <- jagsout$sims.list$cycle_s[,,i]
        cols[ilist] <- 3
        axnames[ilist] <- paste0("Cycle[",i,"]")
      }
    }
  }
  if(!is.null(ar_dim)) {
    ilist <- ilist+1
    ylist[[ilist]] <- jagsout$sims.list$ar1
    cols[ilist] <- 2
    axnames[ilist] <- "AR(1)"
  }

  if(!is.null(y)) {
    totalfit <- ylist[[1]]
    for(i in 2:length(ylist)) totalfit <- totalfit+ylist[[i]]
    ylist[[ilist+1]] <- matrix(y, byrow=T, nrow=nrow(totalfit), ncol=ncol(totalfit))-totalfit
    cols[ilist+1] <- 1
    axnames[ilist+1] <- "Irregular"
  }

  medians <- lapply(ylist, function(x) apply(x,2,median,na.rm=T))
  lsps <- lapply(medians, lomb::lsp, times=x, plot=F, ofac=3, type="period", alpha=alpha)

  plot(NA, xlim=range(lsps[[1]]$scanned), ylim=c(-length(lsps),0),
       log="x", xlab="period",ylab="",yaxt="n",...=...)
  for(i in 1:length(lsps)) {
    infl <- 0.8/max(lsps[[i]]$power,lsps[[i]]$sig.level)
    lines(lsps[[i]]$scanned, infl*lsps[[i]]$power-i, col=cols[i])
    abline(h=infl*lsps[[i]]$sig.level-i, lty=3, col=cols[i])
    abline(h=-i, lty=1, col=adjustcolor(1,alpha.f=.2))
    axis(side=2,at=-i,labels=axnames[i],col.axis=cols[i])
  }
  # axis(side=2,at=-(1:length(lsps)),labels=axnames,col.axis=cols)

}
# psd_components(jagsout=tryit, x=xss, y=y)
# psd_components(jagsout=tryit, x=xss, y=y, alpha=0.1)
# psd_components(jagsout=tryit, x=xss, y=y, main="title")
# psd_components(jagsout=tryit, x=xss)
# psd_components(jagsout=tryit, y=y)
# psd_components(jagsout=tryit, x=xss, y=y, collapsecycle = T)


#' Example State-space output
#'
#' DOCUMENT ME!
#'
#'
"SS_out"


#' Example State-space input data
#'
#' DOCUMENT ME!
#'
#'
"SS_data"


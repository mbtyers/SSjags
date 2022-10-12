#' Construct and Run a State-Space Model
#' @description Automatically constructs and optionally runs a State-Space time series
#' model, given an input time series and optional model components.
#'
#' This is a wrapper function, which constructs a model using the JAGS syntax according
#' to model arguments provided by the user.  This is written to a temporary text file,
#' which is called by JAGS using `jagsUI::jags()`.
#'
#' THIS IS WHERE I SHOULD TALK A LOT ABOUT THE MODEL FORMULATION!!
#' @param y The input time series, expressed as a numeric vector.
#' @param x The corresponding time measurements, expressed as a numeric vector.
#' If the default `NULL` is used, the time measurements will be assumed to be
#' equally-spaced integer values.
#' @param runmodel Whether to run the model in JAGS.  If `FALSE`, the function will
#' instead print the model in JAGS syntax to the console, providing a check to
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
#' @param AR1 Whether to include autocorrelation (red noise) in the irregular term.
#' Defaults to `FALSE`.
#' @param sig_eps_prior The prior used for the IRREGULAR standard deviation(s),
#' expressed in JAGS syntax.  Note that providing a number here will set this value
#' to a constant, instead of treating it as a modeled quantity.  Defaults to `"dexp(0.2)"`
#' @param sig_xi_prior The prior used for the RATE DISTURBANCE standard deviation(s),
#' expressed in JAGS syntax.  Note that providing a number here will set this value
#' to a constant, instead of treating it as a modeled quantity.  Defaults to `"dexp(0.2)"`
#' @param sig_omega_prior The prior used for the CYCLIC DISTURBANCE standard deviation(s),
#' expressed in JAGS syntax.  Note that providing a number here will set this value
#' to a constant, instead of treating it as a modeled quantity.  Note that this argument
#' will be ignored if `stochasticPeriods` is set to `NULL`.  Defaults to `"dexp(0.2)"`
#' @param phi_prior The prior used for the AR(1) autoregressive parameter,
#' expressed in JAGS syntax.  Note that providing a number here will set this value
#' to a constant, instead of treating it as a modeled quantity.  Note that this argument
#' will be ignored if `AR1==FALSE`.  Defaults to `"dunif(0,1)"`
#' @param sigeps_breaks An optional vector of structural breakpoints in the irregular
#' component, which may be interpreted as different irregular standard deviations
#' in different time periods.  Defaults to `NULL`, indicating no breaks.
#' @note DO I WANT A NOTE HERE??
#' @author Matt Tyers
#' @importFrom parallel detectCores
#' @importFrom jagsUI jags
#' @importFrom grDevices adjustcolor
#' @importFrom graphics axis lines abline
#' @importFrom stats median var quantile
#' @examples
#' ## FILL THIS IN .... maybe try something like try(detectCores) at the appropriate spot
#' @export
runSS <- function(y, x=NULL, runmodel=T,
                      niter=2000, ncores=NULL, parallel=TRUE,outlength=1000,
                      stochasticPeriods=NULL, deterministicPeriods=NULL, AR1=FALSE,
                      sig_eps_prior="dexp(0.2)",
                      sig_xi_prior="dexp(0.2)",
                      sig_omega_prior="dexp(0.2)",
                      phi_prior="dunif(0,1)",
                      sigeps_breaks=NULL) {

  if(is.null(ncores)) ncores <- parallel::detectCores()-1
  if(is.na(ncores)) stop("Unable to detect number of cores, please set ncores= manually.")
  tmp <- tempfile()

  isAR1 <- AR1
  isDetcycle <- !is.null(deterministicPeriods)
  isStochcycle <- !is.null(stochasticPeriods)
  isSigeps_split <- !is.null(sigeps_breaks)

  cat('model {
  for(i in 1:n) {
    fit[i] <- trend[i]',file=tmp)
  if(isDetcycle | isStochcycle) cat(' + cycle[i]',file=tmp, append=T)
  if(isAR1) cat(' + ar1[i]',file=tmp, append=T)
  cat('
    y[i] ~ dnorm(fit[i], tau_eps',file=tmp, append=T)      ###### y[i] ~ dnorm(trend[i], tau_eps'
  if(isSigeps_split) cat('[sigeps_split[i]]', file=tmp, append=T)
  cat(')
    ypp[i] ~ dnorm(fit[i], tau_eps',file=tmp, append=T)    ##### ypp[i] ~ dnorm(trend[i], tau_eps'
  if(isSigeps_split) cat('[sigeps_split[i]]', file=tmp, append=T)
  cat(')
  }
  for(i in 2:n) {
    trend[i] <- trend[i-1] + rate[i-1]*dt2[i]
    rate[i] ~ dnorm(rate[i-1], tau_xi)
',file=tmp, append=T)
  if(isAR1) cat('
    res[i] <- y[i]-fit[i]
    ar1[i] <- res[i-1]*phi     ## phi might need to be ^dt[i]
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
  if(isAR1) cat('
  res[1] <- 0
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
  if(isAR1) {
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
    SS_data$rateprecinit <- 1/var(diff(SS_data$y)/diff(SS_data$tt), na.rm=T)

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
                            parameters.to.save=c("trend","rate","ypp","fit",
                                                 "sig_eps","sig_xi","sig_omega",
                                                 "cycle","cycle_s","cycle_d",
                                                 "ar1","phi"),
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


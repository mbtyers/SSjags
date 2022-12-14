% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/SSjags1.R
\name{plot_components}
\alias{plot_components}
\title{Plot Components of State-space Model}
\usage{
plot_components(jagsout, y = NULL, x = NULL, collapsecycle = FALSE, ...)
}
\arguments{
\item{jagsout}{Output object returned from \link{runSS}.  Note that this will
be an object of class \code{jagsUI}.}

\item{y}{Input time series used by \link{runSS}.  Used by function to calculate
the Irregular component.  Accepting the default value \code{NULL} will omit plotting
the Irregular component.}

\item{x}{Time measurements associated with time series \code{y}.  If
default value \code{NULL} is accepted, integer-valued time steps will be plotted.}

\item{collapsecycle}{Whether to collapse all stochastic and/or deterministic
cycle components as one single Cycle component}

\item{...}{additional arguments to \link[jagshelper]{envelope}}
}
\description{
Produces a plot of the respective components of a State-space model.

Possible components include Trend, Cycle (stochastic and/or deterministic), AR(1),
and Irregular.

Model components will be plotted as posterior envelopes, using \link[jagshelper]{envelope},
with default credible interval widths of 50 percent and 95 percent and a line
corresponding to the posterior medians.
}
\examples{
plot_components(jagsout=SS_out, y=SS_data$y, x=SS_data$x)
plot_components(jagsout=SS_out, y=SS_data$y, x=SS_data$x, collapsecycle=TRUE)
}
\author{
Matt Tyers
}

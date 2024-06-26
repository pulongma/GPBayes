% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils.R
\name{cor.to.par}
\alias{cor.to.par}
\title{Find the correlation parameter given effective range}
\usage{
cor.to.par(
  d,
  param,
  family = "CH",
  cor.target = 0.05,
  lower = NULL,
  upper = NULL,
  tol = .Machine$double.eps
)
}
\arguments{
\item{d}{a numerical value containing the effective range}

\item{param}{a list containing correlation parameters. The specification of 
\strong{param} should depend on the covariance model. If the parameter value is
 \code{NULL}, this function will find its value given the effective range via
 root-finding function \code{\link[stats]{uniroot}}.
\itemize{
\item{For the Confluent Hypergeometric class, \strong{range} is used to denote the range parameter \eqn{\beta}. 
\strong{tail} is used to denote the tail decay parameter \eqn{\alpha}. \strong{nu} is used to denote the 
smoothness parameter \eqn{\nu}.} 

\item{For the generalized Cauchy class, \strong{range} is used to denote the range parameter \eqn{\phi}. 
\strong{tail} is used to denote the tail decay parameter \eqn{\alpha}. \strong{nu} is used to denote the 
smoothness parameter \eqn{\nu}.}

\item{For the Matérn class, \strong{range} is used to denote the range parameter \eqn{\phi}. 
\strong{nu} is used to denote the smoothness parameter \eqn{\nu}. When \eqn{\nu=0.5}, the 
Matérn class corresponds to the exponential covariance.}  

\item{For the powered-exponential class, \strong{range} is used to denote the range parameter \eqn{\phi}.
\strong{nu} is used to denote the smoothness parameter. When \eqn{\nu=2}, the powered-exponential class
corresponds to the Gaussian covariance.}
}}

\item{family}{a string indicating the type of covariance structure.
The following correlation functions are implemented:
\describe{
\item{CH}{The Confluent Hypergeometric correlation function is given by 
\deqn{C(h) = \frac{\Gamma(\nu+\alpha)}{\Gamma(\nu)} 
\mathcal{U}\left(\alpha, 1-\nu, \left(\frac{h}{\beta}\right)^2\right),}
where \eqn{\alpha} is the tail decay parameter. \eqn{\beta} is the range parameter.
\eqn{\nu} is the smoothness parameter. \eqn{\mathcal{U}(\cdot)} is the confluent hypergeometric
function of the second kind. Note that this parameterization of the CH covariance is different from the one in Ma and Bhadra (2023). For details about this covariance, 
see Ma and Bhadra (2023; \doi{10.1080/01621459.2022.2027775}).  
}
\item{cauchy}{The generalized Cauchy covariance is given by
\deqn{C(h) = \left\{ 1 + \left( \frac{h}{\phi} \right)^{\nu}  
            \right\}^{-\alpha/\nu},}
where \eqn{\phi} is the range parameter. \eqn{\alpha} is the tail decay parameter.
\eqn{\nu} is the smoothness parameter.
}

\item{matern}{The Matérn correlation function is given by
\deqn{C(h)=\frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{h}{\phi} \right)^{\nu} 
\mathcal{K}_{\nu}\left( \frac{h}{\phi} \right),}
where \eqn{\phi} is the range parameter. \eqn{\nu} is the smoothness parameter. 
\eqn{\mathcal{K}_{\nu}(\cdot)} is the modified Bessel function of the second kind of order \eqn{\nu}.
}
\item{exp}{The exponential correlation function is given by 
\deqn{C(h)=\exp(-h/\phi),}
where \eqn{\phi} is the range parameter. This is the Matérn correlation with \eqn{\nu=0.5}.
}
\item{matern_3_2}{The Matérn correlation with \eqn{\nu=1.5}.}
\item{matern_5_2}{The Matérn correlation with \eqn{\nu=2.5}.}
}}

\item{cor.target}{a numerical value. The default value is 0.05, which 
means that correlation parameters are searched such that the correlation
is approximately 0.05.}

\item{lower}{a numerical value. This sets the lower bound to find the 
correlation parameter via the \code{R} function \code{\link[stats]{uniroot}}.}

\item{upper}{a numerical value. This sets the upper bound to find the 
correlation parameter via the \code{R} function \code{\link[stats]{uniroot}}.}

\item{tol}{a numerical value. This sets the precision of the solution with default value
specified as the machine precision \code{.Machine$double.eps} in \code{R}.}
}
\value{
a numerical value of correlation parameters
}
\description{
This function finds the correlation parameter given effective range
}
\examples{

range = cor.to.par(1,param=list(tail=0.5,nu=2.5), family="CH")
tail = cor.to.par(1,param=list(range=0.5,nu=2.5), family="CH")
range = cor.to.par(1,param=list(nu=2.5),family="matern")

}
\seealso{
\link{GPBayes-package}, \code{\link{GaSP}}, \code{\link{kernel}}, \code{\link{ikernel}}
}
\author{
Pulong Ma \email{mpulong@gmail.com}
}

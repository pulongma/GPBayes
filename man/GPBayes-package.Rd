\name{GPBayes-package}
\alias{GPBayes-package}
\alias{GPBayes-package}
\docType{package}
\title{
  Tools for Gaussian Stochastic Process Modeling in Uncertainty Quantification
}
\description{
Gaussian processes (GPs) have been widely used to model spatial data, spatio-temporal data, and computer experiments in diverse areas of statistics including spatial statistics, spatio-temporal statistics, uncertainty quantification, and machine learning. This package creates basic tools for fitting and prediction based on GPs with spatial data, spatio-temporal data, and computer experiments. Key characteristics for this GP tool include: (1) the comprehensive implementation of various covariance functions including the Matérn family and the Confluent Hypergeometric family with isotropic form, tensor form, and automatic relevance determination form, where the isotropic form is widely used in spatial statistics, the tensor form is widely used in design and analysis of computer experiments and uncertainty quantification, and the automatic relevance determination form is widely used in machine learning; (2) implementations via Markov chain Monte Carlo (MCMC) algorithms and optimization algorithms for GP models with all the implemented covariance functions. The methods for fitting and prediction are mainly implemented in a Bayesian framework; (3) model evaluation via Fisher information and predictive metrics such as predictive scores; (4) built-in functionality for simulating GPs with all the implemented covariance functions; (5) unified implementation to allow easy specification of various GPs.  
}
\details{
\itemize{
  \item{Data types: For many scientific applications, spatial data, spatio-temporal data, and computer experiments arise naturally. This package provides a comprehensive set of basic tools to fit GaSP models for univariate and multivariate spatial data, spatio-temporal data, computer experiments. Various covariance functions have been implemented including the Confluent Hypergeometric covariance functions, the Matérn covariance functions, the Gaussian covariance function, the generalized Cauchy covariance function. These covariance families can be in isotropic form, in tensor form, or in automatic relevance determination form. The routines \code{\link{kernel}} and \code{\link{ikernel}} contain the details of implementation.  }

  \item{Model simulation: This package can simulate realizations from GaSP for different types of data including spatial data, spatio-temporal data, and computer experiments. This feature is quite useful in part because benchmarks are used to evaluate the performance of GaSP models. This functionality is implemented in the routine \code{\link{gp.sim}}.}

  \item{Model fitting: Both maximum likelihood methods (or its variants) and Bayes estimation methods such as maximum a posterior (MAP) and Markov chain Monte Carlo (MCMC) methods are implemented. In this package, the nugget parameter is included in the model by default for the sake of better prediction performance and stable computation in practice. In addition, the smoothness parameter in covariance functions such as the Matérn class and the Confluent Hypergeometric class can be estimated. The routine \code{\link{gp.optim}} provides optimization based estimation approaches and the routine \code{\link{gp.mcmc}} provides MCMC algorithms based estimation approaches.}

  \item{Model prediction: Prediction is made based on the parameter estimation procedure. If maximum likelihood estimation (MLE) methods are used for parameter estimation, the plug-in approach is used for prediction in the sense that MLEs of parameters are plugged into posterior predictive distributions. If partial Bayes methods (e.g., maximum a posterior) are used, the plug-in approach is used for prediction as well. If fully Bayes methods via MCMC algorithms are used, posterior samples are drawn from posterior predictive distributions. The routine \code{\link{gp.mcmc}} allows prediction to be made within the MCMC algorithms, while the routine \code{\link{gp.predict}} generates prediction with estimated parameters.}

  \item{Model assessment: Tools for assessing model adequacy are included in a Bayesian context. Deviance information criteria (DIC), log pointwise predictive density, and log joint predictive density can be computed via the routine \code{\link{gp.model.adequacy}}. }

}
}

\author{
Pulong Ma \email{mpulong@gmail.com}
}
\references{
\itemize{

  \item{Cressie, N. (1993). ``Statistics for Spatial Data.'' John Wiley & Sons, New York, revised edition.}

  \item{Ma and Bhadra (2019). ``Beyond Matérn: On a Class of Interpretable Confluent Hypergeometric Covariance Functions.'' arXiv: 1911.05865. \url{https://arxiv.org/abs/1911.05865}.}

  \item{Sacks, Jerome, William J Welch, Toby J Mitchell, and Henry P Wynn. (1989). ``Design and Analysis of Computer Experiments.'' \emph{Statistical Science} \strong{4}(4). Institute of Mathematical Statistics: 409–435.}

  \item{Santner, Thomas J., Brian J. Williams, and William I. Notz. (2018). ``The Design and Analysis of Computer Experiments''; 2nd Ed. New York: Springer.}
 
  \item{Stein, Michael L. (1999). ``Interpolation of Spatial Data.'' Springer Science & Business Media, New York.}

}
}
\keyword{ package }
\seealso{
  \code{\link{GaSP}}
}
\examples{
#####################################################################
          
\donttest{
#####################################################################
############## Examples for fitting univariate GP models ############

## Set up the Sine example from the tgp package 
 code = function(x){
  y = (sin(pi*x/5) + 0.2*cos(4*pi*x/5))*(x<=9.6) + (x/10-1)*(x>9.6) 
 }
 n=100
 input = seq(0, 20, length=n)
 XX = seq(0, 20, length=99)
 Ztrue = code(input)
 set.seed(1234)
 output = Ztrue + rnorm(length(Ztrue), sd=0.1)
 df.data = data.frame(x=c(input), y=output, y.true=Ztrue)

 ## fitting a GaSP model with the Cauchy prior
 fit = GaSP(formula=~1, output, input, 
           param=list(range=3, nugget=0.1, nu=2.5), 
           smooth.est=FALSE, input.new=XX,
           cov.model=list(family="matern", form="isotropic"),
           proposal=list(range=.35, nugget=.8, nu=0.8),
           dtype="Euclidean", model.fit="Cauchy_prior", nsample=3000, 
           burnin=500, verbose=TRUE)

 ## fitting a GaSP model with the beta prior
 fit = GaSP(formula=~1, output, input, 
           param=list(range=3, nugget=0.1, nu=2.5), 
           smooth.est=FALSE, input.new=XX,
           cov.model=list(family="matern", form="isotropic"),
           prior=list(range=list(a=1,b=1,lb=0,ub=20),
                    nugget=list(a=1,b=1,lb=0,ub=var(output)),
           proposal=list(range=.35, nugget=.8, nu=0.8),
           dtype="Euclidean", model.fit="Beta_prior", nsample=3000, 
           burnin=500, verbose=TRUE))

## fitting a GaSP model with the marginal maximum likelihood approach
 fit = GaSP(formula=~1, output, input, 
           param=list(range=3, nugget=0.1, nu=2.5), 
           smooth.est=FALSE, input.new=XX,
           cov.model=list(family="matern", form="isotropic"),
           dtype="Euclidean", model.fit="MMLE", verbose=TRUE)

## fitting a GaSP model with the profile maximum likelihood approach
 fit = GaSP(formula=~1, output, input, 
           param=list(range=3, nugget=0.1, nu=2.5), 
           smooth.est=FALSE, input.new=XX,
           cov.model=list(family="matern", form="isotropic"),
           dtype="Euclidean", model.fit="MPLE", verbose=TRUE)  
  }
}

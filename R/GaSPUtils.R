#' @title Construct the \code{S4} object \linkS4class{gp}
#' @description This function constructs the \code{S4} object \linkS4class{gp}  that is used for Gaussian process 
#' model fitting and prediction.
#' 
#' @param formula an object of \code{formula} class that specifies regressors; see \code{\link[stats]{formula}} for details.
#' @param output a numerical vector including observations or outputs in a GaSP
#' @param input a matrix including inputs in a GaSP
#' 
#' @param param a list including values for regression parameters, covariance parameters, 
#' and nugget variance parameter.
#' The specification of \strong{param} should depend on the covariance model. 
#' \itemize{
#' \item{The regression parameters are denoted by \strong{coeff}. Default value is \eqn{\mathbf{0}}.}
#' \item{The marginal variance or partial sill is denoted by \strong{sig2}. Default value is 1.}
#' \item{The nugget variance parameter is denoted by \strong{nugget} for all covariance models. 
#' Default value is 0.}
#' \item{For the Confluent Hypergeometric class, \strong{range} is used to denote the range parameter \eqn{\beta}. 
#' \strong{tail} is used to denote the tail decay parameter \eqn{\alpha}. \strong{nu} is used to denote the 
#' smoothness parameter \eqn{\nu}.}
#' \item{For the generalized Cauchy class, \strong{range} is used to denote the range parameter \eqn{\phi}. 
#' \strong{tail} is used to denote the tail decay parameter \eqn{\alpha}. \strong{nu} is used to denote the 
#' smoothness parameter \eqn{\nu}.}
#' \item{For the Matérn class, \strong{range} is used to denote the range parameter \eqn{\phi}. 
#' \strong{nu} is used to denote the smoothness parameter \eqn{\nu}. When \eqn{\nu=0.5}, the 
#' Matérn class corresponds to the exponential covariance.}  
#' \item{For the powered-exponential class, \strong{range} is used to denote the range parameter \eqn{\phi}.
#' \strong{nu} is used to denote the smoothness parameter. When \eqn{\nu=2}, the powered-exponential class
#' corresponds to the Gaussian covariance.}
#' }
#' @param cov.model a list of two strings: \strong{family}, \strong{form}, where \strong{family} indicates the family of covariance functions 
#' including the Confluent Hypergeometric class, the Matérn class, the Cauchy class, the powered-exponential class. \strong{form} indicates the 
#' specific form of covariance structures including the isotropic form, tensor form, automatic relevance determination form. 
#' \describe{
#' \item{\strong{family}}{
#' \describe{
#' \item{CH}{The Confluent Hypergeometric correlation function is given by 
#' \deqn{C(h) = \frac{\Gamma(\nu+\alpha)}{\Gamma(\nu)} 
#' \mathcal{U}\left(\alpha, 1-\nu, \left(\frac{h}{\beta}\right)^2\right),}
#' where \eqn{\alpha} is the tail decay parameter. \eqn{\beta} is the range parameter.
#' \eqn{\nu} is the smoothness parameter. \eqn{\mathcal{U}(\cdot)} is the confluent hypergeometric
#' function of the second kind. For details about this covariance, 
#' see Ma and Bhadra (2019) at \url{https://arxiv.org/abs/1911.05865}.  
#' }
#' \item{cauchy}{The generalized Cauchy covariance is given by
#' \deqn{C(h) = \left\{ 1 + \left( \frac{h}{\phi} \right)^{\nu}  
#'             \right\}^{-\alpha/\nu},}
#' where \eqn{\phi} is the range parameter. \eqn{\alpha} is the tail decay parameter.
#' \eqn{\nu} is the smoothness parameter with default value at 2.
#'}
#'
#' \item{matern}{The Matérn correlation function is given by
#' \deqn{C(h)=\frac{2^{1-\nu}}{\Gamma(\nu)} \left( \frac{h}{\phi} \right)^{\nu} 
#' \mathcal{K}_{\nu}\left( \frac{h}{\phi} \right),}
#' where \eqn{\phi} is the range parameter. \eqn{\nu} is the smoothness parameter. 
#' \eqn{\mathcal{K}_{\nu}(\cdot)} is the modified Bessel function of the second kind of order \eqn{\nu}.
#' }
#' \item{exp}{The exponential correlation function is given by 
#' \deqn{C(h)=\exp(-h/\phi),}
#' where \eqn{\phi} is the range parameter. This is the Matérn correlation with \eqn{\nu=0.5}.
#' }
#' \item{matern_3_2}{The Matérn correlation with \eqn{\nu=1.5}.}
#' \item{matern_5_2}{The Matérn correlation with \eqn{\nu=2.5}.}
#'
#'
#' \item{powexp}{The powered-exponential correlation function is given by
#'                \deqn{C(h)=\exp\left\{-\left(\frac{h}{\phi}\right)^{\nu}\right\},}
#' where \eqn{\phi} is the range parameter. \eqn{\nu} is the smoothness parameter.
#' }
#' \item{gauss}{The Gaussian correlation function is given by 
#' \deqn{C(h)=\exp\left(-\frac{h^2}{\phi^2}\right),}
#' where \eqn{\phi} is the range parameter.
#'  }
#' }
#' }
#' 
#' \item{\strong{form}}{
#' \describe{
#'  \item{isotropic}{This indicates the isotropic form of covariance functions. That is,
#'  \deqn{C(\mathbf{h}) = C^0(\|\mathbf{h}\|; \boldsymbol \theta),} where \eqn{\| \mathbf{h}\|} denotes the 
#' Euclidean distance or the great circle distance for data on sphere. \eqn{C^0(\cdot)} denotes 
#' any isotropic covariance family specified in \strong{family}.}
#'  \item{tensor}{This indicates the tensor product of correlation functions. That is, 
#' \deqn{ C(\mathbf{h}) = \prod_{i=1}^d C^0(|h_i|; \boldsymbol \theta_i),}
#' where \eqn{d} is the dimension of input space. \eqn{h_i} is the distance along the \eqn{i}th input dimension. This type of covariance structure has been often used in Gaussian process emulation for computer experiments.
#'}
#'  \item{ARD}{This indicates the automatic relevance determination form. That is, 
#' \deqn{C(\mathbf{h}) = C^0\left(\sqrt{\sum_{i=1}^d\frac{h_i^2}{\phi^2_i}}; \boldsymbol \theta \right),}
#' where \eqn{\phi_i} denotes the range parameter along the \eqn{i}th input dimension.}
#'  }
#' }
#'
#'}
#' 
#' @param smooth.est a logical value indicating whether smoothness parameter will be estimated.
#'
#' @param dtype a string indicating the type of distance:
#' \describe{
#' \item{Euclidean}{Euclidean distance is used. This is the default choice.}
#' \item{GCD}{Great circle distance is used for data on sphere.}
#'}
#' @seealso \link{GPBayes-package}, \code{\link{GaSP}}
#' @author Pulong Ma \email{mpulong@@gmail.com}
#'  
#' @export
#' @return an \code{S4} object of \linkS4class{gp} class 
#' 
#' @examples  
#' code = function(x){
#' y = (sin(pi*x/5) + 0.2*cos(4*pi*x/5))*(x<=9.6) + (x/10-1)*(x>9.6) 
#' return(y)
#' }
#' n=100
#' input = seq(0, 20, length=n)
#' XX = seq(0, 20, length=99)
#' Ztrue = code(input)
#' set.seed(1234)
#' output = Ztrue + rnorm(length(Ztrue), sd=0.1)
#' obj = gp(formula=~1, output, input, 
#'         param=list(range=4, nugget=0.1,nu=2.5),
#'         smooth.est=FALSE,
#'         cov.model=list(family="matern", form="isotropic"))
#' 
gp <- function(formula=~1, output, input, param, smooth.est=FALSE,
              cov.model=list(family="CH", form="isotropic"), dtype="Euclidean"){
  
## check the arguments
  checkout = .check.arg.gp(formula=formula, output=output, input=input, param=param, cov.model=cov.model, dtype=dtype)
  input = checkout$input
  param = checkout$param 
  output = checkout$output
  cov.model = checkout$cov.model 

  Dim = dim(input)[2]
  n = nrow(input)
  q = ncol(output)

  range = param$range
  tail = param$tail 
  nu = param$nu 
  nugget = param$nugget
  sig2 = param$sig2  

  d = distance(input, input, type=cov.model$form, dtype);
  R = kernel(d, range=range, tail=tail, nu=nu, covmodel=cov.model)

  R = R + nugget*diag(n)

  colnames(input) = paste0("x", 1:Dim)
  df = data.frame(input)
  H = model.matrix(formula, df)

  coeff = param$coeff 

  
  #loglik = mvtnorm::dmvnorm(x=output, mean=H%*%coeff, sigma=covmat, log=TRUE)


  

  if(q==1){
    Q = chol(sig2*R)
    res = output - H%*%coeff 
    Qres = forwardsolve(Q, res)
    S2 = crossprod(Qres)
    loglik = -0.5*n*log(2*pi) -0.5*sum(log(diag(Q))) - 0.5*log(S2)
  }else{
    Q = chol(R)
    res = output - H%*%coeff 
    Qres = forwardsolve(Q, res)
    S2 = crossprod(Qres)
    tr_quad = sum(diag(solve(sig2, S2)))
    loglik = -0.5*n*q*log(2*pi)-0.5*q*sum(log(diag(Q))) - 0.5*sum(log(diag(chol(sig2)))) - 0.5*tr_quad
  }

  
  if(cov.model$form=="isotropic"){
    maxdist = max(d)
  }else{
    maxdist = sapply(d, max)
  }
  
  ## construct the gp object
  new("gp",
      formula = formula,
      output = output,
      input = input,
      cov.model = cov.model,
      param = param,
      smooth.est = smooth.est,
      dtype = dtype,
      loglik = drop(loglik),
      mcmc = list(),
      info = list(max.distance=maxdist)
      )
  
}



#####################################################################
#####################################################################





#####################################################################
#####################################################################
.check.arg.gp <- function(formula, output, input, param, cov.model, dtype){

  message("\n")

  if(!is(formula, "formula")){
    stop("Please specify a formula to extract the design/basis matrix.\n")
  }
  
  
  if(!is(input, "matrix")){
    #message("coerce input to a matrix format.\n")
    input = as.matrix(input)
  }

  if(!is(output, "matrix")){
    #message("coerce output to a matrix format.\n")
    output = as.matrix(output)
  }


    if(!exists("family", where=cov.model)){
      message("set default values for cov.model.\n")
      cov.model$family = "CH"
    }else{
      if(cov.model$family=="CH" || cov.model$family=="cauchy" || cov.model$family=="matern" ||
        cov.model$family=="exp" || cov.model$family=="matern_3_2" || cov.model$family=="matern_5_2" ||
        cov.model$family=="gauss" || cov.model$family=="powexp"){
          if(cov.model$family=="exp" || cov.model$family=="matern_3_2" || cov.model$family=="matern_5_2"){
            cov.model$family = "matern"
          }
      }else{
        message("The specified covariance family is not implemented.\n")
        stop("Please check the instruction of gp for details.")
      }
    }

    if(cov.model$family=="cauchy" || cov.model$family=="powexp" || cov.model$family=="gauss"){
      if(param$nu>2.0){
        message("The specified smoothness parameter is greater than 2.0.\n")
        message("It is reset to be 2.0 to ensure positive definitness.\n")
        param$nu = 2.0;
      }
    }

    if(!exists("form", where=cov.model)){
      message("set default values for cov.model.\n")
      cov.model$form = "isotropic"
    }
  

  
  Dim = dim(input)[2]
  colnames(input) = paste0("x", 1:Dim)
  df = data.frame(input)
  H = model.matrix(formula, df)
  p = dim(H)[2]

  d = distance(input, input, type=cov.model$form, dtype);

  if(!is(param, "list")){
    stop("param should be a list containing initial values for  
         correlation parameters and nugget variance parameter.\n")
  }else{
    if(!exists("coeff", where=param)){
      param$coeff = rep(0, p)
    }

    if(!exists("sig2", where=param)){
      param$sig2 = 1 
    }
    
    if(!exists("nugget", where=param)){
      param$nugget = 0 
    }

    if(!exists("range", where=param)){
      if(cov.model$form=="isotropic"){
        param$range = max(d)/2
      }else{
        param$range = sapply(d, max)/2
      }
      
    }

    if(!exists("tail", where=param)){
      if(cov.model$form=="isotropic"){
        param$tail = 0.5
      }else{
        param$tail = rep(0.5, Dim)
      }
    }

    if(!exists("nu", where=param)){
      if(cov.model$form=="isotropic"){
        param$nu = 0.5
      }else{
        param$nu = rep(0.5, Dim)
      }
    }
    
  }



  
  return(list(input=input, output=output, param=param, cov.model=cov.model))
}

###########################################################################################
##########################################################################################

###########################################################################################
##########################################################################################
#' @title A wraper to fit a Gaussian stochastic process model with MCMC algorithms
#' @description This function is a wraper to estimate parameters via MCMC algorithms in the GaSP model with different
#' choices of priors. 
#' 
#' @param obj an \code{S4} object \linkS4class{gp}
#' @param input.new a matrix of prediction locations. Default value is \code{NULL}, 
#' indicating that prediction is not carried out along with parameter estimation in the MCMC algorithm.
#' @param method a string indicating the Bayes estimation approaches with different choices of priors on correlation parameters:
#' \describe{
#' \item{Cauchy_prior}{This indicates that a fully Bayesian approach with objective priors is used
#' for parameter estimation, where location-scale parameters are assigned with constant priors and 
#' correlation parameters are assigned with half-Cauchy priors (default). 
#' If the smoothness parameter is estimated for \code{isotropic} covariance functions, the smoothness parameter is assigned with a uniform prior on (0, 4), 
#' indicating that the corresponding GP is at most four times mean-square differentiable. This is a 
#' reasonable prior belief for modeling spatial processes; If the smoothness parameter is estimated for \code{tensor} or \code{ARD} covariance functions,
#' the smoothness parameter is assigned with a uniform prior on (0, 6).}
#' 
#' \item{Ref_prior}{This indicates that a fully Bayesian approach with objective priors is used
#' for parameter estimation, where location-scale parameters are assigned with constant priors and 
#' correlation parameters are assigned with reference priors. 
#' If the smoothness parameter is estimated for \code{isotropic} covariance functions, the smoothness parameter is assigned with a uniform prior on (0, 4), 
#' indicating that the corresponding GP is at most four times mean-square differentiable. This is a 
#' reasonable prior belief for modeling spatial processes; If the smoothness parameter is estimated for \code{tensor} or \code{ARD} covariance functions,
#' the smoothness parameter is assigned with a uniform prior on (0, 6).}
#'
#' \item{Beta_prior}{This indicates that a fully Bayesian approach with subjective priors is used
#' for parameter estimation, where location-scale parameters are assigned with constant priors and 
#' correlation parameters are assigned with \link{beta} priors parameterized as \eqn{Beta(a, b, lb, ub)}.
#' In the beta distribution, \strong{lb} and \strong{ub} are the support for correlation parameters, and
#' they should be determined based on domain knowledge. \strong{a} and \strong{b} are two shape parameters with default values at 1,
#' corresponding to the uniform prior over the support \eqn{(lb, ub)}. 
#' }
#' }
#' 
#' @param prior a list containing tuning parameters in prior distributions. This is used only if a Bayes estimation method with subjective priors is used.
#' @param proposal a list containing tuning parameters in proposal distributions. This is used only if a Bayes estimation method is used.
#' @param nsample an integer indicating the number of MCMC samples. 
#'
#' @param verbose a logical value. If it is \code{TRUE}, the MCMC progress bar is shown. 
#' @return a \code{\link{gp}} object with prior, proposal, MCMC samples included. 
#' 
#' @author Pulong Ma \email{mpulong@@gmail.com}
#' 
#' @seealso \link{GPBayes-package}, \code{\link{GaSP}}, \linkS4class{gp}, \code{\link{gp.optim}}
#' @export
#' 
#' @examples  
#'  
#' code = function(x){
#' y = (sin(pi*x/5) + 0.2*cos(4*pi*x/5))*(x<=9.6) + (x/10-1)*(x>9.6) 
#' return(y)
#' }
#' n=100
#' input = seq(0, 20, length=n)
#' XX = seq(0, 20, length=99)
#' Ztrue = code(input)
#' set.seed(1234)
#' output = Ztrue + rnorm(length(Ztrue), sd=0.1)
#' obj = gp(formula=~1, output, input, 
#'         param=list(range=4, nugget=0.1,nu=2.5),
#'         smooth.est=FALSE,
#'         cov.model=list(family="matern", form="isotropic"))
#'         
#' fit.mcmc = gp.mcmc(obj, method="Cauchy_prior",
#'                    proposal=list(range=0.3, nugget=0.8),
#'                    nsample=100, verbose=TRUE)
#'                    
gp.mcmc <- function(obj, input.new=NULL, method="Cauchy_prior", prior=list(), proposal=list(), 
            nsample=10000, verbose=TRUE){


  formula = obj@formula
  output = obj@output 
  input = obj@input 
  param = obj@param
  cov.model = obj@cov.model 
  dtype = obj@dtype
  info = obj@info
  smooth.est=obj@smooth.est 

  Dim = dim(input)[2]
  colnames(input) = paste0("x", 1:Dim)
  df = data.frame(input)
  H = model.matrix(formula, df)

  if(!is.null(input.new)){
    if(!is.matrix(input.new)){
      #message("Converting input.new to a matrix...\n")
      input.new = as.matrix(input.new)
    }

    colnames(input.new) = paste0("x", 1:Dim)
    df = data.frame(input.new)
    Hnew = model.matrix(formula, df)    
  }

  proposal_new = list()
  if(length(obj@proposal)==0){
    if(length(proposal)!=0){
      proposal_new = proposal
    }else{
      message("No proposal distribution is specified.\n Default tuning parameters in random walk proposals will be used.\n")
    }
  }else{
    if(length(proposal)==0){
      proposal_new = obj@proposal
    }else{
      proposal_new = proposal
    }
  }

  prior_new = list()
  if(method=="Beta_prior"){
    if(length(obj@prior)==0){
      if(length(prior)!=0){
        prior_new = prior
      }else{
        message("No prior distribution is specified.\n Default tuning parameters in prior distributions will be used.\n")
      }
    }else{
      if(length(prior)==0){
        prior_new = obj@prior
      }else{
        prior_new = prior
      }
    }    
  }
   

  if(cov.model$form=="isotropic"){
    if(!exists("range", where=proposal_new)){
      proposal_new$range = 0.1
    }
    if(!exists("tail", where=proposal_new)){
      proposal_new$tail = 0.1
    }
    if(!exists("nu", where=proposal_new)){
      proposal_new$nu = 0.1
    }
    if(!exists("nugget", where=proposal_new)){
      proposal_new$nugget = 0.1
    }
  }else if(cov.model$form=="tensor" ||cov.model$form=="ARD"){

    if(!exists("range", where=proposal_new)){
      proposal_new$range = rep(0.1, Dim)
    }
    if(!exists("tail", where=proposal_new)){
      proposal_new$tail = rep(0.1, Dim)
    }
    if(!exists("nu", where=proposal_new)){
      proposal_new$nu = rep(0.1, Dim)
    }
    if(!exists("nugget", where=proposal_new)){
      proposal_new$nugget = 0.1
    }   

  }else{
    stop("gp.mcmc: the form of covariance function is not yet supported.\n")
  }

  obj@proposal = proposal_new

  if(method=="Cauchy_prior"){

    if(is.null(input.new)){
      MCMCsample = MCMCOBayes(output=output, H=H, input=input, par=param, 
                        covmodel=cov.model, smoothness_est=smooth.est,
                        proposal=proposal_new, nsample=nsample,
                        dtype=dtype, verbose=verbose)    
    }else{
      MCMCsample = MCMCOBayes_pred(output=output, H=H, input=input, 
                        input_new=input.new, Hnew=Hnew, par=param, 
                        covmodel=cov.model, smoothness_est=smooth.est,
                        proposal=proposal_new, nsample=nsample,
                        dtype=dtype, verbose=verbose)
    MCMCsample$pred = simplify2array(MCMCsample$pred)
    }

  }else if(method=="Ref_prior"){
    if(is.null(input.new)){
      MCMCsample = MCMCOBayesRef(output=output, H=H, input=input, par=param, 
                        covmodel=cov.model, smoothness_est=smooth.est,
                        proposal=proposal_new, nsample=nsample,
                        dtype=dtype, verbose=verbose)    
    }else{
      MCMCsample = MCMCOBayesRef_pred(output=output, H=H, input=input, 
                        input_new=input.new, Hnew=Hnew, par=param, 
                        covmodel=cov.model, smoothness_est=smooth.est,
                        proposal=proposal_new, nsample=nsample,
                        dtype=dtype, verbose=verbose)
    MCMCsample$pred = simplify2array(MCMCsample$pred)
    }

  }else if(method=="Beta_prior"){


    if(!exists("range", where=prior_new)){
      if(cov.model$form=="isotropic"){
        prior_new$range = list(a=1.0, b=1.0, lb=0, ub=3*info$max.distance)
      }else{
        prior_new$range = list(a=rep(1.0, Dim), b=rep(1.0, Dim),
                          lb=rep(0, Dim), ub=rep(3*info$max.distance, Dim))
      }
    }

    if(!exists("tail", where=prior_new)){
      if(cov.model$form=="isotropic"){
        prior_new$tail = list(a=1.0, b=1.0, lb=0, ub=5)
      }else{
        prior_new$tail = list(a=rep(1.0, Dim), b=rep(1.0, Dim),
                          lb=rep(0, Dim), ub=rep(5, Dim))
      }      
    }

    if(!exists("nugget", where=prior_new)){
      if(cov.model$form=="isotropic"){
        prior_new$nugget = list(a=1, b=1, lb=0, ub=10)
      }else{
        prior_new$nugget = list(a=rep(1.0, Dim), b=rep(1.0, Dim),
                        lb=rep(0, Dim), ub=rep(10, Dim))
      }
    }

    if(!exists("nu", where=prior_new)){
      if(cov.model$form=="isotropic"){
        prior_new$nu = list(a=1, b=1, lb=0, ub=4)
        if(cov.model$family=="cauchy" || cov.model$family=="powexp" || cov.model$family=="gauss"){
          prior_new$nu = list(a=1, b=1, lb=0, ub=2)
        }
      }else{
        prior_new$nu = list(a=rep(1.0, Dim), b=rep(1.0, Dim),
                        lb=rep(0, Dim), ub=rep(6, Dim))
        if(cov.model$family=="cauchy" || cov.model$family=="powexp" || cov.model$family=="gauss"){
          prior_new$nu = list(a=rep(1.0, Dim), b=rep(1.0, Dim),
                        lb=rep(0, Dim), ub=rep(2, Dim))
        }
      }
    }
    obj@prior = prior_new 

    if(is.null(input.new)){
      MCMCsample = MCMCSBayes(output=output, H=H, input=input, par=param, 
                        covmodel=cov.model, smoothness_est=smooth.est,
                        prior=prior_new, proposal=proposal_new, nsample=nsample,
                        dtype=dtype, verbose=verbose) 
    }else{
      MCMCsample = MCMCSBayes_pred(output=output, H=H, input=input, 
                        input_new=input.new, Hnew=Hnew, par=param,
                        covmodel=cov.model, smoothness_est=smooth.est,
                        prior=prior_new, proposal=proposal_new, nsample=nsample,
                        dtype=dtype, verbose=verbose) 
      MCMCsample$pred = simplify2array(MCMCsample$pred)     
    }
   
  }else{
    stop("gp.mcmc: Unsupported MCMC estimation method!\n")
  }

  if(!smooth.est){
    if(exists("nu", where=MCMCsample)){
      MCMCsample$nu = NULL 
    }
  }

  obj@mcmc = MCMCsample
  
  
  return(obj)
  
}


###########################################################################################
##########################################################################################



###########################################################################################
##########################################################################################
#' @title A wraper to fit a Gaussian stochastic process model with optimization methods 
#' @description This function is a wraper to estimate parameters in the GaSP model with different
#' choices of estimation methods using numerical optimization methods. 
#' 
#' @param obj an \code{S4} object \linkS4class{gp}
#' @param method a string indicating the parameter estimation method:
#' \describe{
#' \item{MPLE}{This indicates that the \emph{maximum profile likelihood estimation} 
#' (\strong{MPLE}) is used.}
#' \item{MMLE}{This indicates that the \emph{maximum marginal likelihood estimation} 
#' (\strong{MMLE}) is used.}
#' \item{MAP}{This indicates that the marginal/integrated posterior is maximized.}
#' }
#' @param opt a list of arguments to setup the \code{\link[stats]{optim}} routine. Current implementation uses three arguments: 
#' \describe{
#'  \item{method}{The optimization method: \code{Nelder-Mead} or \code{L-BFGS-B}.}
#' \item{lower}{The lower bound for parameters.}
#' \item{upper}{The upper bound for parameters.}
#'}
#' @param bound Default value is \code{NULL}. Otherwise, it should be a list
#' containing the following elements depending on the covariance class:
#' \describe{
#' \item{\strong{nugget}}{a list of bounds for the nugget parameter.
#' It is a list containing lower bound \strong{lb} and 
#' upper bound \strong{ub} with default value 
#' \code{list(lb=0, ub=Inf)}.}
#' \item{\strong{range}}{a list of bounds for the range parameter. Tt has default value
#' \code{range=list(lb=0, ub=Inf)} for the Confluent Hypergeometric covariance, the Matérn covariance, exponential covariance, Gaussian 
#' covariance, powered-exponential covariance, and Cauchy covariance. The log of range parameterization
#'  is used: \eqn{\log(\phi)}.}
#' \item{\strong{tail}}{a list of bounds for the tail decay parameter. It has default value
#' \code{list(lb=0, ub=Inf)}} for the Confluent Hypergeometric covariance and the Cauchy covariance.
#'  \item{\strong{nu}}{a list of bounds for the smoothness parameter. It has default value 
#' \code{list(lb=0, ub=Inf)} for the Confluent Hypergeometric covariance and the Matérn covariance.
#' when the powered-exponential or Cauchy class 
#' is used, it has default value \strong{nu}=\code{list(lb=0, ub=2)}. 
#' This can be achived by specifying the \strong{lower} bound in \code{opt}.}
#' }
#'
#' @return a list of updated \code{\link{gp}} object \strong{obj} and 
#' fitted information \strong{fit}
#' 
#' @author Pulong Ma \email{mpulong@@gmail.com}
#' 
#' @seealso \link{GPBayes-package}, \code{\link{GaSP}}, \linkS4class{gp}, \code{\link{gp.mcmc}}
#' @export
#' 
#' 
#' @examples 
#'  
#' code = function(x){
#' y = (sin(pi*x/5) + 0.2*cos(4*pi*x/5))*(x<=9.6) + (x/10-1)*(x>9.6) 
#' return(y)
#' }
#' n=100
#' input = seq(0, 20, length=n)
#' XX = seq(0, 20, length=99)
#' Ztrue = code(input)
#' set.seed(1234)
#' output = Ztrue + rnorm(length(Ztrue), sd=0.1)
#' obj = gp(formula=~1, output, input, 
#'         param=list(range=4, nugget=0.1,nu=2.5),
#'         smooth.est=FALSE,
#'         cov.model=list(family="matern", form="isotropic"))
#'         
#' fit.optim = gp.optim(obj, method="MPLE")
#' 
#' 
gp.optim <- function(obj, method="MMLE", opt=NULL, bound=NULL){


  if(method=="MPLE"){
    fit = MPLE(obj=obj, opt=opt, bound=bound)
  }else if(method=="MMLE"){
    fit = MMLE(obj=obj, opt=opt, bound=bound)
  }else if(method=="MAP"){
    stop("To be supported.\n")
  }else{
    stop("gp.optim: unsupported estimation method is specified! Please use any of the three methods: MPLE, MMLE, MAP\n")
  }
  
  # par = fit$par
  # cov.model = obj@cov.model
  # family = cov.model$family
  # obj@param$sig2 = par$sig2
  # obj@param$nugget = par$nugget

  
  # if(family=="CH" || family=="cauchy"){
  #   obj@param$range = par$range
  #   obj@param$tail = par$tail
  #   obj@param$nugget = par$nugget 
  #   obj@param$nu = par$nu 
  # }else if(family=="matern" || family=="exp" || family=="matern_3_2" || family=="matern_5_2" || family=="gauss" || family=="powexp"){
  #   obj@param$range = par$range
  #   obj@param$nugget = par$nugget
  #   obj@param$nu = par$nu 
  # }else{
  #   stop("gp.optim: Unsupported covariance family!\n")
  # }

  return(list(obj=fit$obj, fit=fit$fit))
  
}


###########################################################################################
##########################################################################################


###########################################################################################
##########################################################################################
#' @title Prediction at new inputs based on a Gaussian stochastic process model
#' @description This function provides the capability to make prediction based on a GaSP
#' when different estimation methods are employed. 
#' 
#' @param obj an \code{S4} object \linkS4class{gp}
#' @param input.new a matrix of new input lomessageions
#' @param method a string indicating the parameter estimation method:
#' \describe{
#' \item{MPLE}{This indicates that the \emph{maximum profile likelihood estimation} 
#' (\strong{MPLE}) is used. This correponds to simple kriging formulas}
#' \item{MMLE}{This indicates that the \emph{maximum marginal likelihood estimation} 
#' (\strong{MMLE}) is used. This corresponds to universal kriging formulas when the vairance
#' parameter is not integrated out. If the variance parameter is integrated out, 
#' the predictive variance differs from the universal kriging variance by the 
#' factor \eqn{\frac{n-q}{n-q-2}}, since the predictive distribution is a 
#' Student's \eqn{t}-distribution with degrees of freedom \eqn{n-q}.
#'  }
#' \item{MAP}{This indicates that the posterior estimates of model parameters are plugged into 
#' the posterior predictive distribution. Thus this approach does not take account into uncertainty 
#' in model parameters (\strong{range}, \strong{tail}, \strong{nu}, \strong{nugget}).}
#' \item{Bayes}{This indicates that a fully Bayesian approach is used
#' for parameter estimation (and hence prediction). This approach takes into account uncertainty in 
#' all model parameters.}
#' }
#'
#' @author Pulong Ma \email{mpulong@@gmail.com}
#' 
#' @seealso \link{GPBayes-package}, \code{\link{GaSP}}, \linkS4class{gp}, \code{\link{gp.mcmc}}, \code{\link{gp.optim}}
#' @export
#' @return a list of predictive mean, predictive standard deviation, 95% predictive intervals
#' 
#' @examples 
#'
#'  
#' code = function(x){
#' y = (sin(pi*x/5) + 0.2*cos(4*pi*x/5))*(x<=9.6) + (x/10-1)*(x>9.6) 
#' return(y)
#' }
#' n=100
#' input = seq(0, 20, length=n)
#' XX = seq(0, 20, length=99)
#' Ztrue = code(input)
#' set.seed(1234)
#' output = Ztrue + rnorm(length(Ztrue), sd=0.1)
#' obj = gp(formula=~1, output, input, 
#'         param=list(range=4, nugget=0.1,nu=2.5),
#'         smooth.est=FALSE,
#'         cov.model=list(family="matern", form="isotropic"))
#'  
#' fit.optim = gp.optim(obj, method="MMLE")
#' obj = fit.optim$obj
#' pred = gp.predict(obj, input.new=XX, method="MMLE")
#'                    
#'                    
#'                    
#'                    
#' 
gp.predict <- function(obj, input.new, method="Bayes"){
  
  formula = obj@formula
  output = obj@output
  input = obj@input
  param = obj@param
  cov.model = obj@cov.model
  family = cov.model$family 
  form = cov.model$form 
  dtype = obj@dtype
  MCMCsample = obj@mcmc 
  smooth.est = obj@smooth.est 
  
  if(!is.matrix(input.new)){
    message("Converting input.new to a matrix...\n")
    input.new = as.matrix(input.new)
  }
  
  Dim = dim(input)[2]
  colnames(input) = paste0("x", 1:Dim)
  df = data.frame(input)
  H = model.matrix(formula, df)
  
  colnames(input.new) = paste0("x", 1:Dim)
  df = data.frame(input.new)
  Hnew = model.matrix(formula, df)
  
  n = nrow(output)
  p = ncol(H)

  if(method=="MPLE"){
    sig2 = param$sig2 
    range = param$range 
    tail = param$tail 
    nu = param$nu 
    nugget = param$nugget 

    d = distance(input, input,form,dtype)
    cov.obs = kernel(d, range, tail, nu, cov.model)
    cov.obs = sig2*cov.obs + nugget*diag(nrow(cov.obs))
    dobspred = distance(input, input.new, form, dtype)
    cov.obspred = sig2*kernel(dobspred, range, tail, nu, cov.model)
    dpred = distance(input.new, input.new, form, dtype)
    cov.pred = kernel(dpred, range, tail, nu, cov.model)
    cov.pred = sig2*cov.pred + nugget*diag(nrow(cov.pred))

    bhat = param$coeff
    QInv = chol2inv(chol(cov.obs))
    pred.mean = Hnew%*%bhat + t(cov.obspred)%*%(QInv%*%(output - H%*%bhat))
    pred.var = diag(cov.pred) - diag(t(cov.obspred)%*%QInv%*%cov.obspred)
    pred.result = list(mean=pred.mean, sd=pred.var^0.5, 
                      lower95=pred.mean-qnorm(0.975, 0,1)*pred.var^0.5,
                      upper95=pred.mean+qnorm(0.975, 0,1)*pred.var^0.5)
    pred.result = lapply(pred.result, drop)
    
  }else if(method=="MMLE"){
    sig2 = param$sig2 
    range = param$range 
    tail = param$tail 
    nu = param$nu 
    nugget = param$nugget 

    par_MMLE = list(range=range, tail=tail, nu=nu, nugget=nugget)
    pred.result = GPpredict(output,H,input,input.new,Hnew,par_MMLE,cov.model,dtype)
    pred.result = lapply(pred.result, drop)

    # d = distance(input, input,form,dtype)
    # cov.obs = kernel(d, range, tail, nu, cov.model)
    # cov.obs = sig2*cov.obs + nugget*diag(nrow(cov.obs))
    # dobspred = distance(input, input.new, form, dtype)
    # cov.obspred = sig2*kernel(dobspred, range, tail, nu, cov.model)
    # dpred = distance(input.new, input.new, form, dtype)
    # cov.pred = kernel(dpred, range, tail, nu, cov.model)
    # cov.pred = sig2*cov.pred + nugget*diag(nrow(cov.pred))

    # bhat = param$coeff
    # QInv = chol2inv(chol(cov.obs))
    # pred.mean = Hnew%*%bhat + t(cov.obspred)%*%(QInv%*%(output - H%*%bhat))
    # XXRR = t(Hnew) - t(H)%*%QInv%*%cov.obspred
    # pred.var = diag(cov.pred) - diag(t(cov.obspred)%*%QInv%*%cov.obspred) + 
    #            colSum(XXRR*solve(t(H)%*%QInv%*%H, XXRR))
    # pred.result = list(mean=pred.mean, sd=pred.var^0.5,
    #                   lower95=pred.mean-qt(0.975, df=n-p)*sd,
    #                   upper95=pred.mean+qt(0.975, df=n-p)*sd)

  }else if(method=="MAP"){
    
    if(is.vector(MCMCsample$range)){
      range = mean(MCMCsample$range)
    }else if(is.matrix(MCMCsample$range)){
      range = rowMeans(MCMCsample$range)
    }

    if(!exists("tail", where=MCMCsample)){
      tail = param$tail 
    }else{
      if(is.vector(MCMCsample$tail)){
        tail = mean(MCMCsample$tail)
      }else if(is.matrix(MCMCsample$tail)){
        tail = rowMeans(MCMCsample$tail)
      }
    }

    if(!exists("nugget", where=MCMCsample)){
      nugget = param$nugget
    }else{
      nugget = mean(MCMCsample$nugget)
    }

    if(smooth.est){
      if(is.vector(MCMCsample$nu)){
        nu = mean(MCMCsample$nu)
      }else if(is.matrix(MCMCsample$nu)){
        nu = mean(MCMCsample$nu[,1])
      }
    }else{
      nu = param$nu 
    }

    par_MAP = list(range=range, tail=tail, nu=nu, nugget=nugget)
    pred.result = GPpredict(output,H,input,input.new,Hnew,par_MAP,cov.model,dtype)
    pred.result = lapply(pred.result, drop)

  }else if(method=="Bayes"){

    nu = param$nu
    pred.list = post_predictive_sampling(output,H,input,input.new,Hnew,MCMCsample,nu,smooth.est,cov.model,dtype) 
    predmat = simplify2array(pred.list)
    predmean = apply(predmat, c(1,2), mean)
    predsd = apply(predmat, c(1,2), sd)
    lower95 = apply(predmat, c(1,2), quantile, 0.025)
    upper95 = apply(predmat, c(1,2), quantile, 0.975)

    pred.result = list(mean=drop(predmean), sd=drop(predsd), lower95=drop(lower95), 
                  upper95=drop(upper95), samples=predmat)
  }else{
    stop("gp.predict: unsupported method.")
  }
  
  return(pred.result)
}

###########################################################################################
##########################################################################################

###########################################################################################
##########################################################################################
#' @title Simulate from a Gaussian stochastic process model
#' @description This function simulates realizations from Gaussian processes. 
#' 
#' @param formula an object of \code{formula} class that specifies regressors; see \code{\link[stats]{formula}} for details.
#' @param input a matrix including inputs in a GaSP
#' 
#' @param cov.model a list of two strings: \strong{family}, \strong{form}, where \strong{family} indicates the family of covariance functions 
#' including the Confluent Hypergeometric class, the Matérn class, the Cauchy class, the powered-exponential class. \strong{form} indicates the 
#' specific form of covariance structures including the isotropic form, tensor form, automatic relevance determination form. 
#' \describe{
#' \item{\strong{family}}{
#' \describe{
#' \item{CH}{The Confluent Hypergeometric correlation function is given by 
#' \deqn{C(h) = \frac{\Gamma(\nu+\alpha)}{\Gamma(\nu)} 
#' \mathcal{U}\left(\alpha, 1-\nu, \left(\frac{h}{\beta}\right)^2\right),}
#' where \eqn{\alpha} is the tail decay parameter. \eqn{\beta} is the range parameter.
#' \eqn{\nu} is the smoothness parameter. \eqn{\mathcal{U}(\cdot)} is the confluent hypergeometric
#' function of the second kind. For details about this covariance, 
#' see Ma and Bhadra (2019) at \url{https://arxiv.org/abs/1911.05865}.  
#' }
#' \item{cauchy}{The generalized Cauchy covariance is given by
#' \deqn{C(h) = \left\{ 1 + \left( \frac{h}{\phi} \right)^{\nu}  
#'             \right\}^{-\alpha/\nu},}
#' where \eqn{\phi} is the range parameter. \eqn{\alpha} is the tail decay parameter.
#' \eqn{\nu} is the smoothness parameter with default value at 2.
#'}
#'
#' \item{matern}{The Matérn correlation function is given by
#' \deqn{C(h)=\frac{2^{1-\nu}}{\Gamma(\nu)} \left( \frac{h}{\phi} \right)^{\nu} 
#' \mathcal{K}_{\nu}\left( \frac{h}{\phi} \right),}
#' where \eqn{\phi} is the range parameter. \eqn{\nu} is the smoothness parameter. 
#' \eqn{\mathcal{K}_{\nu}(\cdot)} is the modified Bessel function of the second kind of order \eqn{\nu}.
#' }
#' \item{exp}{The exponential correlation function is given by 
#' \deqn{C(h)=\exp(-h/\phi),}
#' where \eqn{\phi} is the range parameter. This is the Matérn correlation with \eqn{\nu=0.5}.
#' }
#' \item{matern_3_2}{The Matérn correlation with \eqn{\nu=1.5}.}
#' \item{matern_5_2}{The Matérn correlation with \eqn{\nu=2.5}.}
#'
#'
#' \item{powexp}{The powered-exponential correlation function is given by
#'                \deqn{C(h)=\exp\left\{-\left(\frac{h}{\phi}\right)^{\nu}\right\},}
#' where \eqn{\phi} is the range parameter. \eqn{\nu} is the smoothness parameter.
#' }
#' \item{gauss}{The Gaussian correlation function is given by 
#' \deqn{C(h)=\exp\left(-\frac{h^2}{\phi^2}\right),}
#' where \eqn{\phi} is the range parameter.
#'  }
#' }
#' }
#' 
#' \item{\strong{form}}{
#' \describe{
#'  \item{isotropic}{This indicates the isotropic form of covariance functions. That is,
#'  \deqn{C(\mathbf{h}) = C^0(\|\mathbf{h}\|; \boldsymbol \theta),} where \eqn{\| \mathbf{h}\|} denotes the 
#' Euclidean distance or the great circle distance for data on sphere. \eqn{C^0(\cdot)} denotes 
#' any isotropic covariance family specified in \strong{family}.}
#'  \item{tensor}{This indicates the tensor product of correlation functions. That is, 
#' \deqn{ C(\mathbf{h}) = \prod_{i=1}^d C^0(|h_i|; \boldsymbol \theta_i),}
#' where \eqn{d} is the dimension of input space. \eqn{h_i} is the distance along the \eqn{i}th input dimension. This type of covariance structure has been often used in Gaussian process emulation for computer experiments.
#'}
#'  \item{ARD}{This indicates the automatic relevance determination form. That is, 
#' \deqn{C(\mathbf{h}) = C^0\left(\sqrt{\sum_{i=1}^d\frac{h_i^2}{\phi^2_i}}; \boldsymbol \theta \right),}
#' where \eqn{\phi_i} denotes the range parameter along the \eqn{i}th input dimension.}
#'  }
#' }
#'
#'}
#' 
#' @param param a list including values for regression parameters, covariance parameters, 
#' and nugget variance parameter.
#' The specification of \strong{param} should depend on the covariance model. 
#' \itemize{
#' \item{The regression parameters are denoted by \strong{coeff}. Default value is \eqn{\mathbf{0}}.}
#' \item{The marginal variance or partial sill is denoted by \strong{sig2}. Default value is 1.}
#' \item{The nugget variance parameter is denoted by \strong{nugget} for all covariance models. 
#' Default value is 0.}
#' \item{For the Confluent Hypergeometric class, \strong{range} is used to denote the range parameter \eqn{\beta}. 
#' \strong{tail} is used to denote the tail decay parameter \eqn{\alpha}. \strong{nu} is used to denote the 
#' smoothness parameter \eqn{\nu}.}
#' \item{For the generalized Cauchy class, \strong{range} is used to denote the range parameter \eqn{\phi}. 
#' \strong{tail} is used to denote the tail decay parameter \eqn{\alpha}. \strong{nu} is used to denote the 
#' smoothness parameter \eqn{\nu}.}
#' \item{For the Matérn class, \strong{range} is used to denote the range parameter \eqn{\phi}. 
#' \strong{nu} is used to denote the smoothness parameter \eqn{\nu}. When \eqn{\nu=0.5}, the 
#' Matérn class corresponds to the exponential covariance.}  
#' \item{For the powered-exponential class, \strong{range} is used to denote the range parameter \eqn{\phi}.
#' \strong{nu} is used to denote the smoothness parameter. When \eqn{\nu=2}, the powered-exponential class
#' corresponds to the Gaussian covariance.}
#' }
#' 
#'
#' @param dtype a string indicating the type of distance:
#' \describe{
#' \item{Euclidean}{Euclidean distance is used. This is the default choice.}
#' \item{GCD}{Great circle distance is used for data on sphere.}
#'}

#' @param nsample an integer indicating the number of realizations from a Gaussian process
#' @param seed a number specifying random number seed
#'
#' @author Pulong Ma \email{mpulong@@gmail.com}
#' 
#' @seealso \link{GPBayes-package}, \code{\link{GaSP}}, \linkS4class{gp}
#' @export
#' @return a numerical vector or a matrix 
#' @examples 
#' 
#' n=50
#' y.sim = gp.sim(input=seq(0,1,length=n),
#'                param=list(range=0.5,nugget=0.1,nu=2.5),
#'                cov.model=list(family="matern",form="isotropic"),
#'                seed=123)
#' 

gp.sim <- function(formula=~1, input, param, cov.model=list(family="CH", 
          form="isotropic"), dtype="Euclidean", nsample=1, seed=NULL){

  check.out = .check.arg.gp.sim(formula, input, param, cov.model, dtype)
  input = check.out$input 

  Dim = dim(input)[2]
  colnames(input) = paste0("x", 1:Dim)
  df = data.frame(input)
  H = model.matrix(formula, df)

  if(!is.null(seed)){
    set.seed(seed)
  }

  sim = GPsim(input, H, param, cov.model, nsample, dtype)


  return(drop(sim))

}


#####################################################################
#####################################################################
.check.arg.gp.sim <- function(formula, input, param, cov.model, dtype){
  
  message("\n")
  if(!is(formula, "formula")){
    stop("Please specify a formula to extract the design/basis matrix.\n")
  }
  
  
  if(!is(input, "matrix")){
    message("coerce input to a matrix format.\n")
    input = as.matrix(input)
  }


  if(!is(cov.model, "list")){
    message("set default values for cov.model: family=`CH`, form=`isotropic'.\n")
    cov.model = list(family="CH", form="isotropic")
  }else{
    if(!exists("family", where=cov.model)){
      stop("gp.sim: the covariance family is not specified.\n")
    }
    if(!exists("form", where=cov.model)){
      stop("gp.sim: the form of covariance structure is not specified.\n")
    }
  }

  
  Dim = dim(input)[2]
  colnames(input) = paste0("x", 1:Dim)
  df = data.frame(input)
  H = model.matrix(formula, df)
  p = dim(H)[2]

  d = distance(input, input, type=cov.model$form, dtype);

  if(!is(param, "list")){
    stop("param should be a list containing initial values for  
         correlation parameters and nugget variance parameter.\n")
  }else{
    if(!exists("coeff", where=param)){
      param$coeff = rep(0, p)
    }

    if(!exists("sig2", where=param)){
      param$sig2 = 1 
    }
    
    if(!exists("nugget", where=param)){
      param$nugget = 0 
    }

    if(!exists("range", where=param)){
      if(cov.model$form=="isotropic"){
        param$range = max(d)/2
      }else{
        param$range = sapply(d, max)/2
      }
      
    }

    if(!exists("tail", where=param)){
      if(cov.model$form=="isotropic"){
        param$tail = 0.5
      }else{
        param$tail = rep(0.5, Dim)
      }
    }

    if(!exists("nu", where=param)){
      if(cov.model$form=="isotropic"){
        param$nu = 0.5
      }else{
        param$tail = rep(0.5, Dim)
      }
    }
    
  }

  if(!is(dtype, "character")){
    stop("dtype: distance type is not correctly specified.\n")
  }


  
  return(list(input=input, param=param))
}

###########################################################################################
##########################################################################################

#' @title Building, fitting, predicting for a GaSP model 
#' @description This function serves as a wrapper to build, fit, and make prediction 
#' for a Gaussian process model. It calls on functions \code{\link{gp}}, \code{\link{gp.mcmc}},
#' \code{\link{gp.optim}}, \code{\link{gp.predict}}.
#' @param formula an object of \code{formula} class that specifies regressors; see \code{\link[stats]{formula}} for details.
#' @param output a numerical vector including observations or outputs in a GaSP
#' @param input a matrix including inputs in a GaSP
#' 
#' @param param a list including values for regression parameters, covariance parameters, 
#' and nugget variance parameter.
#' The specification of \strong{param} should depend on the covariance model. 
#' \itemize{
#' \item{The regression parameters are denoted by \strong{coeff}. Default value is \eqn{\mathbf{0}}.}
#' \item{The marginal variance or partial sill is denoted by \strong{sig2}. Default value is 1.}
#' \item{The nugget variance parameter is denoted by \strong{nugget} for all covariance models. 
#' Default value is 0.}
#' \item{For the Confluent Hypergeometric class, \strong{range} is used to denote the range parameter \eqn{\beta}. 
#' \strong{tail} is used to denote the tail decay parameter \eqn{\alpha}. \strong{nu} is used to denote the 
#' smoothness parameter \eqn{\nu}.}
#' \item{For the generalized Cauchy class, \strong{range} is used to denote the range parameter \eqn{\phi}. 
#' \strong{tail} is used to denote the tail decay parameter \eqn{\alpha}. \strong{nu} is used to denote the 
#' smoothness parameter \eqn{\nu}.}
#' \item{For the Matérn class, \strong{range} is used to denote the range parameter \eqn{\phi}. 
#' \strong{nu} is used to denote the smoothness parameter \eqn{\nu}. When \eqn{\nu=0.5}, the 
#' Matérn class corresponds to the exponential covariance.}  
#' \item{For the powered-exponential class, \strong{range} is used to denote the range parameter \eqn{\phi}.
#' \strong{nu} is used to denote the smoothness parameter. When \eqn{\nu=2}, the powered-exponential class
#' corresponds to the Gaussian covariance.}
#' }
#' @param input.new a matrix of new input locations
#' @param smooth.est a logical value indicating whether smoothness parameter will be estimated.
#'
#' @param cov.model a list of two strings: \strong{family}, \strong{form}, where \strong{family} indicates the family of covariance functions 
#' including the Confluent Hypergeometric class, the Matérn class, the Cauchy class, the powered-exponential class. \strong{form} indicates the 
#' specific form of covariance structures including the isotropic form, tensor form, automatic relevance determination form. 
#' \describe{
#' \item{\strong{family}}{
#' \describe{
#' \item{CH}{The Confluent Hypergeometric correlation function is given by 
#' \deqn{C(h) = \frac{\Gamma(\nu+\alpha)}{\Gamma(\nu)} 
#' \mathcal{U}\left(\alpha, 1-\nu, \left(\frac{h}{\beta}\right)^2\right),}
#' where \eqn{\alpha} is the tail decay parameter. \eqn{\beta} is the range parameter.
#' \eqn{\nu} is the smoothness parameter. \eqn{\mathcal{U}(\cdot)} is the confluent hypergeometric
#' function of the second kind. For details about this covariance, 
#' see Ma and Bhadra (2019) at \url{https://arxiv.org/abs/1911.05865}.  
#' }
#' \item{cauchy}{The generalized Cauchy covariance is given by
#' \deqn{C(h) = \left\{ 1 + \left( \frac{h}{\phi} \right)^{\nu}  
#'             \right\}^{-\alpha/\nu},}
#' where \eqn{\phi} is the range parameter. \eqn{\alpha} is the tail decay parameter.
#' \eqn{\nu} is the smoothness parameter with default value at 2.
#'}
#'
#' \item{matern}{The Matérn correlation function is given by
#' \deqn{C(h)=\frac{2^{1-\nu}}{\Gamma(\nu)} \left( \frac{h}{\phi} \right)^{\nu} 
#' \mathcal{K}_{\nu}\left( \frac{h}{\phi} \right),}
#' where \eqn{\phi} is the range parameter. \eqn{\nu} is the smoothness parameter. 
#' \eqn{\mathcal{K}_{\nu}(\cdot)} is the modified Bessel function of the second kind of order \eqn{\nu}.
#' }
#' \item{exp}{The exponential correlation function is given by 
#' \deqn{C(h)=\exp(-h/\phi),}
#' where \eqn{\phi} is the range parameter. This is the Matérn correlation with \eqn{\nu=0.5}.
#' }
#' \item{matern_3_2}{The Matérn correlation with \eqn{\nu=1.5}.}
#' \item{matern_5_2}{The Matérn correlation with \eqn{\nu=2.5}.}
#'
#'
#' \item{powexp}{The powered-exponential correlation function is given by
#'                \deqn{C(h)=\exp\left\{-\left(\frac{h}{\phi}\right)^{\nu}\right\},}
#' where \eqn{\phi} is the range parameter. \eqn{\nu} is the smoothness parameter.
#' }
#' \item{gauss}{The Gaussian correlation function is given by 
#' \deqn{C(h)=\exp\left(-\frac{h^2}{\phi^2}\right),}
#' where \eqn{\phi} is the range parameter.
#'  }
#' }
#' }
#' 
#' \item{\strong{form}}{
#' \describe{
#'  \item{isotropic}{This indicates the isotropic form of covariance functions. That is,
#'  \deqn{C(\mathbf{h}) = C^0(\|\mathbf{h}\|; \boldsymbol \theta),} where \eqn{\| \mathbf{h}\|} denotes the 
#' Euclidean distance or the great circle distance for data on sphere. \eqn{C^0(\cdot)} denotes 
#' any isotropic covariance family specified in \strong{family}.}
#'  \item{tensor}{This indicates the tensor product of correlation functions. That is, 
#' \deqn{ C(\mathbf{h}) = \prod_{i=1}^d C^0(|h_i|; \boldsymbol \theta_i),}
#' where \eqn{d} is the dimension of input space. \eqn{h_i} is the distance along the \eqn{i}th input dimension. This type of covariance structure has been often used in Gaussian process emulation for computer experiments.
#'}
#'  \item{ARD}{This indicates the automatic relevance determination form. That is, 
#' \deqn{C(\mathbf{h}) = C^0\left(\sqrt{\sum_{i=1}^d\frac{h_i^2}{\phi^2_i}}; \boldsymbol \theta \right),}
#' where \eqn{\phi_i} denotes the range parameter along the \eqn{i}th input dimension.}
#'  }
#' }
#'
#'}
#' 
#'
#' @param model.fit a string indicating the choice of priors on correlation parameters:
#' \describe{
#' \item{Cauchy_prior}{This indicates that a fully Bayesian approach with objective priors is used
#' for parameter estimation, where location-scale parameters are assigned with constant priors and 
#' correlation parameters are assigned with half-Cauchy priors (default). 
#' }
#' \item{Ref_prior}{This indicates that a fully Bayesian approach with objective priors is used
#' for parameter estimation, where location-scale parameters are assigned with constant priors and 
#' correlation parameters are assigned with reference priors. This is only supported for isotropic
#' covariance functions. For details, see \code{\link{gp.mcmc}}.
#' }
#' \item{Beta_prior}{This indicates that a fully Bayesian approach with subjective priors is used
#' for parameter estimation, where location-scale parameters are assigned with constant priors and 
#' correlation parameters are assigned with \link{beta} priors parameterized as \eqn{Beta(a, b, lb, ub)}.
#' In the beta distribution, \strong{lb} and \strong{ub} are the support for correlation parameters, and
#' they should be determined based on domain knowledge. \strong{a} and \strong{b} are two shape parameters with default values at 1,
#' corresponding to the uniform prior over the support \eqn{(lb, ub)}. 
#' }
#' \item{MPLE}{This indicates that the \emph{maximum profile likelihood estimation} 
#' (\strong{MPLE}) is used.}
#' \item{MMLE}{This indicates that the \emph{maximum marginal likelihood estimation} 
#' (\strong{MMLE}) is used.}
#' \item{MAP}{This indicates that the marginal/integrated posterior is maximized.}
#' }
#' @param prior a list containing tuning parameters in prior distribution. This is used only if a subjective Bayes estimation method with informative priors is used.
#' @param proposal a list containing tuning parameters in proposal distribution. This is used only if a Bayes estimation method is used.
#' @param nsample an integer indicating the number of MCMC samples. 
#' @param burnin an integer indicating the burn-in period.
#' @param opt a list of arguments to setup the \code{\link[stats]{optim}} routine. Current implementation uses three arguments: 
#' \describe{
#'  \item{method}{The optimization method: \code{Nelder-Mead} or \code{L-BFGS-B}.}
#' \item{lower}{The lower bound for parameters.}
#' \item{upper}{The upper bound for parameters.}
#'}
#' @param bound Default value is \code{NULL}. Otherwise, it should be a list
#' containing the following elements depending on the covariance class:
#' \describe{
#' \item{\strong{nugget}}{a list of bounds for the nugget parameter.
#' It is a list containing lower bound \strong{lb} and 
#' upper bound \strong{ub} with default value 
#' \code{list(lb=0, ub=Inf)}.}
#' \item{\strong{range}}{a list of bounds for the range parameter. It has default value
#' \code{range=list(lb=0, ub=Inf)} for the Confluent Hypergeometric covariance, the Matérn covariance, exponential covariance, Gaussian 
#' covariance, powered-exponential covariance, and Cauchy covariance. The log of range parameterization
#'  is used: \eqn{\log(\phi)}.}
#' \item{\strong{tail}}{a list of bounds for the tail decay parameter. It has default value
#' \code{list(lb=0, ub=Inf)}} for the Confluent Hypergeometric covariance and the Cauchy covariance.
#'  \item{\strong{nu}}{a list of bounds for the smoothness parameter. It has default value 
#' \code{list(lb=0, ub=Inf)} for the Confluent Hypergeometric covariance and the Matérn covariance.
#' when the powered-exponential or Cauchy class 
#' is used, it has default value \strong{nu}=\code{list(lb=0, ub=2)}. 
#' This can be achieved by specifying the \strong{lower} bound in \code{opt}.}
#' }
#' @param dtype a string indicating the type of distance:
#' \describe{
#' \item{Euclidean}{Euclidean distance is used. This is the default choice.}
#' \item{GCD}{Great circle distance is used for data on sphere.}
#'}
#' 
#' @param verbose a logical value. If it is \code{TRUE}, the MCMC progress bar is shown. 
#' @seealso \code{\link{GPBayes-package}}, \code{\link{gp}}, \code{\link{gp.mcmc}}, \code{\link{gp.optim}}, \code{\link{gp.predict}}
#' @author Pulong Ma \email{mpulong@@gmail.com}
#'  
#' @export 
#' @return a list containing the \code{S4} object \linkS4class{gp} and prediction results 
#' 
#' @examples 
#'
#'code = function(x){
#' y = (sin(pi*x/5) + 0.2*cos(4*pi*x/5))*(x<=9.6) + (x/10-1)*(x>9.6) 
#' return(y)
#' }
#' n=100
#' input = seq(0, 20, length=n)
#' XX = seq(0, 20, length=99)
#' Ztrue = code(input)
#' set.seed(1234)
#' output = Ztrue + rnorm(length(Ztrue), sd=0.1)
#' 
#'# fitting a GaSP model with the objective Bayes approach
#' fit = GaSP(formula=~1, output, input,  
#'           param=list(range=3, nugget=0.1, nu=2.5), 
#'           smooth.est=FALSE, input.new=XX,
#'           cov.model=list(family="matern", form="isotropic"),
#'           proposal=list(range=.35, nugget=.8, nu=0.8),
#'           dtype="Euclidean", model.fit="Cauchy_prior", nsample=50, 
#'           burnin=10, verbose=TRUE)
#' 
GaSP <- function(formula=~1, output, input, param, smooth.est=FALSE, 
              input.new=NULL,   
              cov.model=list(family="CH", form="isotropic"), 
              model.fit="Cauchy_prior", prior=list(),
              proposal=list(range=0.35, tail=2, nugget=.8, nu=.8),
              nsample=5000, burnin=1000, opt=NULL, bound=NULL, 
              dtype="Euclidean", verbose=TRUE){

  #message("Creating the gp object.\n")
  obj = gp(formula=formula, output=output, input=input, param=param, 
        smooth.est=smooth.est, cov.model=cov.model, dtype=dtype)
  Dim = ncol(obj@input)

  Result = list()
  if(model.fit=="Cauchy_prior" || model.fit=="Ref_prior"){

    message("Starting MCMC fitting...\n")
    fit.obj = gp.mcmc(obj, input.new=input.new, method=model.fit, proposal=proposal,
               nsample=nsample, verbose=verbose)
    message("Finish MCMC fitting...\n")
    if(exists("range", where=fit.obj@mcmc)){
      if(is.matrix(fit.obj@mcmc$range)){
        fit.obj@mcmc$range = fit.obj@mcmc$range[-c(1:burnin),]
      }else{
        fit.obj@mcmc$range = fit.obj@mcmc$range[-c(1:burnin)]
      }
    }
    if(exists("tail", where=fit.obj@mcmc)){
      if(is.matrix(fit.obj@mcmc$tail)){
        fit.obj@mcmc$tail = fit.obj@mcmc$tail[-c(1:burnin),]
      }else{
        fit.obj@mcmc$tail = fit.obj@mcmc$tail[-c(1:burnin)]
      }
    }
    if(exists("nugget", where=fit.obj@mcmc)){
      if(is.matrix(fit.obj@mcmc$nugget)){
        fit.obj@mcmc$nugget = fit.obj@mcmc$nugget[-c(1:burnin),]
      }else{
        fit.obj@mcmc$nugget = fit.obj@mcmc$nugget[-c(1:burnin)]
      }
    }
    if(exists("nu", where=fit.obj@mcmc)){
      if(is.matrix(fit.obj@mcmc$nu)){
        fit.obj@mcmc$nu = fit.obj@mcmc$nu[-c(1:burnin),]
      }else{
        fit.obj@mcmc$nu = fit.obj@mcmc$nu[-c(1:burnin)]
      }
    }

    if(!is.null(input.new)){
      #message("Making prediction ...\n")
      #pred = gp.predict(fit.obj, input.new=input.new, method="Bayes")
      predmat = fit.obj@mcmc$pred[,,-c(1:burnin),drop=FALSE] 
      predmean = apply(predmat, c(1,2), mean)
      predsd = apply(predmat, c(1,2), sd)
      lower95 = apply(predmat, c(1,2), quantile, 0.025)
      upper95 = apply(predmat, c(1,2), quantile, 0.975)  
      Result$pred = list(mean=drop(predmean),sd=drop(predsd),lower95=drop(lower95),
                            upper95=drop(upper95), samples=predmat)
      fit.obj@mcmc$pred = Result$pred 
    }
    fit.obj@info$MCMC=TRUE

  }else if(model.fit=="Beta_prior"){
    message("Starting MCMC fitting...\n")
    fit.obj = gp.mcmc(obj, input.new=input.new, method=model.fit, prior=prior, proposal=proposal,
               nsample=nsample, verbose=verbose)
    message("Finish MCMC fitting...\n")  

    if(exists("range", where=fit.obj@mcmc)){
      if(is.matrix(fit.obj@mcmc$range)){
        fit.obj@mcmc$range = fit.obj@mcmc$range[-c(1:burnin),]
      }else{
        fit.obj@mcmc$range = fit.obj@mcmc$range[-c(1:burnin)]
      }
    }
    if(exists("tail", where=fit.obj@mcmc)){
      if(is.matrix(fit.obj@mcmc$tail)){
        fit.obj@mcmc$tail = fit.obj@mcmc$tail[-c(1:burnin),]
      }else{
        fit.obj@mcmc$tail = fit.obj@mcmc$tail[-c(1:burnin)]
      }
    }
    if(exists("nugget", where=fit.obj@mcmc)){
      if(is.matrix(fit.obj@mcmc$nugget)){
        fit.obj@mcmc$nugget = fit.obj@mcmc$nugget[-c(1:burnin),]
      }else{
        fit.obj@mcmc$nugget = fit.obj@mcmc$nugget[-c(1:burnin)]
      }
    }
    if(exists("nu", where=fit.obj@mcmc)){
      if(is.matrix(fit.obj@mcmc$nu)){
        fit.obj@mcmc$nu = fit.obj@mcmc$nu[-c(1:burnin),]
      }else{
        fit.obj@mcmc$nu = fit.obj@mcmc$nu[-c(1:burnin)]
      }
    }

    if(!is.null(input.new)){
      #message("Making prediction ...\n")
      #pred = gp.predict(fit.obj, input.new=input.new, method="Bayes")
      predmat = fit.obj@mcmc$pred[,,-c(1:burnin),drop=FALSE] 
      predmean = apply(predmat, c(1,2), mean)
      predsd = apply(predmat, c(1,2), sd)
      lower95 = apply(predmat, c(1,2), quantile, 0.025)
      upper95 = apply(predmat, c(1,2), quantile, 0.975)  
      Result$pred = list(mean=drop(predmean),sd=drop(predsd),lower95=drop(lower95),
                            upper95=drop(upper95), samples=predmat)
      fit.obj@mcmc$pred = Result$pred 
    }     
    fit.obj@info$MCMC=TRUE

  }else if(model.fit=="MMLE"){
    message("Starting optimization ...\n")
    fit = gp.optim(obj, method="MMLE", opt=opt, bound=bound)
    message("Finish optimization ...\n")
    fit.obj = fit$obj
    if(!is.null(input.new)){
      pred = gp.predict(fit.obj, input.new=input.new, method="MMLE")
      Result$pred = pred
    }
    fit.obj@info$MMLE = fit$fit 
  }else if(model.fit=="MPLE"){
    message("Starting optimization ...\n")
    fit = gp.optim(obj, method="MPLE", opt=opt, bound=bound)
    message("Finish optimization ...\n")
    fit.obj = fit$obj
    if(!is.null(input.new)){
      pred = gp.predict(fit.obj, input.new=input.new, method="MPLE")
      Result$pred = pred
    }
    fit.obj@info$MPLE = fit$fit
  }else if(model.fit=="MAP"){
    message("Starting optimization ...\n")
    fit = gp.optim(obj, method="MAP", opt=opt, bound=bound)
    message("Finish optimization ...\n")
    fit.obj = fit$obj
    if(!is.null(input.new)){
      pred = gp.predict(fit.obj, input.new=input.new, method="MAP")
      Result$pred = pred
    }
    fit.obj@info$MAP = fit$fit   
  }else{
    stop("GaSP: The model fitting method is not implemented yet.\n")
  }

  Result$obj = fit.obj
  return(Result)

}


#' @title get posterior summary for MCMC samples
#' @description This function processes posterior samples in the \code{\link{gp}} object. 
#' 
#' @param obj a \code{\link{gp}} object 
#' @param burnin a numerical value specifying the burn-in period for calculating posterior summaries.
#' @seealso \link{GPBayes-package}, \code{\link{GaSP}}, \code{\link{gp}}, \code{\link{gp.mcmc}}
#' @export
#' @return a list of posterior summaries 
gp.get.mcmc = function(obj, burnin=500){

  post.stat = list()
  if(length(obj@mcmc)==0){
    stop("No posterior samples are available in the object.\n")
  }else{
    if(exists("range", where=obj@mcmc)){
      if(is.matrix(obj@mcmc$range)){
        post.stat$range = obj@mcmc$range[-c(1:burnin),]
        post.stat$accept_rate_range = colMeans(obj@mcmc$accept_rate_range[-c(1:burnin),])
      }else{
        post.stat$range = obj@mcmc$range[-c(1:burnin)]
        post.stat$accept_rate_range = mean(obj@mcmc$accept_rate_range[-c(1:burnin)])
      }
    }
    if(exists("tail", where=obj@mcmc)){
      if(is.matrix(obj@mcmc$tail)){
        post.stat$tail = obj@mcmc$tail[-c(1:burnin),]
        post.stat$accept_rate_tail = colMeans(obj@mcmc$accept_rate_tail[-c(1:burnin),])
      }else{
        post.stat$tail = obj@mcmc$tail[-c(1:burnin)]
        post.stat$accept_rate_tail = mean(obj@mcmc$accept_rate_tail[-c(1:burnin)])
      }
    }
    if(exists("nugget", where=obj@mcmc)){
      if(is.matrix(obj@mcmc$nugget)){
        post.stat$nugget = obj@mcmc$nugget[-c(1:burnin),]
        post.stat$accept_rate_nugget = colMeans(obj@mcmc$accept_rate_nugget[-c(1:burnin),])
      }else{
        post.stat$nugget = obj@mcmc$nugget[-c(1:burnin)]
        post.stat$accept_rate_nu = mean(obj@mcmc$accept_rate_nu[-c(1:burnin)])
      }
    }
    if(exists("nu", where=obj@mcmc)){
      if(is.matrix(obj@mcmc$nu)){
        post.stat$nu = obj@mcmc$nu[-c(1:burnin),]
        post.stat$accept_rate_nu = colMeans(obj@mcmc$accept_rate_nu[-c(1:burnin),])
      }else{
        post.stat$nu = obj@mcmc$nu[-c(1:burnin)]
        post.stat$accept_rate_nu = mean(obj@mcmc$accept_rate_nu[-c(1:burnin)])
      }
    }

    if(exists("pred", where=obj@mcmc)){
      predmat = obj@mcmc$pred[,,-c(1:burnin),drop=FALSE] 
      predmean = apply(predmat, c(1,2), mean)
      predsd = apply(predmat, c(1,2), sd)
      lower95 = apply(predmat, c(1,2), quantile, 0.025)
      upper95 = apply(predmat, c(1,2), quantile, 0.975)  
      post.stat$pred = list(mean=drop(predmean),sd=drop(predsd),lower95=drop(lower95),
                            upper95=drop(upper95), samples=predmat)
    }

  }

  return(stats=post.stat)

}




###########################################################################################
##########################################################################################
#' @title Fisher information matrix 
#' @description This function computes the Fisher information matrix \eqn{I(\sigma^2, \boldsymbol \theta)} for a 
#' Gaussian process model. 
#' The standard likelihood is defined as 
#' \deqn{ L(\sigma^2, \boldsymbol \theta; \mathbf{y}) = \mathcal{N}_n(\mathbf{H}\mathbf{b}, \sigma^2 \mathbf{R}),
#' }
#' where \eqn{\mathbf{y}:=(y(\mathbf{x}_1), \ldots, y(\mathbf{x}_n))^\top} is a vector of \eqn{n} observations.
#' \eqn{\mathbf{H}} is a matrix of covariates, \eqn{\mathbf{b}} is a vector of regression coefficients, 
#' \eqn{\sigma^2} is the variance parameter, \eqn{\boldsymbol \theta} contains correlation
#' parameters and nugget parameter, \eqn{\mathbf{R}} denotes the correlation matrix 
#' plus nugget variance on the main diagonal.
#' 
#' The integrated likelihood is defined as
#' \deqn{
#'  L^{I}(\sigma^2, \boldsymbol \theta; \mathbf{y}) = \int L(\mathbf{b}, \sigma^2, \boldsymbol \theta; \mathbf{y}) \pi^{R}(\mathbf{b} \mid \sigma^2, \boldsymbol \theta) d \mathbf{b},
#'}
#' where \eqn{\pi^{R}(\mathbf{b} \mid \sigma^2, \boldsymbol \theta)=1} is the conditional Jeffreys-rule (or reference prior) 
#' in the model with the above standard likelihood when \eqn{(\sigma^2, \boldsymbol \theta)} is assumed to be known.
#' 
#' \itemize{
#' \item{For the Matérn class, current implementation only computes Fisher information matrix 
#' for variance parameter \eqn{\sigma^2}, range parameter \eqn{\phi}, and nugget variance 
#' parameter \eqn{\tau^2}. That is, \eqn{I(\sigma^2, \boldsymbol \theta) = I(\sigma^2, \phi, \tau^2)}.
#' }
#' \item{For the Confluent Hypergeometric class, current implementation computes Fisher information matrix
#'  for variance parameter \eqn{\sigma^2}, range parameter \eqn{\beta}, tail decay parameter \eqn{\alpha}, smoothness parameter \eqn{\nu} and nugget variance 
#' parameter \eqn{\tau^2}. That is, \eqn{I(\sigma^2, \boldsymbol \theta) = I(\sigma^2, \beta, \alpha, \nu, \tau^2)}.
#' }
#'}
#'
#' @param obj a \code{\link{gp}} object. It is optional with default value \code{NULL}.
#' @param intloglik a logical value with default value \code{FALSE}. If it is \code{FALSE}, Fisher information matrix \eqn{I(\sigma^2, \boldsymbol \theta)}
#' is derived based on the standard likelihood; otherwise, Fisher information matrix \eqn{I(\sigma^2, \boldsymbol \theta)}
#' is derived based on the integrated likelihood. 
#' @param formula an object of \code{formula} class that specifies regressors; see \code{\link[stats]{formula}} for details.
#' @param input a matrix including inputs in a GaSP
#' 
#' @param param a list including values for regression parameters, covariance parameters, 
#' and nugget variance parameter.
#' The specification of \strong{param} should depend on the covariance model. 
#' \itemize{
#' \item{The regression parameters are denoted by \strong{coeff}. Default value is \eqn{\mathbf{0}}.}
#' \item{The marginal variance or partial sill is denoted by \strong{sig2}. Default value is 1.}
#' \item{The nugget variance parameter is denoted by \strong{nugget} for all covariance models. 
#' Default value is 0.}
#' \item{For the Confluent Hypergeometric class, \strong{range} is used to denote the range parameter \eqn{\beta}. 
#' \strong{tail} is used to denote the tail decay parameter \eqn{\alpha}. \strong{nu} is used to denote the 
#' smoothness parameter \eqn{\nu}.}
#' \item{For the generalized Cauchy class, \strong{range} is used to denote the range parameter \eqn{\phi}. 
#' \strong{tail} is used to denote the tail decay parameter \eqn{\alpha}. \strong{nu} is used to denote the 
#' smoothness parameter \eqn{\nu}.}
#' \item{For the Matérn class, \strong{range} is used to denote the range parameter \eqn{\phi}. 
#' \strong{nu} is used to denote the smoothness parameter \eqn{\nu}. When \eqn{\nu=0.5}, the 
#' Matérn class corresponds to the exponential covariance.}  
#' \item{For the powered-exponential class, \strong{range} is used to denote the range parameter \eqn{\phi}.
#' \strong{nu} is used to denote the smoothness parameter. When \eqn{\nu=2}, the powered-exponential class
#' corresponds to the Gaussian covariance.}
#' }
#'
#' @param cov.model a list of two strings: \strong{family}, \strong{form}, where \strong{family} indicates the family of covariance functions 
#' including the Confluent Hypergeometric class, the Matérn class, the Cauchy class, the powered-exponential class. \strong{form} indicates the 
#' specific form of covariance structures including the isotropic form, tensor form, automatic relevance determination form. 
#' \describe{
#' \item{\strong{family}}{
#' \describe{
#' \item{CH}{The Confluent Hypergeometric correlation function is given by 
#' \deqn{C(h) = \frac{\Gamma(\nu+\alpha)}{\Gamma(\nu)} 
#' \mathcal{U}\left(\alpha, 1-\nu, \left(\frac{h}{\beta}\right)^2\right),}
#' where \eqn{\alpha} is the tail decay parameter. \eqn{\beta} is the range parameter.
#' \eqn{\nu} is the smoothness parameter. \eqn{\mathcal{U}(\cdot)} is the confluent hypergeometric
#' function of the second kind. For details about this covariance, 
#' see Ma and Bhadra (2019) at \url{https://arxiv.org/abs/1911.05865}.  
#' }
#' \item{cauchy}{The generalized Cauchy covariance is given by
#' \deqn{C(h) = \left\{ 1 + \left( \frac{h}{\phi} \right)^{\nu}  
#'             \right\}^{-\alpha/\nu},}
#' where \eqn{\phi} is the range parameter. \eqn{\alpha} is the tail decay parameter.
#' \eqn{\nu} is the smoothness parameter with default value at 2.
#'}
#'
#' \item{matern}{The Matérn correlation function is given by
#' \deqn{C(h)=\frac{2^{1-\nu}}{\Gamma(\nu)} \left( \frac{h}{\phi} \right)^{\nu} 
#' \mathcal{K}_{\nu}\left( \frac{h}{\phi} \right),}
#' where \eqn{\phi} is the range parameter. \eqn{\nu} is the smoothness parameter. 
#' \eqn{\mathcal{K}_{\nu}(\cdot)} is the modified Bessel function of the second kind of order \eqn{\nu}.
#' }
#' \item{exp}{The exponential correlation function is given by 
#' \deqn{C(h)=\exp(-h/\phi),}
#' where \eqn{\phi} is the range parameter. This is the Matérn correlation with \eqn{\nu=0.5}.
#' }
#' \item{matern_3_2}{The Matérn correlation with \eqn{\nu=1.5}.}
#' \item{matern_5_2}{The Matérn correlation with \eqn{\nu=2.5}.}
#'
#'
#' \item{powexp}{The powered-exponential correlation function is given by
#'                \deqn{C(h)=\exp\left\{-\left(\frac{h}{\phi}\right)^{\nu}\right\},}
#' where \eqn{\phi} is the range parameter. \eqn{\nu} is the smoothness parameter.
#' }
#' \item{gauss}{The Gaussian correlation function is given by 
#' \deqn{C(h)=\exp\left(-\frac{h^2}{\phi^2}\right),}
#' where \eqn{\phi} is the range parameter.
#'  }
#' }
#' }
#' 
#' \item{\strong{form}}{
#' \describe{
#'  \item{isotropic}{This indicates the isotropic form of covariance functions. That is,
#'  \deqn{C(\mathbf{h}) = C^0(\|\mathbf{h}\|; \boldsymbol \theta),} where \eqn{\| \mathbf{h}\|} denotes the 
#' Euclidean distance or the great circle distance for data on sphere. \eqn{C^0(\cdot)} denotes 
#' any isotropic covariance family specified in \strong{family}.}
#'  \item{tensor}{This indicates the tensor product of correlation functions. That is, 
#' \deqn{ C(\mathbf{h}) = \prod_{i=1}^d C^0(|h_i|; \boldsymbol \theta_i),}
#' where \eqn{d} is the dimension of input space. \eqn{h_i} is the distance along the \eqn{i}th input dimension. This type of covariance structure has been often used in Gaussian process emulation for computer experiments.
#'}
#'  \item{ARD}{This indicates the automatic relevance determination form. That is, 
#' \deqn{C(\mathbf{h}) = C^0\left(\sqrt{\sum_{i=1}^d\frac{h_i^2}{\phi^2_i}}; \boldsymbol \theta \right),}
#' where \eqn{\phi_i} denotes the range parameter along the \eqn{i}th input dimension.}
#'  }
#' }
#'
#'}
#' 
#'
#' @param dtype a string indicating the type of distance:
#' \describe{
#' \item{Euclidean}{Euclidean distance is used. This is the default choice.}
#' \item{GCD}{Great circle distance is used for data on sphere.}
#'}
#' 
#' @author Pulong Ma \email{mpulong@@gmail.com}
#' 
#' @seealso \link{GPBayes-package}, \code{\link{GaSP}}, \code{\link{gp}}, \code{\link{kernel}}, \code{\link{ikernel}},
#' 
#' @export 
#' @return a numerical matrix of Fisher information
#' @examples  
#' n=100
#' input = seq(0, 20, length=n)
#' range = 1
#' tail = .5
#' nu = 1.5
#' sig2 = 1
#' nugget = 0.01
#' coeff = 0
#' par = list(range=range, tail=tail, nu=nu, sig2=sig2, nugget=nugget, coeff=coeff)
#' I = gp.fisher(formula=~1, input=input, 
#'         param=list(range=4, nugget=0.1,nu=2.5),
#'         cov.model=list(family="CH", form="isotropic"))

gp.fisher <- function(obj=NULL, intloglik=FALSE, formula=~1, input=NULL, param=NULL, cov.model=NULL, dtype="Euclidean"){

  if(!is.null(obj)){
    formula = obj@formula 
    input = obj@input 
    param = obj@param 
    cov.model = obj@cov.model
    dtype = obj@dtype 
  }else{
    if(is.null(input)){
      stop("Please specify input spatial locations\n")
    }

    if(is.null(param)){
      stop("Please specify covariance parameters\n")      
    }
    if(is.null(cov.model)){
      stop("Please specify covariance model\n")      
    }

  }



  if(!is.matrix(input)){
    input = as.matrix(input)
  }

  Dim = dim(input)[2]
  colnames(input) = paste0("x", 1:Dim)
  df = data.frame(input)
  H = model.matrix(formula, df)
  d = distance(input, input, type=cov.model$form, dtype);
  p = ncol(H)

  if(!is(param, "list") || is.null(param)){
    stop("param should be a list containing initial values for  
         correlation parameters and nugget variance parameter.\n")
  }else{
    if(!exists("coeff", where=param)){
      param$coeff = rep(0, p)
    }

    if(!exists("sig2", where=param)){
      param$sig2 = 1 
    }
    
    if(!exists("nugget", where=param)){
      param$nugget = 0 
    }

    if(!exists("range", where=param)){
      if(cov.model$form=="isotropic"){
        param$range = max(d)/2
      }else{
        param$range = sapply(d, max)/2
      }
      
    }

    if(!exists("tail", where=param)){
      if(cov.model$form=="isotropic" || cov.model$form=="ARD"){
        param$tail = 0.5
      }else{
        param$tail = rep(0.5, Dim)
      }
    }

    if(!exists("nu", where=param)){
      if(cov.model$form=="isotropic" || cov.model$form=="ARD"){
        param$nu = 0.5
      }else{
        param$tail = rep(0.5, Dim)
      }
    }
    
  }

  if(!is(dtype, "character")){
    stop("dtype: distance type is not correctly specified.\n")
  }

  sig2 = param$sig2 
  range = param$range 
  tail = param$tail 
  nu = param$nu 
  nugget = param$nugget 



  if(intloglik){
    Imat = FisherIR_intlik(H, input, range, tail, nu, nugget, cov.model, dtype)
  }else{
    Imat = FisherInfo(input, sig2, range, tail, nu, nugget, cov.model, dtype)
  }

  if(cov.model[["form"]]=="isotropic"){
    if(cov.model[["family"]]=="CH"){
      colnames(Imat) = c("sig2", "range", "tail", "nugget", "nu")
      rownames(Imat) = c("sig2", "range", "tail", "nugget", "nu")

    }else if(cov.model[["family"]]=="matern"){
      colnames(Imat) = c("sig2", "range", "nugget")
      rownames(Imat) = c("sig2", "range", "nugget")
    }else{
      stop("The Fisher information matrix for the specified covariance family is not implemented.\n")
    }
    
  }else if(cov.model[["form"]]=="tensor"){
    if(cov.model[["family"]]=="CH"){

      colnames(Imat) = c("sig2", paste0("range", seq(1:Dim)), paste0("tail", seq(1:Dim)), "nugget", "nu")
      rownames(Imat) = c("sig2", paste0("range", seq(1:Dim)), paste0("tail", seq(1:Dim)), "nugget", "nu")

    }else if(cov.model[["family"]]=="matern"){
      colnames(Imat) = c("sig2", paste0("range", seq(1:Dim)), "nugget")
      rownames(Imat) = c("sig2", paste0("range", seq(1:Dim)), "nugget")
    }else{
      stop("The Fisher information matrix for the specified covariance family is not implemented.\n")
    }
  }else if(cov.model[["form"]]=="ARD"){
    if(cov.model[["family"]]=="CH"){
      colnames(Imat) = c("sig2", paste0("range", seq(1:Dim)), "tail", "nugget", "nu")
      rownames(Imat) = c("sig2", paste0("range", seq(1:Dim)), "tail", "nugget", "nu")

    }else if(cov.model[["family"]]=="matern"){
      colnames(Imat) = c("sig2", paste0("range", seq(1:Dim)), "nugget")
      rownames(Imat) = c("sig2", paste0("range", seq(1:Dim)), "nugget")
    }else{
      stop("The Fisher information matrix for the specified covariance family is not implemented.\n")
    }
  }else{
    stop("The Fisher information matrix for the specified covariance form is not implemented.\n")
  }

  return(Imat)

}


##########################################################################################
#' @title Model assessment based on Deviance information criterion (DIC), logarithmic pointwise predictive density (lppd), and
#' logarithmic joint predictive density (ljpd).
#' @description This function computes effective number of parameters (pD), deviance information criterion (DIC), logarithmic pointwise predictive density (lppd), and
#' logarithmic joint predictive density (ljpd). For detailed introduction of these 
#' metrics, see Chapter 7 of Gelman et al. (2013).
#' 
#' The deviance function for a model with a vector of parameters 
#' \eqn{\boldsymbol \theta} is defined as
#'  \deqn{
#'  D(\boldsymbol \theta) = -2\log p(\mathbf{y} \mid \boldsymbol \theta),
#'} 
#' where \eqn{\mathbf{y}:=(y(\mathbf{x}_1), \ldots, y(\mathbf{x}_n))^\top} is a vector of \eqn{n} observations.
#' 
#'\itemize{
#' \item{The effective number of parameters (see p.172 of Gelman et al. 2013) is defined as
#' \deqn{
#'  pD = E_{\boldsymbol \theta| \mathbf{y}}[D(\boldsymbol \theta)] - D(\boldsymbol \hat{\theta}),
#'  }
#' where \eqn{\hat{\boldsymbol \theta} = E_{\boldsymbol \theta | \mathbf{y}}[\boldsymbol \theta]. }
#' The interpretation is that the effective number of parameters is the ``expected" 
#' deviance minus the ``fitted" deviance. Higher \eqn{pD} implies more over-fitting with estimate  \eqn{\hat{\boldsymbol \theta}}.
#' }
#' \item{The Deviance information criteria (DIC) (see pp. 172-173 of Gelman et al. 2013) is 
#' \deqn{DIC = E_{\boldsymbol \theta | \mathbf{y}}[D(\boldsymbol \theta)] + pD.
#' }
#' DIC approximates Akaike information criterion (AIC) and is more appropriate for hierarchical models than AIC and BIC.
#'}
#' \item{The log predictive density (\strong{lpd}) is defined as 
#' \deqn{ p(y(\mathbf{x}_0) \mid \mathbf{y}) = \int p(y(\mathbf{x}_0) \mid 
#' \boldsymbol \theta, \mathbf{y}) p(\boldsymbol \theta \mid \mathbf{y}) 
#' d \boldsymbol \theta,
#' }
#' where \eqn{\mathbf{y}:=(y(\mathbf{x}_1), \ldots, y(\mathbf{x}_n))^\top} is a vector of \eqn{n} observations.
#' \eqn{\boldsymbol \theta} contains correlation
#' parameters and nugget parameter. This predictive density should be understood as an update of the likelihood since data is treated as prior information now.
#' With a set of prediction locations \eqn{\mathcal{X}:=\{x_0^i: i=1, \ldots, m\}}. 
#' The log pointwise predictive density (\strong{lppd}) is defined as
#' \deqn{lppd = \sum_{i=1}^m \log p(y(\mathbf{x}_0^i) 
#' \mid \mathbf{y}).} 
#' The log joint predictive density (\strong{ljpd}) is defined as 
#' \deqn{ljpd = \log p(y(\mathcal{X})). }
#' The \code{lppd} is connected to cross-validation, while the \code{ljpd} measures joint uncertainty across prediction locations.
#' }
#'
#'}
#' @param obj a \code{\link{gp}} object. 
#' 
#' @param testing.input a matrix of testing inputs
#' @param testing.output a vector of testing outputs
#' @param pointwise a logical value with default value \code{TRUE}. If it is
#' \code{TRUE}, \strong{lppd} is calculated.
#' @param joint a logical value with default value \code{TRUE}. If it is
#' \code{TRUE}, \strong{ljpd} is calculated.
#' 
#' @author Pulong Ma \email{mpulong@@gmail.com}
#'  
#' @seealso \link{GPBayes-package}, \code{\link{GaSP}}, \code{\link{gp}}, 
#' @return a list containingg \strong{pD}, \strong{DIC}, \strong{lppd}, \strong{ljpd}.
#' @export 
#' @references 
#' \itemize{
#' \item{Gelman, Andrew, John B. Carlin, Hal S. Stern, David B. Dunson, 
#' Aki Vehtari, and Donald B. Rubin (2013). 
#' Bayesian Data Analysis, Third Edition. CRC Press.}
#' }
#'  
gp.model.adequacy = function(obj, testing.input, testing.output,
                             pointwise=TRUE, joint=TRUE){
 
  formula = obj@formula
  output = obj@output
  input = obj@input
  param = obj@param
  cov.model = obj@cov.model
  family = cov.model$family 
  form = cov.model$form 
  dtype = obj@dtype
  mcmc = obj@mcmc 
  smooth.est = obj@smooth.est 

  smoothness = param$nu 
  
  if(length(mcmc)==0){
    stop("MCMC samples are not provided in obj@mcmc.\n")
  }
  
  if(!is.matrix(testing.input)){
    testing.input = as.matrix(testing.input)
  }
  if(!is.matrix(testing.output)){
    testing.output = as.matrix(testing.output)
  }
  
  Dim = dim(input)[2]
  colnames(input) = paste0("x", 1:Dim)
  df = data.frame(input)
  H = model.matrix(formula, df)
  
  colnames(testing.input) = paste0("x", 1:Dim)
  df = data.frame(testing.input)
  Hnew = model.matrix(formula, df)
  
  n = nrow(output)
  p = ncol(H)
  
  Result = model_evaluation(output, H, input, cov.model, smoothness,
                            mcmc, testing.output, testing.input,
                            Hnew, dtype, pointwise, joint)
  return(Result)
}

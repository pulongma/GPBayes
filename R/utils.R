
###########################################################################################
##########################################################################################



#' @title Find the correlation parameter given effective range
#' @description This function finds the correlation parameter given effective range
#' 
#' @param d a numerical value containing the effective range
#' @param param a list containing correlation parameters. The specification of 
#' \strong{param} should depend on the covariance model. If the parameter value is
#'  \code{NULL}, this function will find its value given the effective range via
#'  root-finding function \code{\link[stats]{uniroot}}.
#' \itemize{
#' \item{For the Confluent Hypergeometric class, \strong{range} is used to denote the range parameter \eqn{\beta}. 
#' \strong{tail} is used to denote the tail decay parameter \eqn{\alpha}. \strong{nu} is used to denote the 
#' smoothness parameter \eqn{\nu}.} 
#' 
#' \item{For the generalized Cauchy class, \strong{range} is used to denote the range parameter \eqn{\phi}. 
#' \strong{tail} is used to denote the tail decay parameter \eqn{\alpha}. \strong{nu} is used to denote the 
#' smoothness parameter \eqn{\nu}.}
#' 
#' \item{For the Matérn class, \strong{range} is used to denote the range parameter \eqn{\phi}. 
#' \strong{nu} is used to denote the smoothness parameter \eqn{\nu}. When \eqn{\nu=0.5}, the 
#' Matérn class corresponds to the exponential covariance.}  
#' 
#' \item{For the powered-exponential class, \strong{range} is used to denote the range parameter \eqn{\phi}.
#' \strong{nu} is used to denote the smoothness parameter. When \eqn{\nu=2}, the powered-exponential class
#' corresponds to the Gaussian covariance.}
#' }
#' 
#' @param family a string indicating the type of covariance structure.
#' The following correlation functions are implemented:
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
#' \eqn{\nu} is the smoothness parameter.
#'}
#'
#' \item{matern}{The Matérn correlation function is given by
#' \deqn{C(h)=\frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{h}{\phi} \right)^{\nu} 
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
#' }
#' 
#' @param cor.target a numerical value. The default value is 0.05, which 
#' means that correlation parameters are searched such that the correlation
#' is approximately 0.05.
#' 
#' @param lower a numerical value. This sets the lower bound to find the 
#' correlation parameter via the \code{R} function \code{\link[stats]{uniroot}}.
#' @param upper a numerical value. This sets the upper bound to find the 
#' correlation parameter via the \code{R} function \code{\link[stats]{uniroot}}.
#' 
#' @param tol a numerical value. This sets the precision of the solution with default value
#' specified as the machine precision \code{.Machine$double.eps} in \code{R}. 
#' @author Pulong Ma \email{mpulong@@gmail.com}
#' 
#' @seealso \link{GPBayes-package}, \code{\link{GaSP}}, \code{\link{kernel}}, \code{\link{ikernel}}
#' @export
#' @return a numerical value of correlation parameters
#' @examples 
#' 
#' range = cor.to.par(1,param=list(tail=0.5,nu=2.5), family="CH")
#' tail = cor.to.par(1,param=list(range=0.5,nu=2.5), family="CH")
#' range = cor.to.par(1,param=list(nu=2.5),family="matern")
#' 
cor.to.par = function(d, param, family="CH", cor.target=0.05, lower=NULL, 
                        upper=NULL, tol=.Machine$double.eps){
  if(!is(d, "matrix")){
    d = as.matrix(d)
  }
  if(is.null(lower)){
    lower=1e-20
  }
  
  if(is.null(upper)){
    upper = 1e20
  }
  
  if(!is(param, "list")){
    stop("param should be a list containing correlation parameters.\n")
  }else{
    if(family=="CH"){
      if(is.null(param$range)){
        out = uniroot(function(x) drop(CH(d, x, param$tail, param$nu)) - cor.target,
                      c(lower, upper), tol=tol)
      }
      if(is.null(param$tail)){
        out = uniroot(function(x) drop(CH(d, param$range, x, param$nu)) - cor.target,
                      c(1e-3, 8), tol=tol) 
      }
    }else if(family=="matern" || family=="exp" || family=="matern_3_2" || family=="matern_5_2"){
      if(is.null(param$range)){
        out = uniroot(function(x) drop(matern(d, x, param$nu)) - cor.target,
                      c(lower, upper), tol=tol)
      }
      
    }else if(family=="cauchy"){
      if(is.null(param$range)){
        out = uniroot(function(x) drop(cauchy(d, x, param$tail, param$nu)) - cor.target,
                      c(lower, upper), tol=tol)
      }
      if(is.null(param$tail)){
        out = uniroot(function(x) drop(cauchy(d, param$range, x, param$nu)) - cor.target,
                      c(lower, upper), tol=tol)        
      }
    }else if(family=="gauss" || family=="powexp"){
      out = uniroot(function(x) drop(powexp(d, x, param$nu)) - cor.target,
                    c(lower, upper), tol=tol)
    }else{
      stop("cor.to.par: unsupported covariance model.\n")
    }
  }
  
  return(out$root)
  
}

###########################################################################################
##########################################################################################


###################################################################################
############## Maximization with the marginal/integrated likelihood
################################################################################### 
MMLE = function(obj, opt=NULL, bound=NULL){

  formula = obj@formula
  output = obj@output 
  input = obj@input 
  param = obj@param
  cov.model = obj@cov.model 
  family = cov.model$family
  form = cov.model$form 
  dtype = obj@dtype
  smooth.est = obj@smooth.est 

  range = param$range
  tail = param$tail
  nu = param$nu 
  nugget = param$nugget 

  
  Dim = dim(input)[2]
  colnames(input) = paste0("x", 1:Dim)
  df = data.frame(input)
  H = model.matrix(formula, df)

  n = nrow(output)
  p = ncol(H)
  

  par = list(range=log(param$range), tail=log(param$tail), 
        nugget=param$nugget, nu=log(nu))  
  

  if(is.null(opt)){
    if(!smooth.est){
      if(family=="matern" || family=="exp" || family=="matern_3_2" || family=="matern_5_2" || family=="gauss" || family=="powexp"){
        opt = list(method="L-BFGS-B")
      }else{
        opt = list(method="Nelder-Mead", maxit=800)
      }
      
    }else{
      opt = list(method="Nelder-Mead", maxit=800)
    }
  }
  
  
  d = distance(input1=input, input2=input, type=form, dtype=dtype)
  


  if(opt$method=="Nelder-Mead"){

    if(smooth.est){
      if(family=="CH" || family=="cauchy"){
        init.val = c(par$range, par$tail, par$nugget, par$nu)
      }else if(family=="matern" || family=="exp" || family=="matern_3_2" || family=="matern_5_2" || family=="gauss" || family=="powexp"){
        init.val = c(par$range, par$nugget, par$nu)
      }
      if(!exists("lower", where=opt)){
        opt$lower = rep(-Inf, length(init.val))
      }
      if(!exists("upper", where=opt)){
        opt$upper = rep(Inf, length(init.val))        
      }  
      if(!exists("maxit", where=opt)){
        opt$maxit = 800        
      }     

      fit = try(optim(init.val, loglik_xi, output=output, H=H, d=d, covmodel=cov.model,
                      smooth=nu, smoothness_est=smooth.est, 
                      method=opt$method, lower=opt$lower, upper=opt$upper,
                      control = list(fnscale = -1, maxit=opt$maxit)),
                silent=T)      
    }else{
      if(family=="CH" || family=="cauchy"){
        init.val = c(par$range, par$tail, par$nugget)
      }else if(family=="matern" || family=="exp" || family=="matern_3_2" || family=="matern_5_2" || family=="gauss" || family=="powexp"){
        init.val = c(par$range, par$nugget)
      }      

      if(!exists("lower", where=opt)){
        opt$lower = rep(-Inf, length(init.val))
      }
      if(!exists("upper", where=opt)){
        opt$upper = rep(Inf, length(init.val))        
      } 
      if(!exists("maxit", where=opt)){
        opt$maxit = 800        
      }    

      fit = try(optim(init.val, loglik_xi, output=output, H=H, d=d, covmodel=cov.model,
                      smooth=nu, smoothness_est=smooth.est, 
                      method=opt$method, lower=opt$lower, upper=opt$upper,
                      control = list(fnscale = -1, maxit=opt$maxit)),
                silent=T)
    } # end smooth.est

  }else if(opt$method=="L-BFGS-B"){
    #message("MMLE: the L-BFGS-B is partially supported.\n")

    if(smooth.est){
      if(family=="CH" || family=="cauchy"){
        init.val = c(par$range, par$tail, par$nugget, par$nu)
      }else if(family=="matern" || family=="gauss" || family=="powexp"){
        init.val = c(par$range, par$nugget, par$nu)
      }

      if(!exists("lower",where=opt)){
        opt$lower = rep(1e-10, length(init.val))
      }
      if(!exists("upper", where=opt)){
        opt$upper = rep(1e10, length(init.val))
      }
      init.val = exp(init.val)
      fit = try(optim(init.val, fn=loglik, gr=gradient_loglik, output=output, H=H, d=d, covmodel=cov.model,
                      smooth=nu, smoothness_est=smooth.est, 
                      method="L-BFGS-B", lower=opt$lower, upper=opt$upper,
                      control = list(fnscale = -1)),
                silent=T)      
    }else{
      if(family=="CH" || family=="cauchy"){
        init.val = c(par$range, par$tail, par$nugget)
      }else if(family=="matern" || family=="gauss" || family=="powexp"){
        init.val = c(par$range, par$nugget)
      }      

      if(!exists("lower",where=opt)){
        opt$lower = rep(1e-10, length(init.val))
      }
      if(!exists("upper", where=opt)){
        opt$upper = rep(1e10, length(init.val))
      }
      init.val = exp(init.val)
      fit = try(optim(init.val, fn=loglik, gr=gradient_loglik, output=output, H=H, d=d, covmodel=cov.model,
                      smooth=nu, smoothness_est=smooth.est, 
                      method="L-BFGS-B", lower=opt$lower, upper=opt$upper,
                      control = list(fnscale = -1)),
                silent=T) 
      init.val = log(init.val) # convert to log-scale

    } # end smooth.est
  }else{
    stop("MMLE: This optimization method is not supported. Please use either Nelder-Mead or L-BFGS-B.\n")
  }

  
  if(inherits(fit, "try-error")){
    par = init.val
    fit$success = FALSE
    # itran_par(par, cov.model, bound, nugget.est)
    message("\noptimization error!\n")
    fit$par = par 
  }else{
    if(opt$method=="Nelder-Mead"){
      par = fit$par
    }else if(opt$method=="L-BFGS-B"){
      par = log(fit$par)
    }
    
    fit$success = TRUE
    # itran_par(par, cov.model, bound, nugget.est)
  }

  
  if(form=="isotropic"){
    if(family=="CH"||family=="cauchy"){
      range = exp(par[1])
      tail = exp(par[2])
      nugget = exp(par[3])
      par.est = list(range=range, tail=tail, nugget=nugget, nu=nu)
      if(smooth.est){
        nu = exp(par[4])
        par.est$nu = nu 
      }

    }else if(family=="matern" || family=="gauss" || family=="powexp"){
      range = exp(par[1])
      nugget = exp(par[2])
      par.est = list(range=range, nugget=nugget, nu=nu)
      if(smooth.est){
        nu = exp(par[3])
        par.est$nu = nu 
      }
    }else{
      stop("MMLE: unsupported covariance family.\n")
    }
  }else if(form=="tensor"){
    if(family=="CH"||family=="cauchy"){
      range = exp(par[1:Dim])
      tail = exp(par[(Dim+1):(2*Dim)])
      nugget = exp(par[2*Dim+1])
      par.est = list(range=range, tail=tail, nugget=nugget, nu=nu)
      if(smooth.est){
        nu = exp(par[2*Dim+2])
        par.est$nu = nu 
      }
    }else if(family=="matern" || family=="gauss" || family=="powexp"){
      range = exp(par[1:Dim])
      nugget = exp(par[Dim+1])
      par.est = list(range=range, nugget=nugget, nu=nu)
      if(smooth.est){
        nu = exp(par[Dim+2])
        par.est$nu = nu 
      }
    }else{
      stop("MMLE: unsupported covariance family.\n")
    }
  }else if(form=="ARD"){
    if(family=="CH"||family=="cauchy"){
      range = exp(par[1:Dim])
      tail = exp(par[Dim+1])
      nugget = exp(par[Dim+2])
      par.est = list(range=range, tail=tail, nugget=nugget, nu=nu)
      if(smooth.est){
        nu = exp(par[Dim+3])
        par.est$nu = nu 
      }
    }else if(family=="matern" || family=="exp" || family=="matern_3_2" || family=="matern_5_2" || family=="gauss" || family=="powexp"){
      range = exp(par[1:Dim])
      nugget = exp(par[Dim+1])
      par.est = list(range=range, nugget=nugget, nu=nu)
      if(smooth.est){
        nu = exp(par[Dim+2])
        par.est$nu = nu 
      }
    }else{
      stop("MMLE: unsupported covariance family.\n")
    }    
  }else{
    stop("MMLE: unsupported covariance form.\n")
  }


  cormat = kernel(d, range, tail, nu, cov.model)
  cormat = cormat + nugget*diag(nrow(cormat))

  Q = chol(cormat)
  QInv = chol2inv(Q)
  sig2hat = t(output)%*%QInv%*%output/(n-p) 
  par.est$sig2 = drop(sig2hat)


  # compute bhat = (H^tR^(-1)H)^{-1}H^tR^(-1)y
  par.est$coeff = drop(solve(t(H)%*%QInv%*%H, t(H)%*%(QInv%*%output)))
  fit$par = par.est

  obj@param$range = par.est$range
  if(exists("tail", where=par.est)){
    obj@param$tail = par.est$tail
  }else{
    obj@param$tail = tail 
  }
  obj@param$nugget = par.est$nugget 
  obj@param$nu = par.est$nu 
  obj@param$sig2 = par.est$sig2  
  obj@param$coeff = par.est$coeff 

  return(list(obj=obj,fit=fit))
  
}

###########################################################################################
##########################################################################################


###################################################################################
############## Maximization with profile likelihood
###################################################################################
MPLE = function(obj, opt=NULL, bound=NULL){

  formula = obj@formula
  output = obj@output 
  input = obj@input 
  param = obj@param
  cov.model = obj@cov.model 
  family = cov.model$family
  form = cov.model$form 
  dtype = obj@dtype
  smooth.est = obj@smooth.est 

  range = param$range
  tail = param$tail
  nu = param$nu 
  nugget = param$nugget 

  
  Dim = dim(input)[2]
  colnames(input) = paste0("x", 1:Dim)
  df = data.frame(input)
  H = model.matrix(formula, df)

  n = nrow(output)
  p = ncol(H)
  

  par = list(range=log(param$range), tail=log(param$tail), 
        nugget=param$nugget, nu=log(nu))

      
  if(is.null(opt)){
    if(!smooth.est){
      if(family=="matern" || family=="gauss" || family=="powexp"){
        opt = list(method="L-BFGS-B")
      }else{
        opt = list(method="Nelder-Mead", maxit=800)
      }
      
    }else{
      opt = list(method="Nelder-Mead", maxit=800)
    }
  }
  
  
  d = distance(input1=input, input2=input, type=form, dtype=dtype)
  


 if(opt$method=="Nelder-Mead"){

    if(smooth.est){
      if(family=="CH" || family=="cauchy"){
        init.val = c(par$range, par$tail, par$nugget, par$nu)
      }else if(family=="matern" || family=="gauss" || family=="powexp"){
        init.val = c(par$range, par$nugget, par$nu)
      }
      if(!exists("lower", where=opt)){
        opt$lower = rep(-Inf, length(init.val))
      }
      if(!exists("upper", where=opt)){
        opt$upper = rep(Inf, length(init.val))        
      }  
      if(!exists("maxit", where=opt)){
        opt$maxit = 800        
      }     

      fit = try(optim(init.val, loglik_xi, output=output, H=H, d=d, covmodel=cov.model,
                      smooth=nu, smoothness_est=smooth.est, 
                      method=opt$method, lower=opt$lower, upper=opt$upper,
                      control = list(fnscale = -1, maxit=opt$maxit)),
                silent=T)      
    }else{
      if(family=="CH" || family=="cauchy"){
        init.val = c(par$range, par$tail, par$nugget)
      }else if(family=="matern" || family=="gauss" || family=="powexp"){
        init.val = c(par$range, par$nugget)
      }      

      if(!exists("lower", where=opt)){
        opt$lower = rep(-Inf, length(init.val))
      }
      if(!exists("upper", where=opt)){
        opt$upper = rep(Inf, length(init.val))        
      } 
      if(!exists("maxit", where=opt)){
        opt$maxit = 800        
      }    

      fit = try(optim(init.val, loglik_xi, output=output, H=H, d=d, covmodel=cov.model,
                      smooth=nu, smoothness_est=smooth.est, 
                      method=opt$method, lower=opt$lower, upper=opt$upper,
                      control = list(fnscale = -1, maxit=opt$maxit)),
                silent=T)
    } # end smooth.est

  }else if(opt$method=="L-BFGS-B"){
    #message("MMLE: the L-BFGS-B is partially supported.\n")

    if(smooth.est){
      if(family=="CH" || family=="cauchy"){
        init.val = c(par$range, par$tail, par$nugget, par$nu)
      }else if(family=="matern" || family=="gauss" || family=="powexp"){
        init.val = c(par$range, par$nugget, par$nu)
      }

      if(!exists("lower",where=opt)){
        opt$lower = rep(1e-10, length(init.val))
      }
      if(!exists("upper", where=opt)){
        opt$upper = rep(1e10, length(init.val))
      }
      init.val = exp(init.val)
      fit = try(optim(init.val, fn=loglik, gr=gradient_loglik, output=output, H=H, d=d, covmodel=cov.model,
                      smooth=nu, smoothness_est=smooth.est, 
                      method="L-BFGS-B", lower=opt$lower, upper=opt$upper,
                      control = list(fnscale = -1)),
                silent=T)      
    }else{
      if(family=="CH" || family=="cauchy"){
        init.val = c(par$range, par$tail, par$nugget)
      }else if(family=="matern" || family=="gauss" || family=="powexp"){
        init.val = c(par$range, par$nugget)
      }      

      if(!exists("lower",where=opt)){
        opt$lower = rep(1e-10, length(init.val))
      }
      if(!exists("upper", where=opt)){
        opt$upper = rep(1e10, length(init.val))
      }
      init.val = exp(init.val)
      fit = try(optim(init.val, fn=loglik, gr=gradient_loglik, output=output, H=H, d=d, covmodel=cov.model,
                      smooth=nu, smoothness_est=smooth.est, 
                      method="L-BFGS-B", lower=opt$lower, upper=opt$upper,
                      control = list(fnscale = -1)),
                silent=T) 
      init.val = log(init.val) # convert to log-scale

    } # end smooth.est
  }else{
    stop("MMLE: This optimization method is not supported. Please use either Nelder-Mead or L-BFGS-B.\n")
  }

  
  if(inherits(fit, "try-error")){
    par = init.val
    fit$success = FALSE
    # itran_par(par, cov.model, bound, nugget.est)
    message("\noptimization error!\n")
    fit$par = par 
  }else{
    if(opt$method=="Nelder-Mead"){
      par = fit$par
    }else if(opt$method=="L-BFGS-B"){
      par = log(fit$par)
    }
    
    fit$success = TRUE
    # itran_par(par, cov.model, bound, nugget.est)
  }

  
  
  if(form=="isotropic"){
    if(family=="CH"||family=="cauchy"){
      range = exp(par[1])
      tail = exp(par[2])
      nugget = exp(par[3])
      par.est = list(range=range, tail=tail, nugget=nugget, nu=nu)
      if(smooth.est){
        nu = exp(par[4])
        par.est$nu = nu 
      }

    }else if(family=="matern" || family=="gauss" || family=="powexp"){
      range = exp(par[1])
      nugget = exp(par[2])
      par.est = list(range=range, nugget=nugget, nu=nu)
      if(smooth.est){
        nu = exp(par[3])
        par.est$nu = nu 
      }
    }else{
      stop("MMLE: unsupported covariance family.\n")
    }
  }else if(form=="tensor"){
    if(family=="CH"||family=="cauchy"){
      range = exp(par[1:Dim])
      tail = exp(par[(Dim+1):(2*Dim)])
      nugget = exp(par[2*Dim+1])
      par.est = list(range=range, tail=tail, nugget=nugget, nu=nu)
      if(smooth.est){
        nu = exp(par[2*Dim+2])
        par.est$nu = nu 
      }
    }else if(family=="matern" || family=="gauss" || family=="powexp"){
      range = exp(par[1:Dim])
      nugget = exp(par[Dim+1])
      par.est = list(range=range, nugget=nugget, nu=nu)
      if(smooth.est){
        nu = exp(par[Dim+2])
        par.est$nu = nu 
      }
    }else{
      stop("MMLE: unsupported covariance family.\n")
    }
  }else if(form=="ARD"){
    if(family=="CH"||family=="cauchy"){
      range = exp(par[1:Dim])
      tail = exp(par[Dim+1])
      nugget = exp(par[Dim+2])
      par.est = list(range=range, tail=tail, nugget=nugget, nu=nu)
      if(smooth.est){
        nu = exp(par[Dim+3])
        par.est$nu = nu 
      }
    }else if(family=="matern" || family=="gauss" || family=="powexp"){
      range = exp(par[1:Dim])
      nugget = exp(par[Dim+1])
      par.est = list(range=range, nugget=nugget, nu=nu)
      if(smooth.est){
        nu = exp(par[Dim+2])
        par.est$nu = nu 
      }
    }else{
      stop("MMLE: unsupported covariance family.\n")
    }    
  }else{
    stop("MMLE: unsupported covariance form.\n")
  }


  cormat = kernel(d, range, tail, nu, cov.model)
  cormat = cormat + nugget*diag(nrow(cormat))

  Q = chol(cormat)
  QInv = chol2inv(Q)
  sig2hat = t(output)%*%QInv%*%output/n 
  par.est$sig2 = drop(sig2hat)


  # compute bhat = (H^tR^(-1)H)^{-1}H^tR^(-1)y
  par.est$coeff = drop(solve(t(H)%*%QInv%*%H, t(H)%*%(QInv%*%output)))
  fit$par = par.est

  obj@param$range = par.est$range
  if(exists("tail", where=par.est)){
    obj@param$tail = par.est$tail
  }else{
    obj@param$tail = tail 
  } 
  obj@param$nugget = par.est$nugget 
  obj@param$nu = par.est$nu 
  obj@param$sig2 = par.est$sig2  
  obj@param$coeff = par.est$coeff 

  return(list(obj=obj,fit=fit))
}

###########################################################################################
##########################################################################################


logit = function(x, lb, ub){
  return(log(x-lb) - log(ub-x));
}

ilogit = function(x, lb, ub){
  return( lb + (exp(x) / (1.0+exp(x))) * (ub-lb) );
}


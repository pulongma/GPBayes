

#ifndef _USE_RcppEigen
#define _USE_RcppEigen
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp11)]]
#endif

#ifndef _USE_GSL
#define _USE_GSL
#include <gsl/gsl_sf.h>
#include <gsl/gsl_errno.h>
#endif

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "utils.h"
#include "UQtool.h"
#include "SPtool.h"


#ifndef _USE_ProgressBar
#define _USE_ProgressBar
#include <R_ext/Utils.h>   // interrupt the Gibbs sampler from R
#include <iostream>
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>
#include <progress_bar.hpp>
#endif


using namespace Rcpp;




// RCPP_MODULE(UQ){
// 	class_<UQ>("UQ")
// 		.constructor()
// 		.method("pdist", &UQ::pdist)
// 		.method("tdist", &UQ::tdist)
// 		.method("adist", &UQ::adist)
// 		.method("kernel", &UQ::kernel)
// 		.method("dervi_kernel", &UQ::dervi_kernel)
// 		.method("MLoglik", &UQ::MLoglik)
//         .method("gradient_MLoglik", &UQ::gradient_MLoglik)
// 	;
// }


// RCPP_MODULE(SP){
// 	class_<SP>("SP")
// 		.constructor()
// 		.method("pdist", &SP::pdist)
// 		.method("tdist", &SP::tdist)
// 		.method("kernel", &SP::kernel)
// 		.method("dervi_kernel", &SP::dervi_kernel)
// 		.method("MLoglik", &SP::MLoglik)
//     .method("gradient_MLoglik", &SP::gradient_MLoglik)
// 	;
// }



//' @title A wraper to compute the natural logarithm of the integrated likelihood function
//' 
//' @description This function wraps existing built-in routines to construct  
//' the natural logarithm of the integrated likelihood function. The constructed 
//' loglikelihood can be directly used for numerical optimization 
//'
//' @param par a numerical vector, with which numerical optimization routine such as \code{\link[stats]{optim}} can be
//' carried out directly. When the confluent Hypergeometric class is used, it is used to hold values 
//' for \strong{range}, \strong{tail}, \strong{nugget}, and \strong{nu} if the smoothness parameter is estimated.
//' When the Matérn class or powered-exponential class is used, it is used to hold values
//' for \strong{range}, \strong{nugget}, and \strong{nu} if the smoothness parameter is estimated.
//' The order of the parameter values in \code{par} cannot be changed. For tensor or ARD form correlation
//' functions, \strong{range} and \strong{tail} becomes a vector. 
//' @param output a matrix of outputs 
//' @param H a matrix of regressors in the mean function of a GaSP model.
//' @param d an R object holding the distances. It should be a distance matrix for constructing isotropic correlation matrix,
//' or a list of distance matrices along each input dimension for constructing tensor or ARD types of correlation matrix.
//' @param covmodel a list of two strings: \strong{family}, \strong{form}, where \strong{family} indicates the family of covariance functions 
//' including the Confluent Hypergeometric class, the Matérn class, the Cauchy class, the powered-exponential class. \strong{form} indicates the 
//' specific form of covariance structures including the isotropic form, tensor form, automatic relevance determination form. 
//' \describe{
//' \item{\strong{family}}{
//' \describe{
//' \item{CH}{The Confluent Hypergeometric correlation function is given by 
//' \deqn{C(h) = \frac{\Gamma(\nu+\alpha)}{\Gamma(\nu)} 
//' \mathcal{U}\left(\alpha, 1-\nu, \left(\frac{h}{\beta}\right)^2\right),}
//' where \eqn{\alpha} is the tail decay parameter. \eqn{\beta} is the range parameter.
//' \eqn{\nu} is the smoothness parameter. \eqn{\mathcal{U}(\cdot)} is the confluent hypergeometric
//' function of the second kind. For details about this covariance, 
//' see Ma and Bhadra (2019) at \url{https://arxiv.org/abs/1911.05865}.  
//' }
//' \item{cauchy}{The generalized Cauchy covariance is given by
//' \deqn{C(h) = \left\{ 1 + \left( \frac{h}{\phi} \right)^{\nu}  
//'             \right\}^{-\alpha/\nu},}
//' where \eqn{\phi} is the range parameter. \eqn{\alpha} is the tail decay parameter.
//' \eqn{\nu} is the smoothness parameter with default value at 2.
//'}
//'
//' \item{matern}{The Matérn correlation function is given by
//' \deqn{C(h)=\frac{2^{1-\nu}}{\Gamma(\nu)} \left( \frac{h}{\phi} \right)^{\nu} 
//' \mathcal{K}_{\nu}\left( \frac{h}{\phi} \right),}
//' where \eqn{\phi} is the range parameter. \eqn{\nu} is the smoothness parameter. 
//' \eqn{\mathcal{K}_{\nu}(\cdot)} is the modified Bessel function of the second kind of order \eqn{\nu}.
//' }
//' \describe{
//' \item{exp}{This is the Matérn correlation with \eqn{\nu=0.5}. This covariance should be specified as \strong{matern} with smoothness parameter \eqn{\nu=0.5}.
//' }
//' \item{matern_3_2}{This is the Matérn correlation with \eqn{\nu=1.5}.
//' This covariance should be specified as \strong{matern} with smoothness parameter \eqn{\nu=1.5}.}
//' \item{matern_5_2}{This is the Matérn correlation with \eqn{\nu=2.5}. 
//' This covariance should be specified as \strong{matern} with smoothness parameter \eqn{\nu=2.5}.}
//' }
//'
//'
//' \item{powexp}{The powered-exponential correlation function is given by
//'                \deqn{C(h)=\exp\left\{-\left(\frac{h}{\phi}\right)^{\nu}\right\},}
//' where \eqn{\phi} is the range parameter. \eqn{\nu} is the smoothness parameter.
//' }
//' \item{gauss}{The Gaussian correlation function is given by 
//' \deqn{C(h)=\exp\left(-\frac{h^2}{\phi^2}\right),}
//' where \eqn{\phi} is the range parameter.
//'  }
//' }
//' }
//' 
//' \item{\strong{form}}{
//' \describe{
//'  \item{isotropic}{This indicates the isotropic form of covariance functions. That is,
//'  \deqn{C(\mathbf{h}) = C^0(\|\mathbf{h}\|; \boldsymbol \theta),} where \eqn{\| \mathbf{h}\|} denotes the 
//' Euclidean distance or the great circle distance for data on sphere. \eqn{C^0(\cdot)} denotes 
//' any isotropic covariance family specified in \strong{family}.}
//'  \item{tensor}{This indicates the tensor product of correlation functions. That is, 
//' \deqn{ C(\mathbf{h}) = \prod_{i=1}^d C^0(|h_i|; \boldsymbol \theta_i),}
//' where \eqn{d} is the dimension of input space. \eqn{h_i} is the distance along the \eqn{i}th input dimension. This type of covariance structure has been often used in Gaussian process emulation for computer experiments.
//'}
//'  \item{ARD}{This indicates the automatic relevance determination form. That is, 
//' \deqn{C(\mathbf{h}) = C^0\left(\sqrt{\sum_{i=1}^d\frac{h_i^2}{\phi^2_i}}; \boldsymbol \theta \right),}
//' where \eqn{\phi_i} denotes the range parameter along the \eqn{i}th input dimension.}
//'  }
//' }
//'
//'}
//' 
//' @param smooth The smoothness parameter \eqn{\nu} in a correlation function.
//' @param smoothness_est a logical value indicating whether the smoothness parameter is estimated. 
//' @return The natural logarithm of marginal or integrated likelihood
//' @author Pulong Ma \email{mpulong@@gmail.com}
//' @seealso \code{\link{CH}}, \code{\link{matern}}, \code{\link{gp.optim}}, \link{GPBayes-package}, \code{\link{GaSP}}
// [[Rcpp::export]]
double loglik(Eigen::VectorXd par, Eigen::Map<Eigen::MatrixXd> output, Eigen::Map<Eigen::MatrixXd> H, 
	  SEXP& d, const Rcpp::List& covmodel, const double& smooth, const bool& smoothness_est){

  	std::string family = Rcpp::as<std::string>(covmodel["family"]);
  	std::string form = Rcpp::as<std::string>(covmodel["form"]);
  	//int npar = par.size();
  // if(family=="exp"||family=="matern_3_2"||family=="matern_5_2"){
  // 	family = "matern";
  // 	covmodel["family"] = "matern";
  // }


	int Dim; // dimension of input domain
	Eigen::MatrixXd dmat;
	Rcpp::List dlist;

	Eigen::VectorXd range, tail, nu;
	double nugget;

	SP sp;
	UQ uq;
	double loglik;

	if(form=="isotropic"){
		dmat = Rcpp::as<Eigen::MatrixXd>(d);
		range.resize(1);
		tail.resize(1);
		nu.resize(1);

	  	if(family=="CH"||family=="cauchy"){

	  		range(0) = par(0);
	  		tail(0) = par(1);
	  		nugget = par(2);
	  		if(smoothness_est){
	  			nu(0) = par(3);
	  		}else{
	  			nu(0) = smooth; 
	  		}
	  	}else if(family=="matern" || family=="gauss" || family=="powexp"){
	  		range(0) = par(0);
	  		nugget = par(1);
	  		if(smoothness_est){
	  			nu(0) = par(2);
	  		}else{
	  			nu(0) = smooth;
	  		}
	  	}else{
  			Rcpp::stop("The covariance family is not implemented.\n");
  		}

	  	loglik = sp.MLoglik(range(0), tail(0), nu(0), nugget, output, H, dmat, covmodel);

	}else if(form=="tensor"){
		dlist = Rcpp::as<Rcpp::List>(d);
		Dim = dlist.size();


  		if(family=="CH"||family=="cauchy"){
  			range.resize(Dim);
			tail.resize(Dim);
			nu.resize(Dim);
	  		range = par.head(Dim);
	  		tail = par.segment(Dim, 2*Dim-1);
	  		nugget = par(2*Dim);

	  		if(smoothness_est){
	  			nu = par(2*Dim+1)*Eigen::VectorXd::Ones(Dim);
	  		}else{
	  			nu = smooth*Eigen::VectorXd::Ones(Dim);
	  		}
	  	}else if(family=="matern" || family=="gauss" || family=="powexp"){
	  		range.resize(Dim);
			nu.resize(1);
	  		range = par.head(Dim);
	  		nugget = par(Dim);
	  		if(smoothness_est){
	  			nu=par(Dim+1)*Eigen::VectorXd::Ones(Dim);
	  		}else{
	  			nu = smooth*Eigen::VectorXd::Ones(Dim);
	  		}
	  	}else{
  			Rcpp::stop("The covariance family is not implemented.\n");
  		}

	  	loglik = uq.MLoglik(range, tail, nu, nugget, output, H, dlist, covmodel);

	}else if(form=="ARD"){
		dlist = Rcpp::as<Rcpp::List>(d);
		Dim = dlist.size();

		range.resize(Dim);
		tail.resize(1);
		nu.resize(1);

  		if(family=="CH"||family=="cauchy"){

	  		range = par.head(Dim);
	  		tail(0) = par(Dim);
	  		nugget = par(Dim+1);

	  		if(smoothness_est){
	  			nu(0) = par(Dim+2);
	  		}else{
	  			nu(0) = smooth;
	  		}
	  	}else if(family=="matern" || family=="gauss" || family=="powexp"){
	  		range = par.head(Dim);
	  		nugget = par(Dim);
	  		if(smoothness_est){
	  			nu(0)=par(Dim+1);
	  		}else{
	  			nu(0) = smooth;
	  		}
	  	}else{
  			Rcpp::stop("The covariance family is not implemented.\n");
  		}

	  	loglik = uq.MLoglik(range, tail, nu, nugget, output, H, dlist, covmodel);

	}else{
		Rcpp::stop("The form of covariance function is not supported yet.\n");
	}


	return loglik;
}


// [[Rcpp::export]]
double iso_loglik(Eigen::VectorXd par, Eigen::Map<Eigen::MatrixXd> output, 
	Eigen::Map<Eigen::MatrixXd> H, const Eigen::MatrixXd& d,
	const Rcpp::List& covmodel, const double& smooth, const bool& smoothness_est){

  	std::string family = Rcpp::as<std::string>(covmodel["family"]);
  	std::string form = Rcpp::as<std::string>(covmodel["form"]);
  	//int npar = par.size();
  // if(family=="exp"||family=="matern_3_2"||family=="matern_5_2"){
  // 	family = "matern";
  // 	covmodel["family"] = "matern";
  // }

  	// if(form=="isotropic"){
  	// 	Rcpp::stop("The form of covariance function should be 'isotropic', but it is not correctly specified.\n");
  	// }

	// Eigen::VectorXd range(1), tail(1), nu(1);
	double tail=0.5, nu=smooth;
	double range, nugget;

	SP sp;
	double loglik=0;

  	if(family=="CH"||family=="cauchy"){
  		range = par(0);
  		tail = par(1);
  		nugget = par(2);
  		if(smoothness_est){
  			nu = par(3);
  		}
  	}else if(family=="matern" || family=="gauss" || family=="powexp"){
  		range = par(0);
  		nugget = par(1);
  		if(smoothness_est){
  			nu = par(2);
  		}
  	}else{
  		Rcpp::stop("The covariance family is not implemented.\n");
  	}

	loglik = sp.MLoglik(range, tail, nu, nugget, output, H, d, covmodel);

	return loglik;
}

// [[Rcpp::export]]
double tensor_loglik(Eigen::VectorXd par, Eigen::Map<Eigen::MatrixXd> output, const Eigen::MatrixXd& H, 
	  const Rcpp::List& d, const Rcpp::List& covmodel, const double& smooth, const bool& smoothness_est){

  	std::string family = Rcpp::as<std::string>(covmodel["family"]);
  	std::string form = Rcpp::as<std::string>(covmodel["form"]);
  	//int npar = par.size();
  // if(family=="exp"||family=="matern_3_2"||family=="matern_5_2"){
  // 	family = "matern";
  // 	covmodel["family"] = "matern";
  // }

	int Dim=d.size(); // dimension of input domain

	Eigen::VectorXd range(Dim), tail(Dim), nu(Dim);
	double nugget;

	UQ uq;
	double loglik;


	if(family=="CH"||family=="cauchy"){
  		range = par.head(Dim);
  		tail = par.segment(Dim, 2*Dim-1);
  		nugget = par(2*Dim);

  		if(smoothness_est){
  			nu = par(2*Dim+1)*Eigen::VectorXd::Ones(Dim);
  		}else{
  			nu = smooth*Eigen::VectorXd::Ones(Dim);
  		}
  	}else if(family=="matern" || family=="gauss" || family=="powexp"){
  		range = par.head(Dim);
  		nugget = par(Dim);
  		if(smoothness_est){
  			nu=par(Dim+1)*Eigen::VectorXd::Ones(Dim);
  		}else{
  			nu = smooth*Eigen::VectorXd::Ones(Dim);
  		}
  	}else{
  		Rcpp::stop("The covariance family is not implemented.\n");
  	}

  	loglik = uq.MLoglik(range, tail, nu, nugget, output, H, d, covmodel);


	return loglik;
}


// [[Rcpp::export]]
double ARD_loglik(Eigen::VectorXd par, Eigen::Map<Eigen::MatrixXd> output, Eigen::Map<Eigen::MatrixXd> H, 
	  const Rcpp::List& d, const Rcpp::List& covmodel, const double& smooth, const bool& smoothness_est){

  	std::string family = Rcpp::as<std::string>(covmodel["family"]);
  	std::string form = Rcpp::as<std::string>(covmodel["form"]);
  	//int npar = par.size();
  // if(family=="exp"||family=="matern_3_2"||family=="matern_5_2"){
  // 	family = "matern";
  // 	covmodel["family"] = "matern";
  // }


	int Dim = d.size(); // dimension of input domain

	Eigen::VectorXd range(Dim);
	Eigen::VectorXd tail=0.5*Eigen::VectorXd::Ones(1);
	Eigen::VectorXd nu=smooth*Eigen::VectorXd::Ones(1);
	double nugget;


	UQ uq;
	double loglik;



	if(family=="CH"||family=="cauchy"){

  		range = par.head(Dim);
  		tail(0) = par(Dim);
  		nugget = par(Dim+1);

  		if(smoothness_est){
  			nu(0) = par(Dim+2);
  		}else{
  			nu(0) = smooth;
  		}
  	}else if(family=="matern" || family=="gauss" || family=="powexp"){
  		range = par.head(Dim);
  		nugget = par(Dim);
  		if(smoothness_est){
  			nu(0)=par(Dim+1);
  		}else{
  			nu(0) = smooth;
  		}
  	}

  	loglik = uq.MLoglik(range, tail, nu, nugget, output, H, d, covmodel);


	return loglik;
}

// [[Rcpp::export]]
Eigen::VectorXd gradient_loglik(Eigen::VectorXd par, Eigen::Map<Eigen::MatrixXd> output, 
	Eigen::Map<Eigen::MatrixXd> H, SEXP& d, 
	const Rcpp::List& covmodel, const double& smooth, const bool& smoothness_est){

    std::string family = Rcpp::as<std::string>(covmodel["family"]);
    std::string form = Rcpp::as<std::string>(covmodel["form"]);
	if(family=="exp"||family=="matern_3_2"||family=="matern_5_2"){
		family = "matern";
		covmodel["family"] = "matern";
	}
  	Eigen::VectorXd range, tail, nu; 
  	double nugget;

	int Dim; // dimension of input domain
	Eigen::MatrixXd dmat;
	Rcpp::List dlist;

	//Eigen::VectorXd range, tail, nu;
	//double nugget;

	SP sp;
	UQ uq;

	Eigen::VectorXd dloglik;

	if(form=="isotropic"){
		dmat = Rcpp::as<Eigen::MatrixXd>(d);
		range.resize(1);
		tail.resize(1);
		nu.resize(1);

	  	if(family=="CH"||family=="cauchy"){

	  		range(0) = par(0);
	  		tail(0) = par(1);
	  		nugget = par(2);
	  		if(smoothness_est){
	  			nu(0) = par(3);
	  		}else{
	  			nu(0) = smooth; 
	  		}
	  	}else if(family=="matern" || family=="gauss" || family=="powexp"){
	  		range(0) = par(0);
	  		nugget = par(1);
	  		if(smoothness_est){
	  			nu(0) = par(2);
	  		}else{
	  			nu(0) = smooth;
	  		}
	  	}else{
  			Rcpp::stop("The covariance family is not implemented.\n");
  		}

	  	dloglik = sp.gradient_MLoglik(range(0), tail(0), nu(0), nugget, output, H, dmat, covmodel, smoothness_est);

	}else if(form=="tensor"){
		dlist = Rcpp::as<Rcpp::List>(d);
		Dim = dlist.size();


  		if(family=="CH"||family=="cauchy"){
  			range.resize(Dim);
			tail.resize(Dim);
			nu.resize(Dim);
	  		range = par.head(Dim);
	  		tail = par.segment(Dim, 2*Dim-1);
	  		nugget = par(2*Dim);

	  		if(smoothness_est){
	  			nu = par(2*Dim+1)*Eigen::VectorXd::Ones(Dim);
	  		}else{
	  			nu = smooth*Eigen::VectorXd::Ones(Dim);
	  		}
	  	}else if(family=="matern" || family=="gauss" || family=="powexp"){
	  		range.resize(Dim);
			nu.resize(1);
	  		range = par.head(Dim);
	  		nugget = par(Dim);
	  		if(smoothness_est){
	  			nu=par(Dim+1)*Eigen::VectorXd::Ones(Dim);
	  		}else{
	  			nu = smooth*Eigen::VectorXd::Ones(Dim);
	  		}
	  	}else{
  			Rcpp::stop("The covariance family is not implemented.\n");
  		}

	  	dloglik = uq.gradient_MLoglik(range, tail, nu, nugget, output, H, dlist, covmodel, smoothness_est);

	}else if(form=="ARD"){
		dlist = Rcpp::as<Rcpp::List>(d);
		Dim = dlist.size();

		range.resize(Dim);
		tail.resize(1);
		nu.resize(1);

  		if(family=="CH"||family=="cauchy"){

	  		range = par.head(Dim);
	  		tail(0) = par(Dim);
	  		nugget = par(Dim+1);

	  		if(smoothness_est){
	  			nu(0) = par(Dim+2);
	  		}else{
	  			nu(0) = smooth;
	  		}
	  	}else if(family=="matern" || family=="gauss" || family=="powexp"){
	  		range = par.head(Dim);
	  		nugget = par(Dim);
	  		if(smoothness_est){
	  			nu(0)=par(Dim+1);
	  		}else{
	  			nu(0) = smooth;
	  		}
	  	}else{
  			Rcpp::stop("The covariance family is not implemented.\n");
  		}

	  	dloglik = uq.gradient_MLoglik(range, tail, nu, nugget, output, H, dlist, covmodel, smoothness_est);

	}else{
		Rcpp::stop("The form of covariance function is not supported yet.\n");
	}


	return dloglik;
}

// [[Rcpp::export]]
double loglik_xi(Eigen::VectorXd par, Eigen::Map<Eigen::MatrixXd> output, Eigen::Map<Eigen::MatrixXd> H, 
	  SEXP& d, const Rcpp::List& covmodel, const double& smooth, const bool& smoothness_est){

  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);
  // if(family=="exp"||family=="matern_3_2"||family=="matern_5_2"){
  // 	family = "matern";
  // 	covmodel["family"] = "matern";
  // }
  	par = par.array().exp();

	int Dim; // dimension of input domain
	Eigen::MatrixXd dmat;
	Rcpp::List dlist;

	Eigen::VectorXd range, tail, nu;
	double nugget;

	SP sp;
	UQ uq;
	double loglik;

	if(form=="isotropic"){
		dmat = Rcpp::as<Eigen::MatrixXd>(d);
		range.resize(1);
		tail.resize(1);
		nu.resize(1);

	  	if(family=="CH"||family=="cauchy"){

	  		range(0) = par(0);
	  		tail(0) = par(1);
	  		nugget = par(2);
	  		if(smoothness_est){
	  			nu(0) = par(3);
	  		}else{
	  			nu(0) = smooth; 
	  		}
	  	}else if(family=="matern" || family=="gauss" || family=="powexp"){
	  		range(0) = par(0);
	  		nugget = par(1);
	  		if(smoothness_est){
	  			nu(0) = par(2);
	  		}else{
	  			nu(0) = smooth;
	  		}
	  	}else{
  			Rcpp::stop("The covariance family is not implemented.\n");
  		}

	  	loglik = sp.MLoglik(range(0), tail(0), nu(0), nugget, output, H, dmat, covmodel);

	}else if(form=="tensor"){
		dlist = Rcpp::as<Rcpp::List>(d);
		Dim = dlist.size();


  		if(family=="CH"||family=="cauchy"){
  			range.resize(Dim);
			tail.resize(Dim);
			nu.resize(Dim);
	  		range = par.head(Dim);
	  		tail = par.segment(Dim, 2*Dim-1);
	  		nugget = par(2*Dim);

	  		if(smoothness_est){
	  			nu = par(2*Dim+1)*Eigen::VectorXd::Ones(Dim);
	  		}else{
	  			nu = smooth*Eigen::VectorXd::Ones(Dim);
	  		}
	  	}else if(family=="matern" || family=="gauss" || family=="powexp"){
	  		range.resize(Dim);
			nu.resize(1);
	  		range = par.head(Dim);
	  		nugget = par(Dim);
	  		if(smoothness_est){
	  			nu=par(Dim+1)*Eigen::VectorXd::Ones(Dim);
	  		}else{
	  			nu = smooth*Eigen::VectorXd::Ones(Dim);
	  		}
	  	}else{
  			Rcpp::stop("The covariance family is not implemented.\n");
  		}

	  	loglik = uq.MLoglik(range, tail, nu, nugget, output, H, dlist, covmodel);

	}else if(form=="ARD"){
		dlist = Rcpp::as<Rcpp::List>(d);
		Dim = dlist.size();

		range.resize(Dim);
		tail.resize(1);
		nu.resize(1);

  		if(family=="CH"||family=="cauchy"){

	  		range = par.head(Dim);
	  		tail(0) = par(Dim);
	  		nugget = par(Dim+1);

	  		if(smoothness_est){
	  			nu(0) = par(Dim+2);
	  		}else{
	  			nu(0) = smooth;
	  		}
	  	}else if(family=="matern" || family=="gauss" || family=="powexp"){
	  		range = par.head(Dim);
	  		nugget = par(Dim);
	  		if(smoothness_est){
	  			nu(0)=par(Dim+1);
	  		}else{
	  			nu(0) = smooth;
	  		}
	  	}else{
  			Rcpp::stop("The covariance family is not implemented.\n");
  		}

	  	loglik = uq.MLoglik(range, tail, nu, nugget, output, H, dlist, covmodel);

	}else{
		Rcpp::stop("The form of covariance function is not supported yet.\n");
	}

	return loglik;
}

// [[Rcpp::export]]
Eigen::VectorXd gradient_loglik_xi(Eigen::VectorXd par, Eigen::Map<Eigen::MatrixXd> output, 
	Eigen::Map<Eigen::MatrixXd> H, SEXP& d, 
	const Rcpp::List& covmodel, const double& smooth, const bool& smoothness_est){

    std::string family = Rcpp::as<std::string>(covmodel["family"]);
    std::string form = Rcpp::as<std::string>(covmodel["form"]);
  // if(family=="exp"||family=="matern_3_2"||family=="matern_5_2"){
  // 	family = "matern";
  // 	covmodel["family"] = "matern";
  // }
  	par = par.array().exp();

  	Eigen::VectorXd range, tail, nu; 
  	double nugget;

	int Dim; // dimension of input domain
	Eigen::MatrixXd dmat;
	Rcpp::List dlist;

	//Eigen::VectorXd range, tail, nu;
	//double nugget;

	SP sp;
	UQ uq;

	Eigen::VectorXd dloglik;

	if(form=="isotropic"){
		dmat = Rcpp::as<Eigen::MatrixXd>(d);
		range.resize(1);
		tail.resize(1);
		nu.resize(1);

	  	if(family=="CH"||family=="cauchy"){

	  		range(0) = par(0);
	  		tail(0) = par(1);
	  		nugget = par(2);
	  		if(smoothness_est){
	  			nu(0) = par(3);
	  		}else{
	  			nu(0) = smooth; 
	  		}
	  	}else if(family=="matern" || family=="gauss" || family=="powexp"){
	  		range(0) = par(0);
	  		nugget = par(1);
	  		if(smoothness_est){
	  			nu(0) = par(2);
	  		}else{
	  			nu(0) = smooth;
	  		}
	  	}else{
  			Rcpp::stop("The covariance family is not implemented.\n");
  		}

	  	dloglik = sp.gradient_MLoglik(range(0), tail(0), nu(0), nugget, output, H, dmat, covmodel, smoothness_est);

	}else if(form=="tensor"){
		dlist = Rcpp::as<Rcpp::List>(d);
		Dim = dlist.size();


  		if(family=="CH"||family=="cauchy"){
  			range.resize(Dim);
			tail.resize(Dim);
			nu.resize(Dim);
	  		range = par.head(Dim);
	  		tail = par.segment(Dim, 2*Dim-1);
	  		nugget = par(2*Dim);

	  		if(smoothness_est){
	  			nu = par(2*Dim+1)*Eigen::VectorXd::Ones(Dim);
	  		}else{
	  			nu = smooth*Eigen::VectorXd::Ones(Dim);
	  		}
	  	}else if(family=="matern" || family=="gauss" || family=="powexp"){
	  		range.resize(Dim);
			nu.resize(1);
	  		range = par.head(Dim);
	  		nugget = par(Dim);
	  		if(smoothness_est){
	  			nu=par(Dim+1)*Eigen::VectorXd::Ones(Dim);
	  		}else{
	  			nu = smooth*Eigen::VectorXd::Ones(Dim);
	  		}
	  	}else{
  			Rcpp::stop("The covariance family is not implemented.\n");
  		}

	  	dloglik = uq.gradient_MLoglik(range, tail, nu, nugget, output, H, dlist, covmodel, smoothness_est);

	}else if(form=="ARD"){
		dlist = Rcpp::as<Rcpp::List>(d);
		Dim = dlist.size();

		range.resize(Dim);
		tail.resize(1);
		nu.resize(1);

  		if(family=="CH"||family=="cauchy"){

	  		range = par.head(Dim);
	  		tail(0) = par(Dim);
	  		nugget = par(Dim+1);

	  		if(smoothness_est){
	  			nu(0) = par(Dim+2);
	  		}else{
	  			nu(0) = smooth;
	  		}
	  	}else if(family=="matern" || family=="gauss" || family=="powexp"){
	  		range = par.head(Dim);
	  		nugget = par(Dim);
	  		if(smoothness_est){
	  			nu(0)=par(Dim+1);
	  		}else{
	  			nu(0) = smooth;
	  		}
	  	}else{
  			Rcpp::stop("The covariance family is not implemented.\n");
  		}

	  	dloglik = uq.gradient_MLoglik(range, tail, nu, nugget, output, H, dlist, covmodel, smoothness_est);

	}else{
		Rcpp::stop("The form of covariance function is not supported yet.\n");
	}


	return dloglik;
}

//' @title Compute distances for two sets of inputs 
//' 
//' @description This function computes distances for two sets of inputs and returns
//' a \code{R} object. 
//'
//' @param input1 a matrix of inputs
//' @param input2 a matrix of inputs
//' @param type a string indicating the form of distances with three froms supported currently: \strong{isotropic}, \strong{tensor}, \strong{ARD}.
//' @param dtype a string indicating distance type: \strong{Euclidean}, \strong{GCD}, where the latter indicates great circle distance.
//' @return a R object holding distances for two sets of inputs. If \strong{type} is \strong{isotropic}, a matrix of distances
//' is returned; if \strong{type} is \strong{tensor} or \strong{ARD}, a list of distance matrices 
//' along each input dimension is returned.
//' @author Pulong Ma \email{mpulong@@gmail.com}
//' @return a numeric vector or matrix of distances 
//' @examples
//' input = seq(0,1,length=20)
//' d = distance(input, input, type="isotropic", dtype="Euclidean")
//'
// [[Rcpp::export]]
SEXP distance(Eigen::Map<Eigen::MatrixXd> input1, Eigen::Map<Eigen::MatrixXd> input2, 
	std::string type="isotropic", std::string dtype="Euclidean"){

	if(type=="isotropic"){
		SP sp;
		Eigen::MatrixXd mat_iso = sp.pdist(input1, input2, dtype);
		return Rcpp::wrap(mat_iso);
	}else if(type=="tensor"||type=="ARD"){
		UQ uq;
		Rcpp::List mat = uq.adist(input1, input2);
		return Rcpp::wrap(mat);
	}else{
		Rcpp::stop("The covariance kernel is not supported yet.\n");
	}

}



//' @title A wraper to build different kinds of correlation matrices between two sets of inputs
//' @name ikernel
//' @description This function wraps existing built-in routines to construct a covariance 
//' matrix for two input matrices based on data type, covariance type, and distance type. The constructed 
//' covariance matrix can be directly used for GaSP fitting and and prediction for spatial 
//' data, spatio-temporal data, and computer experiments. This function explicitly takes inputs as arguments. 
//' The prefix ``i" in \code{\link{ikernel}} standards for ``input".
//'
//' @param input1 a matrix of input locations 
//' @param input2 a matrix of input locations
//' 
//' @param range a vector of range parameters, which could be a scalar.
//' @param tail a vector of tail decay parameters, which could be a scalar.
//' @param nu a vector of smoothness parameters, which could be a scalar.
//' @param covmodel a list of two strings: \strong{family}, \strong{form}, where \strong{family} indicates the family of covariance functions 
//' including the Confluent Hypergeometric class, the Matérn class, the Cauchy class, the powered-exponential class. \strong{form} indicates the 
//' specific form of covariance structures including the isotropic form, tensor form, automatic relevance determination form. 
//' \describe{
//' \item{\strong{family}}{
//' \describe{
//' \item{CH}{The Confluent Hypergeometric correlation function is given by 
//' \deqn{C(h) = \frac{\Gamma(\nu+\alpha)}{\Gamma(\nu)} 
//' \mathcal{U}\left(\alpha, 1-\nu, \left(\frac{h}{\beta}\right)^2\right),}
//' where \eqn{\alpha} is the tail decay parameter. \eqn{\beta} is the range parameter.
//' \eqn{\nu} is the smoothness parameter. \eqn{\mathcal{U}(\cdot)} is the confluent hypergeometric
//' function of the second kind. For details about this covariance, 
//' see Ma and Bhadra (2019) at \url{https://arxiv.org/abs/1911.05865}.  
//' }
//' \item{cauchy}{The generalized Cauchy covariance is given by
//' \deqn{C(h) = \left\{ 1 + \left( \frac{h}{\phi} \right)^{\nu}  
//'             \right\}^{-\alpha/\nu},}
//' where \eqn{\phi} is the range parameter. \eqn{\alpha} is the tail decay parameter.
//' \eqn{\nu} is the smoothness parameter with default value at 2.
//'}
//'
//' \item{matern}{The Matérn correlation function is given by
//' \deqn{C(h)=\frac{2^{1-\nu}}{\Gamma(\nu)} \left( \frac{h}{\phi} \right)^{\nu} 
//' \mathcal{K}_{\nu}\left( \frac{h}{\phi} \right),}
//' where \eqn{\phi} is the range parameter. \eqn{\nu} is the smoothness parameter. 
//' \eqn{\mathcal{K}_{\nu}(\cdot)} is the modified Bessel function of the second kind of order \eqn{\nu}.
//' }
//' \describe{
//' \item{exp}{This is the Matérn correlation with \eqn{\nu=0.5}. This covariance should be specified as \strong{matern} with smoothness parameter \eqn{\nu=0.5}.
//' }
//' \item{matern_3_2}{This is the Matérn correlation with \eqn{\nu=1.5}.
//' This covariance should be specified as \strong{matern} with smoothness parameter \eqn{\nu=1.5}.}
//' \item{matern_5_2}{This is the Matérn correlation with \eqn{\nu=2.5}. 
//' This covariance should be specified as \strong{matern} with smoothness parameter \eqn{\nu=2.5}.}
//' }
//'
//'
//' \item{powexp}{The powered-exponential correlation function is given by
//'                \deqn{C(h)=\exp\left\{-\left(\frac{h}{\phi}\right)^{\nu}\right\},}
//' where \eqn{\phi} is the range parameter. \eqn{\nu} is the smoothness parameter.
//' }
//' \item{gauss}{The Gaussian correlation function is given by 
//' \deqn{C(h)=\exp\left(-\frac{h^2}{\phi^2}\right),}
//' where \eqn{\phi} is the range parameter.
//'  }
//' }
//' }
//' 
//' \item{\strong{form}}{
//' \describe{
//'  \item{isotropic}{This indicates the isotropic form of covariance functions. That is,
//'  \deqn{C(\mathbf{h}) = C^0(\|\mathbf{h}\|; \boldsymbol \theta),} where \eqn{\| \mathbf{h}\|} denotes the 
//' Euclidean distance or the great circle distance for data on sphere. \eqn{C^0(\cdot)} denotes 
//' any isotropic covariance family specified in \strong{family}.}
//'  \item{tensor}{This indicates the tensor product of correlation functions. That is, 
//' \deqn{ C(\mathbf{h}) = \prod_{i=1}^d C^0(|h_i|; \boldsymbol \theta_i),}
//' where \eqn{d} is the dimension of input space. \eqn{h_i} is the distance along the \eqn{i}th input dimension. This type of covariance structure has been often used in Gaussian process emulation for computer experiments.
//'}
//'  \item{ARD}{This indicates the automatic relevance determination form. That is, 
//' \deqn{C(\mathbf{h}) = C^0\left(\sqrt{\sum_{i=1}^d\frac{h_i^2}{\phi^2_i}}; \boldsymbol \theta \right),}
//' where \eqn{\phi_i} denotes the range parameter along the \eqn{i}th input dimension.}
//'  }
//' }
//'
//'}
//' 
//' @param dtype a string indicating distance type: \strong{Euclidean}, \strong{GCD}, where the latter indicates great circle distance.

//' @return a correlation matrix
//' @author Pulong Ma \email{mpulong@@gmail.com}
//' @examples
//' input = seq(0,1,length=10)
//' 
//' cormat = ikernel(input,input,range=0.5,tail=0.2,nu=2.5,
//'          covmodel=list(family="CH",form="isotropic"))
//'
//' @seealso \code{\link{CH}}, \code{\link{matern}}, \code{\link{kernel}}, \link{GPBayes-package}, \code{\link{GaSP}}
// [[Rcpp::export]]
Eigen::MatrixXd ikernel(Eigen::Map<Eigen::MatrixXd> input1, Eigen::Map<Eigen::MatrixXd> input2, const Eigen::VectorXd& range, 
	const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, const Rcpp::List& covmodel, std::string dtype="Euclidean"){
	
	std::string family = Rcpp::as<std::string>(covmodel["family"]);
	std::string form = Rcpp::as<std::string>(covmodel["form"]);

	// Eigen::VectorXd range, tail, nu;
	// double nugget = Rcpp::as<double>(par["nugget"]);
	// range = Rcpp::as<Eigen::VectorXd>(par["range"]);
	// tail = Rcpp::as<Eigen::VectorXd>(par["tail"]);
	// nu = Rcpp::as<Eigen::VectorXd>(par["nu"]);


	UQ uq;
	Eigen::MatrixXd mat;
	if(form=="isotropic"){
		SP sp;

		Eigen::MatrixXd dmat = sp.pdist(input1, input2, dtype);
		mat = sp.iso_kernel(dmat, range(0), tail(0), nu(0), family);
	}else if(form=="tensor"){
		Rcpp::List dlist = uq.adist(input1, input2);
		mat = uq.tensor_kernel(dlist, range, tail, nu, family);
	}else if(form=="ARD"){
		Rcpp::List dlist1 = uq.adist(input1, input2);
		mat = uq.ARD_kernel(dlist1, range, tail(0), nu(0), family);		
	}else{
		Rcpp::stop("The covariance kernel is not supported yet.\n");
	}

	return mat;
}

//' @title A wraper to build different kinds of correlation matrices with distance as arguments 
//' 
//' @description This function wraps existing built-in routines to construct a covariance 
//' matrix based on data type, covariance type, and distance type with distances as inputs. The constructed 
//' covariance matrix can be directly used for GaSP fitting and and prediction for spatial 
//' data, spatio-temporal data, and computer experiments. 
//'
//' @param d a matrix or a list of distances
//' 
//' @param range a vector of range parameters, which could be a scalar. 
//' @param tail a vector of tail decay parameters, which could be a scalar.
//' @param nu a vector of smoothness parameters, which could be a scalar.
//' @param covmodel a list of two strings: \strong{family}, \strong{form}, where \strong{family} indicates the family of covariance functions 
//' including the Confluent Hypergeometric class, the Matérn class, the Cauchy class, the powered-exponential class. \strong{form} indicates the 
//' specific form of covariance structures including the isotropic form, tensor form, automatic relevance determination form. 
//' \describe{
//' \item{\strong{family}}{
//' \describe{
//' \item{CH}{The Confluent Hypergeometric correlation function is given by 
//' \deqn{C(h) = \frac{\Gamma(\nu+\alpha)}{\Gamma(\nu)} 
//' \mathcal{U}\left(\alpha, 1-\nu, \left(\frac{h}{\beta}\right)^2\right),}
//' where \eqn{\alpha} is the tail decay parameter. \eqn{\beta} is the range parameter.
//' \eqn{\nu} is the smoothness parameter. \eqn{\mathcal{U}(\cdot)} is the confluent hypergeometric
//' function of the second kind. For details about this covariance, 
//' see Ma and Bhadra (2019) at \url{https://arxiv.org/abs/1911.05865}.  
//' }
//' \item{cauchy}{The generalized Cauchy covariance is given by
//' \deqn{C(h) = \left\{ 1 + \left( \frac{h}{\phi} \right)^{\nu}  
//'             \right\}^{-\alpha/\nu},}
//' where \eqn{\phi} is the range parameter. \eqn{\alpha} is the tail decay parameter.
//' \eqn{\nu} is the smoothness parameter with default value at 2.
//'}
//'
//' \item{matern}{The Matérn correlation function is given by
//' \deqn{C(h)=\frac{2^{1-\nu}}{\Gamma(\nu)} \left( \frac{h}{\phi} \right)^{\nu} 
//' \mathcal{K}_{\nu}\left( \frac{h}{\phi} \right),}
//' where \eqn{\phi} is the range parameter. \eqn{\nu} is the smoothness parameter. 
//' \eqn{\mathcal{K}_{\nu}(\cdot)} is the modified Bessel function of the second kind of order \eqn{\nu}.
//' }
//' \describe{
//' \item{exp}{This is the Matérn correlation with \eqn{\nu=0.5}. This covariance should be specified as \strong{matern} with smoothness parameter \eqn{\nu=0.5}.
//' }
//' \item{matern_3_2}{This is the Matérn correlation with \eqn{\nu=1.5}.
//' This covariance should be specified as \strong{matern} with smoothness parameter \eqn{\nu=1.5}.}
//' \item{matern_5_2}{This is the Matérn correlation with \eqn{\nu=2.5}. 
//' This covariance should be specified as \strong{matern} with smoothness parameter \eqn{\nu=2.5}.}
//' }
//'
//' \item{powexp}{The powered-exponential correlation function is given by
//'                \deqn{C(h)=\exp\left\{-\left(\frac{h}{\phi}\right)^{\nu}\right\},}
//' where \eqn{\phi} is the range parameter. \eqn{\nu} is the smoothness parameter.
//' }
//' \item{gauss}{The Gaussian correlation function is given by 
//' \deqn{C(h)=\exp\left(-\frac{h^2}{\phi^2}\right),}
//' where \eqn{\phi} is the range parameter.
//'  }
//' }
//' }
//' 
//' \item{\strong{form}}{
//' \describe{
//'  \item{isotropic}{This indicates the isotropic form of covariance functions. That is,
//'  \deqn{C(\mathbf{h}) = C^0(\|\mathbf{h}\|; \boldsymbol \theta),} where \eqn{\| \mathbf{h}\|} denotes the 
//' Euclidean distance or the great circle distance for data on sphere. \eqn{C^0(\cdot)} denotes 
//' any isotropic covariance family specified in \strong{family}.}
//'  \item{tensor}{This indicates the tensor product of correlation functions. That is, 
//' \deqn{ C(\mathbf{h}) = \prod_{i=1}^d C^0(|h_i|; \boldsymbol \theta_i),}
//' where \eqn{d} is the dimension of input space. \eqn{h_i} is the distance along the \eqn{i}th input dimension. This type of covariance structure has been often used in Gaussian process emulation for computer experiments.
//'}
//'  \item{ARD}{This indicates the automatic relevance determination form. That is, 
//' \deqn{C(\mathbf{h}) = C^0\left(\sqrt{\sum_{i=1}^d\frac{h_i^2}{\phi^2_i}}; \boldsymbol \theta \right),}
//' where \eqn{\phi_i} denotes the range parameter along the \eqn{i}th input dimension.}
//'  }
//' }
//'
//'}
//' 
//' @return a correlation matrix 
//' @author Pulong Ma \email{mpulong@@gmail.com}
//' @seealso \code{\link{CH}}, \code{\link{matern}}, \code{\link{ikernel}}, \link{GPBayes-package}, \code{\link{GaSP}}
//' @examples
//' input = seq(0,1,length=10)
//' d = distance(input,input,type="isotropic",dtype="Euclidean")
//' cormat = kernel(d,range=0.5,tail=0.2,nu=2.5,
//'          covmodel=list(family="CH",form="isotropic"))
// [[Rcpp::export]]
Eigen::MatrixXd kernel(SEXP& d, const Eigen::VectorXd& range, 
	const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, const Rcpp::List& covmodel){
	
	std::string family = Rcpp::as<std::string>(covmodel["family"]);
	std::string form = Rcpp::as<std::string>(covmodel["form"]);
	// if(family=="exp" || family=="matern_3_2" || family=="matern_5_2"){
	// 	family = "matern";
	// }
	// Eigen::VectorXd range, tail, nu;
	// double nugget = Rcpp::as<double>(par["nugget"]);
	// range = Rcpp::as<Eigen::VectorXd>(par["range"]);
	// tail = Rcpp::as<Eigen::VectorXd>(par["tail"]);
	// nu = Rcpp::as<Eigen::VectorXd>(par["nu"]);

	UQ uq;
	Eigen::MatrixXd mat;
	if(form=="isotropic"){
		SP sp;
		Eigen::MatrixXd dmat = Rcpp::as<Eigen::MatrixXd>(d);
		mat = sp.iso_kernel(dmat, range(0), tail(0), nu(0), family);
	}else if(form=="tensor"){
		Rcpp::List dlist = Rcpp::as<Rcpp::List>(d);
		mat = uq.tensor_kernel(dlist, range, tail, nu, family);
	}else if(form=="ARD"){
		Rcpp::List dlist1 = Rcpp::as<Rcpp::List>(d);
		mat = uq.ARD_kernel(dlist1, range, tail(0), nu(0), family);		
	}else{
		Rcpp::stop("The covariance kernel is not supported yet.\n");
	}

	return mat;
}

//' @title A wraper to construct the derivative of correlation matrix with respect to correlation parameters 
//' 
//' @description This function wraps existing built-in routines to construct the 
//' derivative of correlation matrix with respect to correlation parameters. 
//' @param d a matrix or a list of distances returned from \code{\link{distance}}.
//' 
//' @param range a vector of range parameters 
//' @param tail a vector of tail decay parameters
//' @param nu a vector of smoothness parameters
//' @param covmodel a list of two strings: \strong{family}, \strong{form}, where \strong{family} indicates the family of covariance functions 
//' including the Confluent Hypergeometric class, the Matérn class, the Cauchy class, the powered-exponential class. \strong{form} indicates the 
//' specific form of covariance structures including the isotropic form, tensor form, automatic relevance determination form. 
//' \describe{
//' \item{\strong{family}}{
//' \describe{
//' \item{CH}{The Confluent Hypergeometric correlation function is given by 
//' \deqn{C(h) = \frac{\Gamma(\nu+\alpha)}{\Gamma(\nu)} 
//' \mathcal{U}\left(\alpha, 1-\nu, \left(\frac{h}{\beta}\right)^2\right),}
//' where \eqn{\alpha} is the tail decay parameter. \eqn{\beta} is the range parameter.
//' \eqn{\nu} is the smoothness parameter. \eqn{\mathcal{U}(\cdot)} is the confluent hypergeometric
//' function of the second kind. For details about this covariance, 
//' see Ma and Bhadra (2019) at \url{https://arxiv.org/abs/1911.05865}.  
//' }
//' \item{cauchy}{The generalized Cauchy covariance is given by
//' \deqn{C(h) = \left\{ 1 + \left( \frac{h}{\phi} \right)^{\nu}  
//'             \right\}^{-\alpha/\nu},}
//' where \eqn{\phi} is the range parameter. \eqn{\alpha} is the tail decay parameter.
//' \eqn{\nu} is the smoothness parameter with default value at 2.
//'}
//'
//' \item{matern}{The Matérn correlation function is given by
//' \deqn{C(h)=\frac{2^{1-\nu}}{\Gamma(\nu)} \left( \frac{h}{\phi} \right)^{\nu} 
//' \mathcal{K}_{\nu}\left( \frac{h}{\phi} \right),}
//' where \eqn{\phi} is the range parameter. \eqn{\nu} is the smoothness parameter. 
//' \eqn{\mathcal{K}_{\nu}(\cdot)} is the modified Bessel function of the second kind of order \eqn{\nu}.
//' }
//' \describe{
//' \item{exp}{This is the Matérn correlation with \eqn{\nu=0.5}. This covariance should be specified as \strong{matern} with smoothness parameter \eqn{\nu=0.5}.
//' }
//' \item{matern_3_2}{This is the Matérn correlation with \eqn{\nu=1.5}.
//' This covariance should be specified as \strong{matern} with smoothness parameter \eqn{\nu=1.5}.}
//' \item{matern_5_2}{This is the Matérn correlation with \eqn{\nu=2.5}. 
//' This covariance should be specified as \strong{matern} with smoothness parameter \eqn{\nu=2.5}.}
//' }
//'
//'
//' \item{powexp}{The powered-exponential correlation function is given by
//'                \deqn{C(h)=\exp\left\{-\left(\frac{h}{\phi}\right)^{\nu}\right\},}
//' where \eqn{\phi} is the range parameter. \eqn{\nu} is the smoothness parameter.
//' }
//' \item{gauss}{The Gaussian correlation function is given by 
//' \deqn{C(h)=\exp\left(-\frac{h^2}{\phi^2}\right),}
//' where \eqn{\phi} is the range parameter.
//'  }
//' }
//' }
//' 
//' \item{\strong{form}}{
//' \describe{
//'  \item{isotropic}{This indicates the isotropic form of covariance functions. That is,
//'  \deqn{C(\mathbf{h}) = C^0(\|\mathbf{h}\|; \boldsymbol \theta),} where \eqn{\| \mathbf{h}\|} denotes the 
//' Euclidean distance or the great circle distance for data on sphere. \eqn{C^0(\cdot)} denotes 
//' any isotropic covariance family specified in \strong{family}.}
//'  \item{tensor}{This indicates the tensor product of correlation functions. That is, 
//' \deqn{ C(\mathbf{h}) = \prod_{i=1}^d C^0(|h_i|; \boldsymbol \theta_i),}
//' where \eqn{d} is the dimension of input space. \eqn{h_i} is the distance along the \eqn{i}th input dimension. This type of covariance structure has been often used in Gaussian process emulation for computer experiments.
//'}
//'  \item{ARD}{This indicates the automatic relevance determination form. That is, 
//' \deqn{C(\mathbf{h}) = C^0\left(\sqrt{\sum_{i=1}^d\frac{h_i^2}{\phi^2_i}}; \boldsymbol \theta \right),}
//' where \eqn{\phi_i} denotes the range parameter along the \eqn{i}th input dimension.}
//'  }
//' }
//'
//'}
//' 
//' @return a list of matrices 
//' @author Pulong Ma \email{mpulong@@gmail.com}
//' @seealso \code{\link{CH}}, \code{\link{matern}}, \code{\link{kernel}}, \link{GPBayes-package}, \code{\link{GaSP}}
//' @examples
//' input = seq(0,1,length=10)
//' d = distance(input,input,type="isotropic",dtype="Euclidean")
//' dR = deriv_kernel(d,range=0.5,tail=0.2,nu=2.5,
//'          covmodel=list(family="CH",form="isotropic"))
// [[Rcpp::export]]
Rcpp::List deriv_kernel(SEXP& d, const Eigen::VectorXd& range, 
	const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, 
	const Rcpp::List& covmodel){

	std::string family = Rcpp::as<std::string>(covmodel["family"]);
	std::string form = Rcpp::as<std::string>(covmodel["form"]);
	if(family=="exp" || family=="matern_3_2" || family=="matern_5_2"){
		family = "matern";
	}
	// Eigen::VectorXd range, tail, nu;
	// double nugget = Rcpp::as<double>(par["nugget"]);
	// range = Rcpp::as<Eigen::VectorXd>(par["range"]);
	// tail = Rcpp::as<Eigen::VectorXd>(par["tail"]);
	// nu = Rcpp::as<Eigen::VectorXd>(par["nu"]);
  

	UQ uq;

	Rcpp::List matdot;

	if(form=="isotropic"){
		SP sp;
		Eigen::MatrixXd dmat = Rcpp::as<Eigen::MatrixXd>(d);
		matdot = sp.deriv_iso_kernel(dmat, range(0), tail(0), nu(0), family);
	}else if(form=="tensor"){
		Rcpp::List dlist = Rcpp::as<Rcpp::List>(d);
		matdot = uq.deriv_tensor_kernel(dlist, range, tail, nu, family);
	}else if(form=="ARD"){
		Rcpp::List dlist1 = Rcpp::as<Rcpp::List>(d);
		matdot = uq.deriv_ARD_kernel(dlist1, range, tail(0), nu(0), family);
	}else{
		Rcpp::stop("The covariance kernel is not supported yet.\n");
	}

	return matdot;
}




//[[Rcpp::export]]
double reference_prior(const Eigen::MatrixXd& H, SEXP d, const Rcpp::List& par, const Rcpp::List& covmodel,
    const bool& smoothness_est){

  	std::string family = Rcpp::as<std::string>(covmodel["family"]);
  	std::string form = Rcpp::as<std::string>(covmodel["form"]);

	Eigen::VectorXd range;
	if(par.containsElementNamed("range")){
		range = Rcpp::as<Eigen::VectorXd>(par["range"]);
	}else{
		Rcpp::stop("reference_prior: No range parameter is specified.\n");
	}

	Eigen::VectorXd tail;
	if(par.containsElementNamed("tail")){
		tail = Rcpp::as<Eigen::VectorXd>(par["tail"]);
	}else{
		tail = 0.5*Eigen::VectorXd::Ones(1); 
	}

	Eigen::VectorXd nu;
	if(par.containsElementNamed("nu")){
		nu = Rcpp::as<Eigen::VectorXd>(par["nu"]);
	}else{
		nu = 0.5*Eigen::VectorXd::Ones(1);
	}
	
	double nugget = 0;
	if(par.containsElementNamed("nugget")){
		nugget = Rcpp::as<double>(par["nugget"]);
	}


	int n=H.rows();
	int p=H.cols();
	
	Eigen::MatrixXd R(n,n), RInv(n,n), Q(n,n), RH(n,p), HRH(p,p);
	Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH;
	
  Rcpp::List dR, dlist;
  UQ uq;
  SP sp;

  if(form=="tensor"){
  	dlist = Rcpp::as<Rcpp::List>(d);
    R = uq.tensor_kernel(dlist, range, tail, nu, family);
    dR = uq.deriv_tensor_kernel(dlist, range, tail, nu, family);
  }else if(form=="ARD"){
  	dlist = Rcpp::as<Rcpp::List>(d);
    R = uq.ARD_kernel(dlist, range, tail(0), nu(0), family);
    dR = uq.deriv_ARD_kernel(dlist, range, tail(0), nu(0), family);
  }else if(form=="isotropic"){
  	Eigen::MatrixXd dmat = Rcpp::as<Eigen::MatrixXd>(d);
  	R = sp.iso_kernel(dmat, range(0), tail(0), nu(0), family);
  	dR = sp.deriv_iso_kernel(dmat, range(0), tail(0), nu(0), family);
  }else{
    Rcpp::stop("The reference_prior for the specified form of covariance functions is not supported yet.\n");
  } 
  int npars = dR.size()-1;

  R.diagonal().array() += nugget;

  ldltR.compute(R);
  RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
  
  RH = RInv * H;
  HRH = H.transpose() * RH;
  ldltH.compute(HRH);
  Q = RInv - RH*ldltH.solve(RH.transpose());

  Rcpp::List W(npars+2);
  Eigen::MatrixXd W_l(n, n), W_k(n, n);

  for(int k=0; k<npars; k++){
    W[k] = Rcpp::as<Eigen::MatrixXd>(dR[k])*Q;
  }

  W[npars] = Q; // corresponding to nugget

  int count;

  Eigen::MatrixXd FisherIR(npars+3, npars+3);


	if(family=="CH"){
	  if(smoothness_est){ //include smoothness parameter

	    W[npars+1] = Rcpp::as<Eigen::MatrixXd>(dR[npars])*Q;  // corresponding to smoothness parameter

	    FisherIR(0,0) = n - p;  // corresponding to variance parameter
	    for(int l=0; l<(npars+2); l++){
	      W_l = Rcpp::as<Eigen::MatrixXd>(W[l]);

	      FisherIR(0, l+1) = W_l.trace();
	      FisherIR(l+1, 0) = W_l.trace();

	      for(int k=0; k<(npars+2); k++){
	        W_k = Rcpp::as<Eigen::MatrixXd>(W[k]);
	        FisherIR(l+1, k+1) = (W_l*W_k).trace();
	        FisherIR(k+1, l+1) = (W_l*W_k).trace();
	      }
	    }

	    count = npars + 3;    

	  }else{

	    FisherIR(0,0) = n - p;
	    for(int l=0; l<npars+1; l++){
	      W_l = Rcpp::as<Eigen::MatrixXd>(W[l]);
	      FisherIR(0, l+1) = W_l.trace();
	      FisherIR(l+1, 0) = W_l.trace();

	      for(int k=0; k<npars+1; k++){
	        W_k = Rcpp::as<Eigen::MatrixXd>(W[k]);
	        FisherIR(l+1, k+1) = (W_l*W_k).trace();
	        FisherIR(k+1, l+1) = (W_l*W_k).trace();
	      }

	    }  

	    count = npars + 2;

	  }
	}else{

	    FisherIR(0,0) = n - p;
	    for(int l=0; l<npars+1; l++){
	      W_l = Rcpp::as<Eigen::MatrixXd>(W[l]);
	      FisherIR(0, l+1) = W_l.trace();
	      FisherIR(l+1, 0) = W_l.trace();

	      for(int k=0; k<npars+1; k++){
	        W_k = Rcpp::as<Eigen::MatrixXd>(W[k]);
	        FisherIR(l+1, k+1) = (W_l*W_k).trace();
	        FisherIR(k+1, l+1) = (W_l*W_k).trace();
	      }

	    }  

	    count = npars + 2;
	}


  ldltR.compute(FisherIR.block(0,0,count,count));
  double lndetI = 0.5*ldltR.vectorD().array().log().sum();


	return lndetI;
} 





// [[Rcpp::export]]
Eigen::MatrixXd FisherInfo(const Eigen::MatrixXd& input, 
	const double& sig2, const Eigen::VectorXd& range, 
	const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, 
	const double& nugget, const Rcpp::List& covmodel, 
	const std::string& dtype){

	std::string family = Rcpp::as<std::string>(covmodel["family"]);
	std::string form = Rcpp::as<std::string>(covmodel["form"]);

	// double sig2=1.0;
	// if(par.containsElementNamed("sig2")){
	// 	sig2 = Rcpp::as<double>(par["sig2"]);
	// }else{
	// 	Rcpp::Rcout<<"The variance parameter is set as default value 1.\n";
	// }

	// Eigen::VectorXd range;
	// if(par.containsElementNamed("range")){
	// 	range = Rcpp::as<Eigen::VectorXd>(par["range"]);
	// }else{
	// 	Rcpp::stop("FisherInfo: No range parameter value is specified.\n");
	// }

	// Eigen::VectorXd tail;
	// if(par.containsElementNamed("tail")){
	// 	tail = Rcpp::as<Eigen::VectorXd>(par["tail"]);
	// }else{
	// 	tail = 0.5*Eigen::VectorXd::Ones(1); 
	// }

	// Eigen::VectorXd nu;
	// if(par.containsElementNamed("nu")){
	// 	nu = Rcpp::as<Eigen::VectorXd>(par["nu"]);
	// }else{
	// 	nu = 0.5*Eigen::VectorXd::Ones(1);
	// }
	
	// double nugget = 0;
	// if(par.containsElementNamed("nugget")){
	// 	nugget = Rcpp::as<double>(par["nugget"]);
	// }

	SP sp;
	UQ uq;
	Eigen::MatrixXd d;
	Rcpp::List dlist;

	Eigen::MatrixXd R;
	Rcpp::List dR;

	if(form=="isotropic"){
		d = sp.pdist(input, input, dtype);
		R = sp.iso_kernel(d, range(0), tail(0), nu(0), family);
		dR = sp.deriv_iso_kernel(d, range(0), tail(0), nu(0), family);
	}else if(form=="tensor"){
		dlist = uq.adist(input, input);
		R = uq.tensor_kernel(dlist, range, tail, nu, family);
		dR = uq.deriv_tensor_kernel(dlist, range, tail, nu, family);
	}else if(form=="ARD"){
		dlist = uq.adist(input, input);
		R = uq.ARD_kernel(dlist, range, tail(0), nu(0), family);
		dR = uq.deriv_ARD_kernel(dlist, range, tail(0), nu(0), family);
	}else{
    	Rcpp::stop("The specified form of covariance functions is not supported yet.\n");
  	} 

	// SEXP d = distance(input, input, form, dtype);
	// Eigen::MatrixXd R = kernel(d, range, tail, nu, covmodel);
	// Rcpp::List dR = deriv_kernel(d, range, tail, nu, covmodel);

	R.diagonal().array() += nugget;
	int n = R.rows();
	Eigen::MatrixXd E = Eigen::MatrixXd::Identity(n,n);

	Eigen::MatrixXd RInv = R.llt().solve(E);

	int ncorpar = dR.size()-1;


	Rcpp::List U(ncorpar+3); // add derivative w.r.t. sig2, nugget, nu 
	U[0] = 1.0/sig2 * E; // w.r.t. sig2 
	for(int k=0; k<ncorpar; k++){
		U[k+1] = RInv*Rcpp::as<Eigen::MatrixXd>(dR[k]); 
	}


	Eigen::MatrixXd Ui(n,n), Uj(n,n);
	Eigen::MatrixXd I(ncorpar+3,ncorpar+3);
	I.setZero();

	if(family=="CH"){
		U[ncorpar+1] = RInv*Rcpp::as<Eigen::MatrixXd>(dR[ncorpar]); // w.r.t. nu 
		U[ncorpar+2] = RInv; // w.r.t. nugget

		for(int i=0; i<ncorpar+3; i++){
			Ui = Rcpp::as<Eigen::MatrixXd>(U[i]);
			I(i,i) = 0.5 * (Ui * Ui).diagonal().sum();
			for(int j=0; j<i; j++){
				Uj = Rcpp::as<Eigen::MatrixXd>(U[j]);
				I(i,j) = 0.5 * (Ui * Uj).diagonal().sum();
				I(j,i) = I(i,j);
			}
		}
	}else if(family=="matern"){
		U[ncorpar+1] = RInv; // w.r.t. nugget
		I.resize(ncorpar+2, ncorpar+2);
		for(int i=0; i<ncorpar+2; i++){
			Ui = Rcpp::as<Eigen::MatrixXd>(U[i]);
			I(i,i) = 0.5 * (Ui * Ui).diagonal().sum();
			for(int j=0; j<i; j++){
				Uj = Rcpp::as<Eigen::MatrixXd>(U[j]);
				I(i,j) = 0.5 * (Ui * Uj).diagonal().sum();
				I(j,i) = I(i,j);
			}
		}		
	}else{
		Rcpp::stop("Fisher information matrix for the specified correlation function is not supported.\n");
	}

	return I;
}

//[[Rcpp::export]]
Eigen::MatrixXd FisherIR_intlik(const Eigen::MatrixXd& H, const Eigen::MatrixXd& input, 
	const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
	const Eigen::VectorXd& nu, const double& nugget, 
	const Rcpp::List& covmodel, const std::string& dtype){

  	std::string family = Rcpp::as<std::string>(covmodel["family"]);
  	std::string form = Rcpp::as<std::string>(covmodel["form"]);

	// Eigen::VectorXd range;
	// if(par.containsElementNamed("range")){
	// 	range = Rcpp::as<Eigen::VectorXd>(par["range"]);
	// }else{
	// 	Rcpp::stop("reference_prior: No range parameter is specified.\n");
	// }

	// Eigen::VectorXd tail;
	// if(par.containsElementNamed("tail")){
	// 	tail = Rcpp::as<Eigen::VectorXd>(par["tail"]);
	// }else{
	// 	tail = 0.5*Eigen::VectorXd::Ones(1); 
	// }

	// Eigen::VectorXd nu;
	// if(par.containsElementNamed("nu")){
	// 	nu = Rcpp::as<Eigen::VectorXd>(par["nu"]);
	// }else{
	// 	nu = 0.5*Eigen::VectorXd::Ones(1);
	// }
	
	// double nugget = 0;
	// if(par.containsElementNamed("nugget")){
	// 	nugget = Rcpp::as<double>(par["nugget"]);
	// }


	int n=H.rows();
	int p=H.cols();
	
	Eigen::MatrixXd R(n,n), RInv(n,n), Q(n,n), RH(n,p), HRH(p,p);
	Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH;
	
	// SEXP d = distance(input, input, form, dtype);

  Rcpp::List dR, dlist;
  UQ uq;
  SP sp;

  if(form=="tensor"){
  	// dlist = Rcpp::as<Rcpp::List>(d);
  	dlist = uq.adist(input, input);
    R = uq.tensor_kernel(dlist, range, tail, nu, family);
    dR = uq.deriv_tensor_kernel(dlist, range, tail, nu, family);
  }else if(form=="ARD"){
  	// dlist = Rcpp::as<Rcpp::List>(d);
  	dlist = uq.adist(input, input);
    R = uq.ARD_kernel(dlist, range, tail(0), nu(0), family);
    dR = uq.deriv_ARD_kernel(dlist, range, tail(0), nu(0), family);
  }else if(form=="isotropic"){
  	// Eigen::MatrixXd dmat = Rcpp::as<Eigen::MatrixXd>(d);
  	Eigen::MatrixXd dmat = sp.pdist(input, input, dtype);
  	R = sp.iso_kernel(dmat, range(0), tail(0), nu(0), family);
  	dR = sp.deriv_iso_kernel(dmat, range(0), tail(0), nu(0), family);
  }else{
    Rcpp::stop("The specified form of covariance functions is not supported yet.\n");
  } 
  int npars = dR.size()-1;

  R.diagonal().array() += nugget;

  ldltR.compute(R);
  RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
  
  RH = RInv * H;
  HRH = H.transpose() * RH;
  ldltH.compute(HRH);
  Q = RInv - RH*ldltH.solve(RH.transpose());

  Rcpp::List W(npars+2);
  Eigen::MatrixXd W_l(n, n), W_k(n, n);

  for(int k=0; k<npars; k++){
    W[k] = Rcpp::as<Eigen::MatrixXd>(dR[k])*Q;
  }


  int count;

  Eigen::MatrixXd FisherIR(npars+3, npars+3);

	if(family=="CH"){

	    W[npars] = Rcpp::as<Eigen::MatrixXd>(dR[npars])*Q;  // corresponding to smoothness parameter
		W[npars+1] = Q; // corresponding to nugget
	    FisherIR(0,0) = n - p;
	    for(int l=0; l<(npars+2); l++){
	      W_l = Rcpp::as<Eigen::MatrixXd>(W[l]);

	      FisherIR(0, l+1) = W_l.trace();
	      FisherIR(l+1, 0) = FisherIR(0, l+1);

	      for(int k=0; k<(npars+2); k++){
	        W_k = Rcpp::as<Eigen::MatrixXd>(W[k]);
	        FisherIR(l+1, k+1) = (W_l*W_k).trace();
	        FisherIR(k+1, l+1) = FisherIR(l+1, k+1);
	      }
	    }

	    count = npars + 3;    

	}else{
		W[npars] = Q; // corresponding to nugget

	    FisherIR(0,0) = n - p;
	    for(int l=0; l<(npars+1); l++){
	      W_l = Rcpp::as<Eigen::MatrixXd>(W[l]);
	      FisherIR(0, l+1) = W_l.trace();
	      FisherIR(l+1, 0) = FisherIR(0, l+1);

	      for(int k=0; k<(npars+1); k++){
	        W_k = Rcpp::as<Eigen::MatrixXd>(W[k]);
	        FisherIR(l+1, k+1) = (W_l*W_k).trace();
	        FisherIR(k+1, l+1) = FisherIR(l+1, k+1);
	      }

	    }  

	    count = npars + 2;
	}


	return FisherIR.block(0,0,count,count);
} 

// [[Rcpp::export]]
Eigen::MatrixXd GPsim(Eigen::Map<Eigen::MatrixXd> input, Eigen::Map<Eigen::MatrixXd> H, const Rcpp::List& par, const Rcpp::List& covmodel,
	int nsample=1, std::string dtype="Euclidean"){

	std::string family = Rcpp::as<std::string>(covmodel["family"]);
	std::string form = Rcpp::as<std::string>(covmodel["form"]);

	int p = H.cols();
	Eigen::VectorXd coeff = Eigen::VectorXd::Zero(p);
	if(par.containsElementNamed("coeff")){
		coeff = Rcpp::as<Eigen::VectorXd>(par["coeff"]);
	}
	double sig2 = 1;
	if(par.containsElementNamed("sig2")){
		sig2 = Rcpp::as<double>(par["sig2"]);
	}

	Eigen::VectorXd range;
	if(par.containsElementNamed("range")){
		range = Rcpp::as<Eigen::VectorXd>(par["range"]);
	}else{
		Rcpp::stop("No range parameter value is specified.\n");
	}

	Eigen::VectorXd tail;
	if(par.containsElementNamed("tail")){
		tail = Rcpp::as<Eigen::VectorXd>(par["tail"]);
	}else{
		tail = Eigen::VectorXd::Ones(1); 
	}

	Eigen::VectorXd nu;
	if(par.containsElementNamed("nu")){
		nu = Rcpp::as<Eigen::VectorXd>(par["nu"]);
	}else{
		nu = 0.5*Eigen::VectorXd::Ones(1);
	}
	
	double nugget = 0;
	if(par.containsElementNamed("nugget")){
		nugget = Rcpp::as<double>(par["nugget"]);
	}
	
	UQ uq;
	Eigen::MatrixXd sim;

	if(form=="tensor"||form=="ARD"){
		sim = uq.simulate(input, H, coeff, sig2, range, tail, nu, nugget, covmodel, nsample);
	}else if(form=="isotropic"){
	  SP sp;
		sim = sp.simulate(input, H, coeff, sig2, range(0), tail(0), nu(0), nugget, covmodel, nsample, dtype);
	}else{
		Rcpp::stop("The form of covariance kernels is not implemented yet.\n");
	}

	return sim;

}


// [[Rcpp::export]]
Rcpp::List GPpredict(Eigen::Map<Eigen::MatrixXd> output, Eigen::Map<Eigen::MatrixXd> H, 
  Eigen::Map<Eigen::MatrixXd> input, Eigen::Map<Eigen::MatrixXd> input_new, 
  Eigen::Map<Eigen::MatrixXd> Hnew, const Rcpp::List& par, const Rcpp::List& covmodel, 
  const std::string& dtype){

	std::string form = Rcpp::as<std::string>(covmodel["form"]);

	Eigen::VectorXd range;
	if(par.containsElementNamed("range")){
		if(Rf_isNumeric(par["range"])){
			range = Rcpp::as<Eigen::VectorXd>(par["range"]);
		}else{
			Rcpp::stop("The range parameter is neither a numerical vector nor a scalar.\n");
		}
	}else{
		Rcpp::stop("The range parameter value is not specified in the list.\n");
	}

	Eigen::VectorXd tail;
	if(par.containsElementNamed("tail")){
		if(Rf_isNumeric(par["tail"])){
			tail = Rcpp::as<Eigen::VectorXd>(par["tail"]);
		}else{
			Rcpp::stop("The tail parameter is neither a numerical vector nor a scalar.\n");
		}
		
	}else{
		tail = Eigen::VectorXd::Ones(1); 
	}

	Eigen::VectorXd nu;
	if(par.containsElementNamed("nu")){
		if(Rf_isNumeric(par["nu"])){
			nu = Rcpp::as<Eigen::VectorXd>(par["nu"]);
		}else{
			Rcpp::stop("The nu parameter is neither a numerical vector nor a scalar.\n");
		}
	}else{
		Rcpp::stop("The smoothness parameter is not specified in the list.\n");
	}
	
	double nugget = 0;
	if(par.containsElementNamed("nugget")){
		if(Rf_isNumeric(par["nugget"])){
			nugget = Rcpp::as<double>(par["nugget"]);
		}else{
			Rcpp::stop("The nugget parameter is neither a numerical vector nor a scalar.\n");
		}
	}else{
		Rcpp::stop("The nugget parameter is not specified in the list.\n");		
	}


	UQ uq;

	Rcpp::List pred;

	if(form=="tensor"||form=="ARD"){
		pred = uq.predict(output, H, input, input_new, Hnew, range, tail, nu, 
			nugget, covmodel);
	}else if(form=="isotropic"){
		SP sp;
		pred = sp.predict(output, H, input, input_new, Hnew, range(0), tail(0), nu(0),
			nugget, covmodel, dtype);
	}else{
		Rcpp::stop("The form of covariance kernels is not implemented yet.\n");
	}

	return pred;

}


// [[Rcpp::export]]
Rcpp::List post_predictive_sampling(Eigen::Map<Eigen::MatrixXd> output, Eigen::Map<Eigen::MatrixXd> H, 
  Eigen::Map<Eigen::MatrixXd> input, Eigen::Map<Eigen::MatrixXd> input_new, 
  Eigen::Map<Eigen::MatrixXd> Hnew, const Rcpp::List& MCMCsample, const double& smooth, bool& smoothness_est, const Rcpp::List& covmodel, 
  const std::string& dtype){

	std::string form = Rcpp::as<std::string>(covmodel["form"]);
	//int Dim = input.cols();
	
	int nsample; 
	Eigen::MatrixXd range;
	if(Rf_isMatrix(MCMCsample["range"])){
		// Rcpp::Rcout<<"range is a matrix\n";
		range = Rcpp::as<Eigen::MatrixXd>(MCMCsample["range"]);
		nsample = range.rows();
	}else if(Rf_isNumeric(MCMCsample["range"])){
		// Rcpp::Rcout<<"range is a numerical vector\n";
		Eigen::VectorXd range_sample = Rcpp::as<Eigen::VectorXd>(MCMCsample["range"]);
		nsample = range_sample.size();
		range.resize(nsample,1);
		range.col(0) = range_sample;
		// range = Rcpp::as<Eigen::VectorXd>(MCMCsample["range"]);		
	}else{
		Rcpp::stop("range is neither a matrix nor a numerical vector in MCMC samples.\n");
	}


	Eigen::MatrixXd tail;
	if(MCMCsample.containsElementNamed("tail")){
		if(Rf_isMatrix(MCMCsample["tail"])){
			// Rcpp::Rcout<<"tail is a matrix\n";
			tail = Rcpp::as<Eigen::VectorXd>(MCMCsample["tail"]);
		}else if(Rf_isNumeric(MCMCsample["tail"])){
			// Rcpp::Rcout<<"tail is a numerical vector\n";
			Eigen::VectorXd tail_sample = Rcpp::as<Eigen::VectorXd>(MCMCsample["tail"]);
			tail.resize(nsample,1);
			tail.col(0) = tail_sample;
			// tail = Rcpp::as<Eigen::VectorXd>(MCMCsample["tail"]);		
		}else{
			Rcpp::stop("tail is neither a matrix nor a numerical vector in MCMC samples.\n");
		}
	}else{ // covariance without tail such as the matern class is used, 
		tail = 0.5*Eigen::MatrixXd::Ones(nsample,1);
	}

	Eigen::MatrixXd nu;
	if(smoothness_est){
		if(MCMCsample.containsElementNamed("nu")){
			if(Rf_isMatrix(MCMCsample["nu"])){
				// Rcpp::Rcout<<"nu is a matrix\n";
				nu = Rcpp::as<Eigen::MatrixXd>(MCMCsample["nu"]);
			}else if(Rf_isNumeric(MCMCsample["nu"])){
				// Rcpp::Rcout<<"nu is a numerical vector\n";
				Eigen::VectorXd nu_sample = Rcpp::as<Eigen::VectorXd>(MCMCsample["nu"]);
				nu.resize(nsample, 1);
				nu.col(0) = nu_sample;
			}else{
				Rcpp::stop("The nu is specified as neither a vector nor a matrix.\n");
			}
		}else{
			Rcpp::stop("nu is not in MCMC samples, but it was requested being estimated.\n");
		}
	}else{
		nu = smooth * Eigen::MatrixXd::Ones(nsample, 1);
	}


	Eigen::VectorXd nugget;
	if(Rf_isNumeric(MCMCsample["nugget"])){
		// Rcpp::Rcout<<"nugget is a numerical vector\n";
		nugget = Rcpp::as<Eigen::VectorXd>(MCMCsample["nugget"]);
	}else{
		Rcpp::stop("The nugget is not specified as a vector in MCMC samples.\n"); 		
	}



	UQ uq;

	Rcpp::List pred;

	if(form=="isotropic"){
		SP sp;
		pred = sp.simulate_predictive_dist(output, H, input, input_new, Hnew, range.col(0), tail.col(0), nu.col(0),
	        nugget, covmodel, dtype);
	}else if(form=="tensor"){
		pred = uq.tensor_simulate_predictive_dist(output, H, input, input_new, 
				   Hnew, range, tail, nu, nugget, covmodel);

	}else if(form=="ARD"){
		pred = uq.ARD_simulate_predictive_dist(output, H, input, input_new, 
				   Hnew, range, tail.col(0), nu.col(0), nugget, covmodel);

	}


	return pred;

}

/**************************************************************************************/
// [[Rcpp::export]]
Rcpp::List MCMCOBayes(Eigen::Map<Eigen::MatrixXd> output, Eigen::Map<Eigen::MatrixXd> H,  
	Eigen::Map<Eigen::MatrixXd> input, const Rcpp::List& par, const Rcpp::List& covmodel, 
	const bool& smoothness_est, const Rcpp::List& proposal, const int& nsample, 
	const std::string& dtype,  bool verbose=true){

	std::string family = Rcpp::as<std::string>(covmodel["family"]);
	std::string form = Rcpp::as<std::string>(covmodel["form"]);

	Eigen::VectorXd range;
	if(par.containsElementNamed("range")){
		range = Rcpp::as<Eigen::VectorXd>(par["range"]);
	}else{
		Rcpp::stop("No range parameter value is specified.\n");
	}

	Eigen::VectorXd tail;
	if(par.containsElementNamed("tail")){
		tail = Rcpp::as<Eigen::VectorXd>(par["tail"]);
	}else{
		tail = Eigen::VectorXd::Ones(1); 
	}

	Eigen::VectorXd nu;
	if(par.containsElementNamed("nu")){
		nu = Rcpp::as<Eigen::VectorXd>(par["nu"]);
	}else{
		nu = 0.5*Eigen::VectorXd::Ones(1);
	}
	
	double nugget = 0;
	if(par.containsElementNamed("nugget")){
		nugget = Rcpp::as<double>(par["nugget"]);
	}


	UQ uq; 
	SP sp;
	Rcpp::List MCMCsample;
	std::string dist = "Normal"; // RW as proposal by default 
	if(proposal.containsElementNamed("dist")){
		dist = Rcpp::as<std::string>(proposal["dist"]);
	}
	


	if(dist=="Normal"){

		if(form=="tensor"){
			MCMCsample = uq.tensor_MCMCOBayes(output, H, input, range, tail, nu,  
			  	nugget, covmodel, smoothness_est, proposal, nsample, verbose);
		}else if(form=="ARD"){
			MCMCsample = uq.ARD_MCMCOBayes(output, H, input, range, tail, nu, 
			  	nugget, covmodel, smoothness_est, proposal, nsample, verbose);		
		}else if(form=="isotropic"){
			MCMCsample = sp.iso_MCMCOBayes(output, H, input, range(0), tail(0), nu(0),
			  	 nugget, covmodel, smoothness_est, proposal, nsample, dtype, verbose); 
		}else{
			Rcpp::stop("The MCMC algorithm for the covariance kernels is not implemented yet.\n");
		}			
		

		
	}else{
		Rcpp::stop("This proposal distribution is not used in current MCMC algorithms.\n");
	}



return MCMCsample;

}
/**************************************************************************************/

/**************************************************************************************/
// [[Rcpp::export]]
Rcpp::List MCMCOBayes_pred(Eigen::Map<Eigen::MatrixXd> output, 
  	Eigen::Map<Eigen::MatrixXd> H,  Eigen::Map<Eigen::MatrixXd> input, 
	Eigen::Map<Eigen::MatrixXd> input_new, Eigen::Map<Eigen::MatrixXd> Hnew,
	const Rcpp::List& par, Rcpp::List& covmodel, 
	const bool& smoothness_est, const Rcpp::List& proposal, const int& nsample, 
	const std::string& dtype, const bool& verbose){

	std::string family = Rcpp::as<std::string>(covmodel["family"]);
	std::string form = Rcpp::as<std::string>(covmodel["form"]);

	Eigen::VectorXd range;
	if(par.containsElementNamed("range")){
		range = Rcpp::as<Eigen::VectorXd>(par["range"]);
	}else{
		Rcpp::stop("No range parameter value is specified.\n");
	}

	Eigen::VectorXd tail;
	if(par.containsElementNamed("tail")){
		tail = Rcpp::as<Eigen::VectorXd>(par["tail"]);
	}else{
		tail = Eigen::VectorXd::Ones(1); 
	}

	Eigen::VectorXd nu;
	if(par.containsElementNamed("nu")){
		nu = Rcpp::as<Eigen::VectorXd>(par["nu"]);
	}else{
		nu = 0.5*Eigen::VectorXd::Ones(1);
	}
	
	double nugget = 0;
	if(par.containsElementNamed("nugget")){
		nugget = Rcpp::as<double>(par["nugget"]);
	}


	UQ uq; 
	SP sp;
	Rcpp::List MCMCsample;
	std::string dist = "Normal"; // RW as proposal by default 
	if(proposal.containsElementNamed("dist")){
		dist = Rcpp::as<std::string>(proposal["dist"]);
	}
	


	if(dist=="Normal"){
		if(form=="tensor"){
			MCMCsample = uq.tensor_MCMCOBayes(output, H, input, range, tail, nu,  
			  	nugget, covmodel, smoothness_est, proposal, nsample, verbose,
			  	input_new, Hnew);
		}else if(form=="ARD"){
			MCMCsample = uq.ARD_MCMCOBayes(output, H, input, range, tail, nu, 
			  	nugget, covmodel, smoothness_est, proposal, nsample, verbose,
			  	input_new, Hnew);		
		}else if(form=="isotropic"){
			MCMCsample = sp.iso_MCMCOBayes(output, H, input, range(0), tail(0), nu(0),
			  	 nugget, covmodel, smoothness_est, proposal, nsample, dtype, verbose,
			  	 input_new, Hnew); 
		}else{
			Rcpp::stop("The MCMC algorithm for the covariance kernels is not implemented yet.\n");
		}
	// else if(dist=="LogNormal"){
	// 	if(form=="tensor"){
	// 		MCMCsample = uq.tensor_MCMC_LN(output, H, input, par, covmodel, 
	// 		  	nugget_est, proposal, nsample, verbose);
	// 	}else if(form=="ARD"){
	// 		MCMCsample = uq.ARD_MCMC_LN(output, H, input, par, covmodel, 
	// 		  	nugget_est, proposal, nsample, verbose);		
	// 	}else if(form=="isotropic"){
	// 		MCMCsample = sp.iso_RWMCMC(output, H, input, par, covmodel, 
	// 		  	nugget_est, proposal, nsample, dtype, verbose); 
	// 	}else{
	// 		Rcpp::stop("The MCMC algorithm for the covariance kernels is not implemented yet.\n");
	// 	}		
	}else{
		Rcpp::stop("This proposal distribution is not used in current MCMC algorithms.\n");
	}

return MCMCsample;

}
/**************************************************************************************/

/**************************************************************************************/
// [[Rcpp::export]]
Rcpp::List MCMCOBayesRef(Eigen::Map<Eigen::MatrixXd> output, Eigen::Map<Eigen::MatrixXd> H,  
	Eigen::Map<Eigen::MatrixXd> input, const Rcpp::List& par, const Rcpp::List& covmodel, 
	const bool& smoothness_est, const Rcpp::List& proposal, const int& nsample, 
	const std::string& dtype, const bool& verbose){

	std::string family = Rcpp::as<std::string>(covmodel["family"]);
	std::string form = Rcpp::as<std::string>(covmodel["form"]);

	Eigen::VectorXd range;
	if(par.containsElementNamed("range")){
		range = Rcpp::as<Eigen::VectorXd>(par["range"]);
	}else{
		Rcpp::stop("No range parameter value is specified.\n");
	}

	Eigen::VectorXd tail;
	if(par.containsElementNamed("tail")){
		tail = Rcpp::as<Eigen::VectorXd>(par["tail"]);
	}else{
		tail = Eigen::VectorXd::Ones(1); 
	}

	Eigen::VectorXd nu;
	if(par.containsElementNamed("nu")){
		nu = Rcpp::as<Eigen::VectorXd>(par["nu"]);
	}else{
		nu = 0.5*Eigen::VectorXd::Ones(1);
	}
	
	double nugget = 0;
	if(par.containsElementNamed("nugget")){
		nugget = Rcpp::as<double>(par["nugget"]);
	}


	UQ uq; 
	SP sp;
	Rcpp::List MCMCsample;
	std::string dist = "Normal"; // RW as proposal by default 
	if(proposal.containsElementNamed("dist")){
		dist = Rcpp::as<std::string>(proposal["dist"]);
	}
	


	if(dist=="Normal"){

		if(form=="tensor"){
			MCMCsample = uq.tensor_MCMCOBayes(output, H, input, range, tail, nu,  
			  	nugget, covmodel, smoothness_est, proposal, nsample, verbose);
		}else if(form=="ARD"){
			MCMCsample = uq.ARD_MCMCOBayes(output, H, input, range, tail, nu, 
			  	nugget, covmodel, smoothness_est, proposal, nsample, verbose);		
		}else if(form=="isotropic"){
			MCMCsample = sp.iso_MCMCOBayes_Ref(output, H, input, range(0), tail(0), nu(0),
			  	 nugget, covmodel, smoothness_est, proposal, nsample, dtype, verbose); 
		}else{
			Rcpp::stop("The MCMC algorithm for the covariance kernels is not implemented yet.\n");
		}			
		

		
	}else{
		Rcpp::stop("This proposal distribution is not used in current MCMC algorithms.\n");
	}



return MCMCsample;

}
/**************************************************************************************/

/**************************************************************************************/
// [[Rcpp::export]]
Rcpp::List MCMCOBayesRef_pred(Eigen::Map<Eigen::MatrixXd> output, 
  	Eigen::Map<Eigen::MatrixXd> H,  Eigen::Map<Eigen::MatrixXd> input, 
	Eigen::Map<Eigen::MatrixXd> input_new, Eigen::Map<Eigen::MatrixXd> Hnew,
	const Rcpp::List& par, const Rcpp::List& covmodel, 
	const bool& smoothness_est, const Rcpp::List& proposal, const int& nsample, 
	const std::string& dtype, const bool& verbose){

	std::string family = Rcpp::as<std::string>(covmodel["family"]);
	std::string form = Rcpp::as<std::string>(covmodel["form"]);

	Eigen::VectorXd range;
	if(par.containsElementNamed("range")){
		range = Rcpp::as<Eigen::VectorXd>(par["range"]);
	}else{
		Rcpp::stop("No range parameter value is specified.\n");
	}

	Eigen::VectorXd tail;
	if(par.containsElementNamed("tail")){
		tail = Rcpp::as<Eigen::VectorXd>(par["tail"]);
	}else{
		tail = Eigen::VectorXd::Ones(1); 
	}

	Eigen::VectorXd nu;
	if(par.containsElementNamed("nu")){
		nu = Rcpp::as<Eigen::VectorXd>(par["nu"]);
	}else{
		nu = 0.5*Eigen::VectorXd::Ones(1);
	}
	
	double nugget = 0;
	if(par.containsElementNamed("nugget")){
		nugget = Rcpp::as<double>(par["nugget"]);
	}


	UQ uq; 
	SP sp;
	Rcpp::List MCMCsample;
	std::string dist = "Normal"; // RW as proposal by default 
	if(proposal.containsElementNamed("dist")){
		dist = Rcpp::as<std::string>(proposal["dist"]);
	}
	


	if(dist=="Normal"){
		if(form=="tensor"){
			MCMCsample = uq.tensor_MCMCOBayes(output, H, input, range, tail, nu,  
			  	nugget, covmodel, smoothness_est, proposal, nsample, verbose,
			  	input_new, Hnew);
		}else if(form=="ARD"){
			MCMCsample = uq.ARD_MCMCOBayes(output, H, input, range, tail, nu, 
			  	nugget, covmodel, smoothness_est, proposal, nsample, verbose,
			  	input_new, Hnew);		
		}else if(form=="isotropic"){
			MCMCsample = sp.iso_MCMCOBayes_Ref(output, H, input, range(0), tail(0), nu(0),
			  	 nugget, covmodel, smoothness_est, proposal, nsample, dtype, verbose,
			  	 input_new, Hnew); 
		}else{
			Rcpp::stop("The MCMC algorithm for the covariance kernels is not implemented yet.\n");
		}
	// else if(dist=="LogNormal"){
	// 	if(form=="tensor"){
	// 		MCMCsample = uq.tensor_MCMC_LN(output, H, input, par, covmodel, 
	// 		  	nugget_est, proposal, nsample, verbose);
	// 	}else if(form=="ARD"){
	// 		MCMCsample = uq.ARD_MCMC_LN(output, H, input, par, covmodel, 
	// 		  	nugget_est, proposal, nsample, verbose);		
	// 	}else if(form=="isotropic"){
	// 		MCMCsample = sp.iso_RWMCMC(output, H, input, par, covmodel, 
	// 		  	nugget_est, proposal, nsample, dtype, verbose); 
	// 	}else{
	// 		Rcpp::stop("The MCMC algorithm for the covariance kernels is not implemented yet.\n");
	// 	}		
	}else{
		Rcpp::stop("This proposal distribution is not used in current MCMC algorithms.\n");
	}

return MCMCsample;

}
/**************************************************************************************/


/**************************************************************************************/
// [[Rcpp::export]]
Rcpp::List MCMCSBayes(Eigen::Map<Eigen::MatrixXd> output, Eigen::Map<Eigen::MatrixXd> H,  
	Eigen::Map<Eigen::MatrixXd> input, const Rcpp::List& par, const Rcpp::List& covmodel, 
	const bool& smoothness_est, const Rcpp::List& prior, const Rcpp::List& proposal, const int& nsample, 
	const std::string& dtype, const bool& verbose){

	std::string family = Rcpp::as<std::string>(covmodel["family"]);
	std::string form = Rcpp::as<std::string>(covmodel["form"]);

	Eigen::VectorXd range;
	if(par.containsElementNamed("range")){
		range = Rcpp::as<Eigen::VectorXd>(par["range"]);
	}else{
		Rcpp::stop("No range parameter value is specified.\n");
	}

	Eigen::VectorXd tail;
	if(par.containsElementNamed("tail")){
		tail = Rcpp::as<Eigen::VectorXd>(par["tail"]);
	}else{
		tail = Eigen::VectorXd::Ones(1); 
	}

	Eigen::VectorXd nu;
	if(par.containsElementNamed("nu")){
		nu = Rcpp::as<Eigen::VectorXd>(par["nu"]);
	}else{
		nu = 0.5*Eigen::VectorXd::Ones(1);
	}
	
	double nugget = 0;
	if(par.containsElementNamed("nugget")){
		nugget = Rcpp::as<double>(par["nugget"]);
	}


	UQ uq; 
	SP sp;
	Rcpp::List MCMCsample;
	std::string dist = "Normal"; // RW as proposal by default 
	if(proposal.containsElementNamed("dist")){
		dist = Rcpp::as<std::string>(proposal["dist"]);
	}
	


	if(dist=="Normal"){
		if(form=="tensor"){
			MCMCsample = uq.tensor_MCMCSBayes(output, H, input, range, tail, nu,  
			  	nugget, covmodel, smoothness_est, prior, proposal, nsample, verbose);
		}else if(form=="ARD"){
			MCMCsample = uq.ARD_MCMCSBayes(output, H, input, range, tail, nu, 
			  	nugget, covmodel, smoothness_est, prior, proposal, nsample, verbose);		
		}else if(form=="isotropic"){
			MCMCsample = sp.iso_MCMCSBayes(output, H, input, range(0), tail(0), nu(0),
			  	 nugget, covmodel, smoothness_est, prior, proposal, nsample, dtype, verbose); 
		}else{
			Rcpp::stop("The MCMC algorithm for the covariance kernels is not implemented yet.\n");
		}
	// else if(dist=="LogNormal"){
	// 	if(form=="tensor"){
	// 		MCMCsample = uq.tensor_MCMC_LN(output, H, input, par, covmodel, 
	// 		  	nugget_est, proposal, nsample, verbose);
	// 	}else if(form=="ARD"){
	// 		MCMCsample = uq.ARD_MCMC_LN(output, H, input, par, covmodel, 
	// 		  	nugget_est, proposal, nsample, verbose);		
	// 	}else if(form=="isotropic"){
	// 		MCMCsample = sp.iso_RWMCMC(output, H, input, par, covmodel, 
	// 		  	nugget_est, proposal, nsample, dtype, verbose); 
	// 	}else{
	// 		Rcpp::stop("The MCMC algorithm for the covariance kernels is not implemented yet.\n");
	// 	}		
	}else{
		Rcpp::stop("This proposal distribution is not used in current MCMC algorithm.\n");
	}

return MCMCsample;

}

/******************************************************************************/

/**************************************************************************************/
// [[Rcpp::export]]
Rcpp::List MCMCSBayes_pred(Eigen::Map<Eigen::MatrixXd> output, Eigen::Map<Eigen::MatrixXd> H,  
	Eigen::Map<Eigen::MatrixXd> input, 	Eigen::Map<Eigen::MatrixXd> input_new, 
	Eigen::Map<Eigen::MatrixXd> Hnew, const Rcpp::List& par, const Rcpp::List& covmodel, 
	const bool& smoothness_est, const Rcpp::List& prior, const Rcpp::List& proposal, const int& nsample, 
	const std::string& dtype, const bool& verbose){

	std::string family = Rcpp::as<std::string>(covmodel["family"]);
	std::string form = Rcpp::as<std::string>(covmodel["form"]);

	Eigen::VectorXd range;
	if(par.containsElementNamed("range")){
		range = Rcpp::as<Eigen::VectorXd>(par["range"]);
	}else{
		Rcpp::stop("No range parameter value is specified.\n");
	}

	Eigen::VectorXd tail;
	if(par.containsElementNamed("tail")){
		tail = Rcpp::as<Eigen::VectorXd>(par["tail"]);
	}else{
		tail = Eigen::VectorXd::Ones(1); 
	}

	Eigen::VectorXd nu;
	if(par.containsElementNamed("nu")){
		nu = Rcpp::as<Eigen::VectorXd>(par["nu"]);
	}else{
		nu = 0.5*Eigen::VectorXd::Ones(1);
	}
	
	double nugget = 0;
	if(par.containsElementNamed("nugget")){
		nugget = Rcpp::as<double>(par["nugget"]);
	}


	UQ uq; 
	SP sp;
	Rcpp::List MCMCsample;
	std::string dist = "Normal"; // RW as proposal by default 
	if(proposal.containsElementNamed("dist")){
		dist = Rcpp::as<std::string>(proposal["dist"]);
	}
	


	if(dist=="Normal"){
		if(form=="tensor"){
			MCMCsample = uq.tensor_MCMCSBayes(output, H, input, range, tail, nu,  
			  	nugget, covmodel, smoothness_est, prior, proposal, nsample, verbose,
			  	input_new, Hnew);
		}else if(form=="ARD"){
			MCMCsample = uq.ARD_MCMCSBayes(output, H, input, range, tail, nu, 
			  	nugget, covmodel, smoothness_est, prior, proposal, nsample, verbose,
			  	input_new, Hnew);		
		}else if(form=="isotropic"){
			MCMCsample = sp.iso_MCMCSBayes(output, H, input, range(0), tail(0), nu(0),
			  	 nugget, covmodel, smoothness_est, prior, proposal, nsample, dtype, verbose,
			  	 input_new, Hnew); 
		}else{
			Rcpp::stop("The MCMC algorithm for the covariance kernels is not implemented yet.\n");
		}
	// else if(dist=="LogNormal"){
	// 	if(form=="tensor"){
	// 		MCMCsample = uq.tensor_MCMC_LN(output, H, input, par, covmodel, 
	// 		  	nugget_est, proposal, nsample, verbose);
	// 	}else if(form=="ARD"){
	// 		MCMCsample = uq.ARD_MCMC_LN(output, H, input, par, covmodel, 
	// 		  	nugget_est, proposal, nsample, verbose);		
	// 	}else if(form=="isotropic"){
	// 		MCMCsample = sp.iso_RWMCMC(output, H, input, par, covmodel, 
	// 		  	nugget_est, proposal, nsample, dtype, verbose); 
	// 	}else{
	// 		Rcpp::stop("The MCMC algorithm for the covariance kernels is not implemented yet.\n");
	// 	}		
	}else{
		Rcpp::stop("This proposal distribution is not used in current MCMC algorithms.\n");
	}

return MCMCsample;

}

/******************************************************************************/


// [[Rcpp::export]]
double SPLoglik(const double& range, const double& tail, const double& nu, const double& nugget,
 const Eigen::MatrixXd& y, const Eigen::MatrixXd& H, const Eigen::MatrixXd& d, 
  const Rcpp::List& covmodel){

  std::string family = Rcpp::as<std::string>(covmodel["family"]);



  int n = y.rows();
  int q = y.cols();
  int p = H.cols();
  Eigen::MatrixXd R(n,n), RInv(n,n), Q(n,n), RH(n,p), HRH(p,p);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH;


  SP sp;
  R = sp.iso_kernel(d, range, tail, nu, family);

  R.diagonal().array() += nugget;

  Eigen::MatrixXd L = R.llt().matrixL();
  Eigen::MatrixXd Ly = L.triangularView<Eigen::Lower>().solve(y);
  Eigen::MatrixXd LH = L.triangularView<Eigen::Lower>().solve(H);
  Eigen::MatrixXd HRy = LH.transpose()*Ly;

  double lndetR = 2.0*L.diagonal().array().log().sum();
  // RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
  
  // RH = RInv * H;
  HRH = LH.transpose()*LH;
  // ldltH.compute(HRH);
  Eigen::MatrixXd HRHchol = HRH.llt().matrixL();
  Eigen::MatrixXd HRHInvHRy = HRHchol.triangularView<Eigen::Lower>().solve(HRy);
  double lndetHRH = 2.0*HRHchol.diagonal().array().log().sum();


  //Q = RInv - RH*ldltH.solve(RH.transpose());
  double log_S2;
  Eigen::MatrixXd S2 = Ly.transpose()*Ly - HRHInvHRy.transpose()*HRHInvHRy; 
  // Eigen::MatrixXd S2 = (output - H*bhat).transpose()*ldltR.solve(output - H*bhat);

  if(q==1){
    log_S2 = log(S2(0,0));
  }else{
    log_S2 = S2.ldlt().vectorD().array().log().sum();
  }


  double loglik = -0.5*q*lndetR -0.5*q*lndetHRH - 0.5*(n-p)*log_S2;

  return loglik;


}



// Test MCMC: 
// [[Rcpp::export]]
Rcpp::List MCMCtest(Eigen::Map<Eigen::MatrixXd> output, Eigen::Map<Eigen::MatrixXd> H, 
   Eigen::Map<Eigen::MatrixXd> input, const Rcpp::List& par, bool smoothness_est, 
   Rcpp::List& proposal, int nsample, std::string dtype, bool verbose){


  double range, tail, nu, nugget;
  if(par.containsElementNamed("range")){
  	range = Rcpp::as<double>(par["range"]);
  }else{
  	range = 1.0;
  }

  if(par.containsElementNamed("tail")){
  	tail = Rcpp::as<double>(par["tail"]);
  }else{
  	tail = 0.3;
  }

  if(par.containsElementNamed("nu")){
  	nu = Rcpp::as<double>(par["nu"]);
  }else{
  	nu=0.5;
  }

  if(par.containsElementNamed("nugget")){
  	nugget = Rcpp::as<double>(par["nugget"]);
  }else{
  	nugget = 0.0;
  }
  

  SP sp;

  Rcpp::List covmodel = Rcpp::List::create(Rcpp::_["family"]="matern",
                                          Rcpp::_["form"]="isotropic");

  Eigen::MatrixXd d = sp.pdist(input, input, dtype);


  double Delta_range =0.1;
  if(proposal.containsElementNamed("range")){
  	Delta_range = Rcpp::as<double>(proposal["range"]);
  }
  double Delta_nugget=0.1; // sd in the proposal distribution.

  double Delta_nu=0.1; // sd in the proposal distribution.


  if(proposal.containsElementNamed("nugget")){
    Delta_nugget = Rcpp::as<double>(proposal["nugget"]);
  }
  if(proposal.containsElementNamed("nu")){
    Delta_nu = Rcpp::as<double>(proposal["nu"]);
  }

  double loglik_curr = sp.MLoglik(range, tail, nu, nugget, output, H, d, covmodel);
  double loglik_prop=0.0;
  Eigen::VectorXd loglik(nsample);
  
  
  // initial values for parameters 
  double range_curr, tail_curr, nu_curr, nugget_curr;
  range_curr = range;
  tail_curr = tail;
  nu_curr = nu; 
  nugget_curr = nugget;

  Eigen::VectorXd range_sample(nsample);
  Eigen::VectorXd tail_sample(nsample);
  Eigen::VectorXd nu_sample(nsample);
  Eigen::VectorXd nugget_sample(nsample);

  double range_prop, nu_prop, nugget_prop;
  range_prop = range_curr;
  //tail_prop = tail_curr;
  nu_prop = nu_curr;
  nugget_prop = nugget_curr;

  // MCMC stuff
  double log_prior_curr, log_prior_prop, log_pr_tran, log_pr_rtran, MH_ratio;
  double unif_rnd;
  double accept_rate_range=0, accept_rate_nugget=0, 
    accept_rate_nu=0;
  double Jacobian_curr=0, Jacobian_prop=0;

  // create an empty list for abortion 
  // Rcpp::List abort_value = Rcpp::List::create(Rcpp::_["abort"]=R_NilValue);

  // gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937);
  
  Progress prog(nsample, verbose);
    
    // Rcpp::Rcout<<"inital range_curr = "<<range_curr<<"\n";
    // Rcpp::Rcout<<"********************Starting MCMC************************\n";

  #ifdef USE_R
    GetRNGstate();
  #endif
    // double range_temp; 
    // begin MCMC sampling 
    for(int it=0; it<nsample; it++){
      if(Progress::check_abort()){
        return R_NilValue;
      }
      prog.increment();

      loglik_curr = sp.MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // update range parameter 
      // generate proposal
      range_prop = exp(R::rnorm(log(range_curr), Delta_range));
      // range_prop = exp(-log(range_curr)-gsl_ran_gaussian(rng, Delta_range));
      // Rcpp::Rcout<<"range_prop["<<it+1<<"]"<<" = "<<range_prop<<"\n";


      loglik_prop = sp.MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
       // Rcpp::Rcout<<"loglik_prop["<<it+1<<"]"<<" = "<<loglik_prop<<"\n";

      // log prior density of cauchy dist
      log_prior_curr = -log(1.0+range_curr*range_curr);
      log_prior_prop = -log(1.0+range_prop*range_prop);
      // Rcpp::Rcout<<"log_prior_curr["<<it+1<<"]"<<" = "<<log_prior_curr<<"\n";
      // Rcpp::Rcout<<"log_prior_prop["<<it+1<<"]"<<" = "<<log_prior_prop<<"\n";


      // log proposal density 
      log_pr_tran = - (log(range_prop)-log(range_curr))*(log(range_prop)-log(range_curr)) / (2.0*Delta_range*Delta_range);
      log_pr_rtran = - (log(range_prop)-log(range_curr))*(log(range_prop)-log(range_curr)) / (2.0*Delta_range*Delta_range);
      // Rcpp::Rcout<<"log_pr_tran["<<it+1<<"]"<<" = "<<log_pr_tran<<"\n";
      // Rcpp::Rcout<<"log_pr_rtran["<<it+1<<"]"<<" = "<<log_pr_rtran<<"\n";

      // log_pr_tran = 0;
      // log_pr_rtran = 0;

      // Jacobian
      Jacobian_curr = log(range_curr);
      Jacobian_prop = log(range_prop);
      // Jacobian_curr = 0;
      // Jacobian_prop = 0;

      MH_ratio = (loglik_prop - loglik_curr) + (log_prior_prop-log_prior_curr) 
                     + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);
      // Rcpp::Rcout<<"MH_ratio["<<it+1<<"]"<<" = "<<MH_ratio<<"\n";

      unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
      // unif_rnd = gsl_rng_uniform(rng);
      // Rcpp::Rcout<<"unif_rnd["<<it+1<<"]"<<" = "<<log(unif_rnd)<<"\n";

      if(log(unif_rnd)<MH_ratio){ // accept 
        accept_rate_range += 1.0;
        range_curr = range_prop;
        loglik_curr = loglik_prop;
        // Rcpp::Rcout<<"acceptance at iteration = "<<it+1<<"\n";
      }

      range_sample(it) = range_curr;
      // Rcpp::Rcout<<"loglik_curr["<<it+1<<"]"<<" = "<<loglik_curr<<"\n";
      // Rcpp::Rcout<<"range_curr["<<it+1<<"]"<<" = "<<range_curr<<"\n";
      // Rcpp::Rcout<<"range_sample["<<it+1<<"]"<<" = "<<range_sample(it)<<"\n";
    	
      loglik(it) = loglik_curr;


	    loglik_curr = sp.MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
	    // generate proposal
	    nugget_prop = exp(Rcpp::rnorm(1, log(nugget_curr), Delta_nugget)[0]);
	    // nugget_prop = exp(log(nugget_curr) + gsl_ran_gaussian(rng, Delta_nugget));
	    // par_prop["nugget"] = nugget_prop;
	    // Rcpp::Rcout<<"nugget_prop["<<it+1<<"]"<<" = "<<nugget_prop<<"\n";
	    loglik_prop = sp.MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);
	    // Rcpp::Rcout<<"loglik_prop["<<it+1<<"]"<<" = "<<loglik_prop<<"\n";
	    // log prior density of cauchy dist
	    log_prior_curr = -log(1.0+nugget_curr*nugget_curr);
	    log_prior_prop = -log(1.0+nugget_prop*nugget_prop);

	    // log proposal density 
	    log_pr_tran = -(log(nugget_curr)-log(nugget_prop))*(log(nugget_curr)-log(nugget_prop)) / (2.0*Delta_nugget*Delta_nugget);
	    log_pr_rtran = -(log(nugget_curr)-log(nugget_prop))*(log(nugget_curr)-log(nugget_prop)) / (2.0*Delta_nugget*Delta_nugget);

	    // Jacobian
	    Jacobian_curr = log(nugget_curr);
	    Jacobian_prop = log(nugget_prop);

	    MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
	                  + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);
	    // Rcpp::Rcout<<"MH_ratio["<<it+1<<"]"<<" = "<<MH_ratio<<"\n";
	    unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
	    // unif_rnd = gsl_rng_uniform(rng);
	    // Rcpp::Rcout<<"unif_rnd["<<it+1<<"]"<<" = "<<log(unif_rnd)<<"\n";
	    if(log(unif_rnd)<MH_ratio){ // accept 
	      accept_rate_nugget +=1.0;
	      nugget_curr = nugget_prop;
	      loglik_curr = loglik_prop;
	    }

	    nugget_sample(it) = nugget_curr;
      

      if(smoothness_est){

        loglik_curr = sp.MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // generate proposal
        nu_prop = exp(Rcpp::rnorm(1, log(nu_curr), Delta_nu)[0]);
        loglik_prop = sp.MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

        // log prior density of cauchy dist
        log_prior_curr = -log(1.0+nu_curr*nu_curr);
        log_prior_prop = -log(1.0+nu_prop*nu_prop);

        // log proposal density 
        log_pr_tran = 0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(nu_curr);
        Jacobian_prop = log(nu_prop);

        MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_nu +=1.0;
          nu_curr = nu_prop;
          loglik_curr = loglik_prop;
        }

        nu_sample(it) = nu_curr;
      }

    }    

#ifdef USE_R
  PutRNGstate();
#endif

  // gsl_rng_free(rng);
/****************************************************************************/



    if(smoothness_est){
      return Rcpp::List::create(Rcpp::_["range"]=range_sample, 
                                Rcpp::_["nugget"]=nugget_sample,
                                Rcpp::_["nu"]=nu_sample,
                                Rcpp::_["loglik"]=loglik,
                                Rcpp::_["accept_rate_range"]=accept_rate_range/(nsample),
                                Rcpp::_["accept_rate_nugget"]=accept_rate_nugget/(nsample),
                                Rcpp::_["accept_rate_nu"]=accept_rate_nu/(nsample));      
    }else{
      return Rcpp::List::create(Rcpp::_["range"]=range_sample, 
      							Rcpp::_["nugget"]=nugget_sample,
      							Rcpp::_["loglik"]=loglik,
                                Rcpp::_["accept_rate_range"]=accept_rate_range/(nsample),
                                Rcpp::_["accept_rate_nugget"]=accept_rate_nugget/(nsample));
    }

}



// [[Rcpp::export]]
Rcpp::List MCMCOBayes_Ref(Eigen::Map<Eigen::MatrixXd> output, Eigen::Map<Eigen::MatrixXd> H, 
  Eigen::Map<Eigen::MatrixXd> input, const Rcpp::List& par, const Rcpp::List& covmodel,
  const bool& smoothness_est, const Rcpp::List& proposal, const int& nsample, const std::string& dtype, const bool& verbose){

  SP sp;

  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  double range, tail, nu, nugget;
  if(par.containsElementNamed("range")){
    range = Rcpp::as<double>(par["range"]);
  }else{
    range = 1.0;
  }

  if(par.containsElementNamed("tail")){
    tail = Rcpp::as<double>(par["tail"]);
  }else{
    tail = 0.3;
  }

  if(par.containsElementNamed("nu")){
    nu = Rcpp::as<double>(par["nu"]);
  }else{
    nu=0.5;
  }

  if(par.containsElementNamed("nugget")){
    nugget = Rcpp::as<double>(par["nugget"]);
  }else{
    nugget = 0.0;
  }

  Eigen::MatrixXd d = sp.pdist(input, input, dtype);

  double Delta_range;
  if(proposal.containsElementNamed("range")){
    Delta_range = Rcpp::as<double>(proposal["range"]);
  }else{
    Delta_range = 0.15;
  }
  double Delta_tail=0.4, Delta_nugget=0.1, Delta_nu=0.15; 

  if(proposal.containsElementNamed("tail")){
    Delta_tail = Rcpp::as<double>(proposal["tail"]);
  }

  if(proposal.containsElementNamed("nugget")){
    Delta_nugget = Rcpp::as<double>(proposal["nugget"]);
  }
  
  if(proposal.containsElementNamed("nu")){
    Delta_nu = Rcpp::as<double>(proposal["nu"]);
  }

  double loglik_curr = sp.MLoglik(range, tail, nu, nugget, output, H, d, covmodel);
  double loglik_prop=0.0;
  
  // initial values for parameters 
  double range_curr, tail_curr, nu_curr, nugget_curr;
  range_curr = range;
  tail_curr = tail;
  nu_curr = nu; 
  nugget_curr = nugget;

  Eigen::VectorXd range_sample(nsample);
  Eigen::VectorXd tail_sample(nsample);
  Eigen::VectorXd nu_sample(nsample);
  Eigen::VectorXd nugget_sample(nsample);

  double range_prop, tail_prop, nu_prop, nugget_prop;
  range_prop = range_curr;
  tail_prop = tail_curr;
  nu_prop = nu_curr;
  nugget_prop = nugget_curr;

  // MCMC stuff
  double log_prior_curr, log_prior_prop, log_pr_tran, log_pr_rtran, MH_ratio;
  double unif_rnd;
  Rcpp::LogicalVector accept_rate_range(nsample), accept_rate_nugget(nsample), 
    accept_rate_tail(nsample), accept_rate_nu(nsample);

  double Jacobian_curr=0, Jacobian_prop=0;

  // uniform prior on nu 
  double nu_lb = 0.0, nu_ub = 4.0;


  // create an empty list for abortion 
  // Rcpp::List abort_value = Rcpp::List::create(Rcpp::_["abort"]=R_NilValue);

  #ifdef USE_R
    GetRNGstate();
  #endif
  

  /****************************************************************************/
  Progress prog(nsample, verbose);
  if(family=="CH"){
    // begin MCMC sampling 
    for(int it=0; it<nsample; it++){
      if(Progress::check_abort()){
        return R_NilValue;
      }
      prog.increment();

      // update range 
      // loglik_curr = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // generate proposal
      range_prop = exp(Rcpp::rnorm(1, log(range_curr), Delta_range)[0]);
      loglik_prop = sp.MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        range_sample[it] = range_curr;
        // accept_rate_range(it) = FALSE;
      }else{
        // log prior density of cauchy dist
        // log_prior_curr = -log(1.0+range_curr*range_curr);
        // log_prior_prop = -log(1.0+range_prop*range_prop);
        log_prior_curr = sp.reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_curr, covmodel, false);
        log_prior_prop = sp.reference_prior(H, d, range_prop, tail_curr, nu_curr, nugget_curr, covmodel, false);

        // difference of log proposal density is zero because RW normal propsal is used
        log_pr_tran =  0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(range_curr);
        Jacobian_prop = log(range_prop);
        MH_ratio = (loglik_prop - loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_range[it] = TRUE;
          range_curr = range_prop;
          loglik_curr = loglik_prop;
        }

        range_sample(it) = range_curr;
      }

      // update tail decay parameter

      // generate proposal
      tail_prop = exp(Rcpp::rnorm(1, log(tail_curr), Delta_tail)[0]);
      loglik_prop = sp.MLoglik(range_curr, tail_prop, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        tail_sample(it) = tail_curr;
        // accept_rate_tail[it]= FALSE;
      }else{
        // log prior density of cauchy dist
        // log_prior_curr = -log(1.0+tail_curr*tail_curr);
        // log_prior_prop = -log(1.0+tail_prop*tail_prop);
        log_prior_curr = sp.reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_curr, covmodel, false);
        log_prior_prop = sp.reference_prior(H, d, range_curr, tail_prop, nu_curr, nugget_curr, covmodel, false);

        // log proposal density 
        log_pr_tran = 0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(tail_curr);
        Jacobian_prop = log(tail_prop);
        MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                      + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_tail[it]= TRUE;
          tail_curr = tail_prop;
          loglik_curr = loglik_prop;
        }

        tail_sample(it) = tail_curr;
      }


        // generate proposal
        nugget_prop = exp(Rcpp::rnorm(1, log(nugget_curr), Delta_nugget)[0]);
        loglik_prop = sp.MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nugget_sample(it) = nugget_curr;
          // accept_rate_nugget[it]= FALSE;
        }else{
          // log prior density of cauchy dist
          // log_prior_curr = -log(1.0+nugget_curr*nugget_curr);
          // log_prior_prop = -log(1.0+nugget_prop*nugget_prop);
          log_prior_curr = sp.reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_curr, covmodel, false);
          log_prior_prop = sp.reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_prop, covmodel, false);

          // log proposal density 
          log_pr_tran = 0;
          log_pr_rtran = 0;

          // Jacobian 
          Jacobian_curr = log(nugget_curr);
          Jacobian_prop = log(nugget_prop);

          MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                          + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_nugget[it] =TRUE;
            nugget_curr = nugget_prop;
            loglik_curr = loglik_prop;
          }

          nugget_sample(it) = nugget_curr;
        }

      if(smoothness_est){

        // loglik_curr = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // generate proposal
        // nu_prop = exp(Rcpp::rnorm(1, log(nu_curr), Delta_nu)[0]);
        nu_prop = ilogit(Rcpp::rnorm(1, logit(nu_curr, nu_lb, nu_ub), Delta_nu)[0], nu_lb, nu_ub);   

        loglik_prop = sp.MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nu_sample(it) = nu_curr;
          // accept_rate_nu[it]= FALSE;
        }else{
          // log prior density 
          log_prior_curr = 0;
          log_prior_prop = 0;
          // log_prior_curr = sp.reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_curr, covmodel, smoothness_est);
          // log_prior_prop = sp.reference_prior(H, d, range_curr, tail_curr, nu_prop, nugget_curr, covmodel, smoothness_est);

          // log proposal density 
          log_pr_tran = 0;
          log_pr_rtran = 0;

          // Jacobian 
          // Jacobian_curr = log(nu_curr);
          // Jacobian_prop = log(nu_prop);
          Jacobian_curr = log(nu_curr-nu_lb) + log(nu_ub-nu_curr);
          Jacobian_prop = log(nu_prop-nu_lb) + log(nu_ub-nu_prop);

          MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                          + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_nu[it]=TRUE;
            nu_curr = nu_prop;
            loglik_curr = loglik_prop;
          }

          nu_sample(it) = nu_curr;
        }
      }     

    } // end for-loop for CH correlation

  }else if(family=="matern"){
    // begin MCMC sampling 
    for(int it=0; it<nsample; it++){
      if(Progress::check_abort()){
        return R_NilValue;
      }
      prog.increment();

      // update range 
      // loglik_curr = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // generate proposal
      range_prop = exp(Rcpp::rnorm(1, log(range_curr), Delta_range)[0]);
      loglik_prop = sp.MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        range_sample[it] = range_curr;
        // accept_rate_range(it) = FALSE;
      }else{
        // log prior density of cauchy dist
        // log_prior_curr = -log(1.0+range_curr*range_curr);
        // log_prior_prop = -log(1.0+range_prop*range_prop);
        log_prior_curr = sp.reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_curr, covmodel, false);
        log_prior_prop = sp.reference_prior(H, d, range_prop, tail_curr, nu_curr, nugget_curr, covmodel, false);

        // difference of log proposal density is zero because RW normal propsal is used
        log_pr_tran =  0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(range_curr);
        Jacobian_prop = log(range_prop);
        MH_ratio = (loglik_prop - loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_range[it] = TRUE;
          range_curr = range_prop;
          loglik_curr = loglik_prop;
        }

        range_sample(it) = range_curr;
      }

        // generate proposal
        nugget_prop = exp(Rcpp::rnorm(1, log(nugget_curr), Delta_nugget)[0]);
        loglik_prop = sp.MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nugget_sample(it) = nugget_curr;
          // accept_rate_nugget[it]= FALSE;
        }else{
          // log prior density of cauchy dist
          // log_prior_curr = -log(1.0+nugget_curr*nugget_curr);
          // log_prior_prop = -log(1.0+nugget_prop*nugget_prop);
          log_prior_curr = sp.reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_curr, covmodel, false);
          log_prior_prop = sp.reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_prop, covmodel, false);

          // log proposal density 
          log_pr_tran = 0;
          log_pr_rtran = 0;

          // Jacobian 
          Jacobian_curr = log(nugget_curr);
          Jacobian_prop = log(nugget_prop);

          MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                          + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_nugget[it] =TRUE;
            nugget_curr = nugget_prop;
            loglik_curr = loglik_prop;
          }

          nugget_sample(it) = nugget_curr;
        }

      if(smoothness_est){

        // loglik_curr = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // generate proposal
        // nu_prop = exp(Rcpp::rnorm(1, log(nu_curr), Delta_nu)[0]);
        nu_prop = ilogit(Rcpp::rnorm(1, logit(nu_curr, nu_lb, nu_ub), Delta_nu)[0], nu_lb, nu_ub);   
        loglik_prop = sp.MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nu_sample(it) = nu_curr;
          // accept_rate_nu[it]= FALSE;
        }else{
          // log prior density 
          log_prior_curr = 0;
          log_prior_prop = 0;
          // log_prior_curr = sp.reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_curr, covmodel, smoothness_est);
          // log_prior_prop = sp.reference_prior(H, d, range_curr, tail_curr, nu_prop, nugget_curr, covmodel, smoothness_est);

          // log proposal density 
          log_pr_tran = 0;
          log_pr_rtran = 0;

          // Jacobian 
          // Jacobian_curr = log(nu_curr);
          // Jacobian_prop = log(nu_prop);
            Jacobian_curr = log(nu_curr-nu_lb) + log(nu_ub-nu_curr);
            Jacobian_prop = log(nu_prop-nu_lb) + log(nu_ub-nu_prop);

          MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                          + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_nu[it]=TRUE;
            nu_curr = nu_prop;
            loglik_curr = loglik_prop;
          }

          nu_sample(it) = nu_curr;
        }
      }     

    } // end for-loop for matern correlation
  }else{
    Rcpp::stop("The MCMC algorithm for the specified covariance family is not implemented.\n");
  }




/****************************************************************************/

  #ifdef USE_R
    PutRNGstate();
  #endif


  if(family=="CH"){

    if(smoothness_est){
      return Rcpp::List::create(Rcpp::_["range"]=range_sample, 
                                Rcpp::_["tail"]=tail_sample,
                                Rcpp::_["nugget"]=nugget_sample,
                                Rcpp::_["nu"]=nu_sample,
                                Rcpp::_["accept_rate_range"]=accept_rate_range,
                                Rcpp::_["accept_rate_tail"]=accept_rate_tail,
                                Rcpp::_["accept_rate_nugget"]=accept_rate_nugget,
                                Rcpp::_["accept_rate_nu"]=accept_rate_nu
                                );      
    }else{
      return Rcpp::List::create(Rcpp::_["range"]=range_sample, 
                                Rcpp::_["tail"]=tail_sample,
                                Rcpp::_["nugget"]=nugget_sample,
                                Rcpp::_["accept_rate_range"]=accept_rate_range,
                                Rcpp::_["accept_rate_tail"]=accept_rate_tail,
                                Rcpp::_["accept_rate_nugget"]=accept_rate_nugget);
    }

  }else if(family=="matern"){

    if(smoothness_est){
      return Rcpp::List::create(Rcpp::_["range"]=range_sample, 
                                Rcpp::_["nugget"]=nugget_sample,
                                Rcpp::_["nu"]=nu_sample,
                                Rcpp::_["accept_rate_range"]=accept_rate_range,
                                Rcpp::_["accept_rate_nugget"]=accept_rate_nugget,
                                Rcpp::_["accept_rate_nu"]=accept_rate_nu
                                );      
    }else{
      return Rcpp::List::create(Rcpp::_["range"]=range_sample, 
                                Rcpp::_["nugget"]=nugget_sample,
                                Rcpp::_["accept_rate_range"]=accept_rate_range,
                                Rcpp::_["accept_rate_nugget"]=accept_rate_nugget);
    }
  }else{
    Rcpp::stop("The MCMC algorithm for the specified covariance family is not implemented.\n");
  }


}


// [[Rcpp::export]]
Rcpp::List model_evaluation(Eigen::Map<Eigen::MatrixXd> output,  
	Eigen::Map<Eigen::MatrixXd> H, Eigen::Map<Eigen::MatrixXd> input,
	const Rcpp::List& covmodel, const double& smoothness,
	const Rcpp::List& sample, Eigen::Map<Eigen::MatrixXd> output_new,
	Eigen::Map<Eigen::MatrixXd> input_new, Eigen::Map<Eigen::MatrixXd> Hnew,
	const std::string& dtype, bool pointwise=true, bool joint=true){

  	std::string family = Rcpp::as<std::string>(covmodel["family"]);
  	std::string form = Rcpp::as<std::string>(covmodel["form"]);

  	SP sp;
  	UQ uq;

  	int Dim = input.cols();
  	int nsample;
  	
  	nsample = (Rcpp::as<Eigen::VectorXd>(sample["nugget"])).size();
  	

  	Eigen::MatrixXd range_sample(nsample, Dim);
  	if(sample.containsElementNamed("range")){
  		if(Rf_isMatrix(sample("range"))){
  			range_sample = Rcpp::as<Eigen::MatrixXd>(sample["range"]);
  		}else if(Rf_isNumeric(sample["range"])){
  			range_sample.col(0) = Rcpp::as<Eigen::VectorXd>(sample["range"]);
  		}
  	}else{
  		Rcpp::stop("The range parameter is not provided.\n");
  	}

  	Eigen::MatrixXd tail_sample(nsample, Dim);
  	if(sample.containsElementNamed("tail")){
  		if(Rf_isMatrix(sample("tail"))){
  			tail_sample = Rcpp::as<Eigen::MatrixXd>(sample["tail"]);
  		}else if(Rf_isNumeric(sample["tail"])){
  			tail_sample.col(0) = Rcpp::as<Eigen::VectorXd>(sample["tail"]);
  		}
  	}else{
  		if(family=="CH"){
  			Rcpp::stop("The tail decay parameter is not provided.\n");
  		}else{
  			tail_sample = 0.5*Eigen::MatrixXd::Ones(nsample, Dim);
  		}
  	}

  	Eigen::MatrixXd nu_sample(nsample, Dim);
  	if(sample.containsElementNamed("nu")){
  		if(Rf_isMatrix(sample("nu"))){
  			nu_sample = Rcpp::as<Eigen::MatrixXd>(sample["nu"]);
  		}else if(Rf_isNumeric(sample["nu"])){
  			nu_sample.col(0) = Rcpp::as<Eigen::VectorXd>(sample["nu"]);
  		}
  	}else{
  		nu_sample = smoothness*Eigen::MatrixXd::Ones(nsample, Dim);
  	}

   	Eigen::VectorXd nugget_sample(nsample);
  	if(sample.containsElementNamed("nugget")){
  		nugget_sample = Rcpp::as<Eigen::VectorXd>(sample["nugget"]);
  	}else{
  		Rcpp::stop("The nugget parameter is not provided.\n");
  	}

  	Rcpp::List Result; 
  	// if(joint){
  		if(form=="isotropic"){
  			Result = sp.model_evaluation(output, input, H, 
  				range_sample.col(0), tail_sample.col(0),
  				nu_sample.col(0), nugget_sample, covmodel, 
  				output_new, input_new, Hnew, dtype,
  				pointwise, joint);
  		}else if(form=="tensor"){
  			Result = uq.tensor_model_evaluation(output, input, H, 
  				range_sample, tail_sample, nu_sample, nugget_sample,
  				 covmodel, output_new, input_new, Hnew, dtype,
  				pointwise, joint); 			
  		}else if(form=="ARD"){
  			Result = uq.ARD_model_evaluation(output, input, H, 
  				range_sample, tail_sample.col(0), nu_sample.col(0), nugget_sample,
  				 covmodel, output_new, input_new, Hnew, dtype,
  				pointwise, joint); 			
  		}else{
  			Rcpp::stop("The covariance kernel is not supported yet.\n");
  		}
  	// }else{
  	// 	if(form=="isotropic"){

  	// 	}else if(form=="tensor" || form=="ARD"){

  	// 	}else{
  	// 		Rcpp::stop("The covariance kernel is not supported yet.\n");
  	// 	}
  	// }
 	

  	return Result;

}


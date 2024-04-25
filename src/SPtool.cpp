#ifndef _USE_RcppEigen
#define _USE_RcppEigen
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp11)]]
#endif

#ifndef _USE_GSL
#define _USE_GSL
#include <gsl/gsl_sf.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_errno.h>
#endif


#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#include <cmath>
#endif


#include "utils.h"

#ifndef _USE_ProgressBar
#define _USE_ProgressBar
#include <R_ext/Utils.h>   // interrupt the Gibbs sampler from R
#include <iostream>
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>
#include <progress_bar.hpp>
#endif


class SP
{
  public:
    // Default constructor.

    // member functions 
    Eigen::MatrixXd pdist(const Eigen::MatrixXd& locs1, const Eigen::MatrixXd& locs2, 
      const std::string& dtype);
    Rcpp::List tdist(const Eigen::MatrixXd& locs1, const Eigen::MatrixXd& locs2, 
      const double& cutRange, const std::string& dtype);
    Eigen::MatrixXd iso_kernel(const Eigen::MatrixXd& d, const double& range, 
      const double& tail, const double& nu, const std::string & family);
    Eigen::MatrixXd iso_kernel0(const Eigen::MatrixXd& input1, 
      const Eigen::MatrixXd& input2, const double& range, const double& tail, 
      const double& nu, const std::string & family, const std::string& dtype);
    Rcpp::List deriv_iso_kernel(const Eigen::MatrixXd& d, const double& range, 
      const double& tail, const double& nu, const std::string& family);

    Eigen::MatrixXd FisherInfo(const Eigen::MatrixXd& d, const double& sig2,
      const Eigen::VectorXd& range, const Eigen::VectorXd& tail,
      const Eigen::VectorXd& nu, const double& nugget,  
      const std::string& family);
    
    double reference_prior(const Eigen::MatrixXd& H, const Eigen::MatrixXd& d, 
      const double& range, const double& tail, const double& nu, const double& nugget, 
      const Rcpp::List& covmodel, const bool& smoothness_est);

    // helper
    // double logit(double x, double lb, double ub);
    // double ilogit(double x, double lb, double ub);
    // void tran_par(Rcpp::List& par, std::string covmodel="CH", Rcpp::Nullable<Rcpp::List> bound=R_NilValue,
    //             bool nugget_est=true);
    // void itran_par(Rcpp::List& par, std::string covmodel="CH", Rcpp::Nullable<Rcpp::List> bound=R_NilValue,
    //             bool nugget_est=true);

    double MLoglik(const double& range, const double& tail, const double& nu, 
      const double& nugget,  const Eigen::MatrixXd& y, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& d, const Rcpp::List& covmodel);
    Eigen::VectorXd gradient_MLoglik(const double& range, const double& tail, 
      const double& nu, const double& nugget, const Eigen::MatrixXd& y, 
      const Eigen::MatrixXd& H, const Eigen::MatrixXd& d, const Rcpp::List& covmodel,
      const bool & smoothness_est);
    double PLoglik(const double& range, const double& tail, const double& nu, 
      const double& nugget, const Eigen::MatrixXd& y, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& d, const Rcpp::List& covmodel);

    // prediction 
    Eigen::MatrixXd simulate(const Eigen::MatrixXd& input, const Eigen::MatrixXd& H, 
      const Eigen::VectorXd& coeff, const double& sig2, const double& range, 
      const double& tail, const double& nu, const double& nugget, const Rcpp::List& covmodel, 
      const int & nsample, const std::string& dtype);
    Rcpp::List predict(const Eigen::MatrixXd& y, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& input, const Eigen::MatrixXd& input_new, 
      const Eigen::MatrixXd& Hnew, const double& range, const double& tail, 
      const double& nu, const double& nugget, const Rcpp::List& covmodel, 
      const std::string& dtype);
    Rcpp::List simulate_predictive_dist(const Eigen::MatrixXd& y, 
      const Eigen::MatrixXd& H, const Eigen::MatrixXd& input,
      const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew, 
      const Eigen::VectorXd& range, const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, 
      const Eigen::VectorXd& nugget, const Rcpp::List& covmodel, const std::string& dtype);
    
    // Conditional Simulation
    Rcpp::List condsim(const Eigen::MatrixXd& y, 
      const Eigen::MatrixXd& H, const Eigen::MatrixXd& input,
      const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew, 
      const double& range, const double& tail, const double& nu, 
      const double& nugget, const Rcpp::List& covmodel, const std::string& dtype, int nsample);

    // MCMC algorithm
    Rcpp::List iso_MCMCOBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& input, const double& range, const double& tail, 
      const double& nu, const double& nugget, const Rcpp::List& covmodel, 
      const bool & smoothness_est, const Rcpp::List& proposal, 
      const int& nsample, const std::string& dtype, const bool & verbose);

    Rcpp::List iso_MCMCOBayes_Ref(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& input, const double& range, const double& tail, const double& nu, 
      const double& nugget, const Rcpp::List& covmodel, const bool& smoothness_est, const Rcpp::List& proposal,
      const int & nsample, const std::string & dtype, const bool& verbose);

    Rcpp::List iso_MCMCSBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& input, const double& range, const double& tail, 
      const double& nu, const double& nugget, const Rcpp::List& covmodel, 
      const bool & smoothness_est, const Rcpp::List& prior, const Rcpp::List& proposal, 
      const int& nsample, const std::string & dtype, const bool& verbose);

    // MCMC estimation + prediction
    // MCMC for isotropic covariance kernels with Random Walk on unconstrained parameter space
    Rcpp::List iso_MCMCOBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& input, const double& range, const double& tail, const double& nu, 
      const double& nugget, const Rcpp::List& covmodel, const bool& smoothness_est, const Rcpp::List& proposal,
      const int& nsample, const std::string& dtype, const bool& verbose, const Eigen::MatrixXd& input_new, 
      const Eigen::MatrixXd& Hnew);   

    Rcpp::List iso_MCMCOBayes_Ref(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& input, const double& range, const double& tail, const double& nu, 
      const double& nugget, const Rcpp::List& covmodel, const bool& smoothness_est, const Rcpp::List& proposal,
      const int& nsample, const std::string& dtype, const bool& verbose, const Eigen::MatrixXd& input_new, 
      const Eigen::MatrixXd& Hnew); 

    Rcpp::List iso_MCMCSBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& input, const double& range, const double& tail, const double& nu, 
      const double& nugget, const Rcpp::List& covmodel, const bool& smoothness_est, const Rcpp::List& prior,
      const Rcpp::List& proposal, const int& nsample, const std::string& dtype, const bool& verbose, 
      const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew);

    // model comparison and evaluation 
    Rcpp::List model_evaluation(const Eigen::MatrixXd& output, const Eigen::MatrixXd& input,
      const Eigen::MatrixXd& H, const Eigen::VectorXd& range,
      const Eigen::VectorXd& tail, const Eigen::VectorXd& nu,
      const Eigen::VectorXd& nugget, const Rcpp::List& covmodel,
      const Eigen::MatrixXd& output_new, const Eigen::MatrixXd& input_new, 
      const Eigen::MatrixXd& Hnew, const std::string& dtype, 
      const bool& pointwise, const bool& joint);
    
    // constructor 
    SP()
    {
      
    }
    // destructor 
    ~SP()
    {
      
    }
};


// @title Compute a distance matrix based on two sets of locations
// 
// @description This function computes the distance matrix based on two sets of locations.
// @param locs1 a matrix of locations
// @param locs2 a matrix of locations
// @param dtype a string indicates which type of distance is used: \stron{Euclidean} (default), \strong{GCD}, where the latter indicates Great Circle Distance.
// 
// @author Pulong Ma <mpulong@gmail.com>
//
Eigen::MatrixXd  SP::pdist(const Eigen::MatrixXd& locs1, const Eigen::MatrixXd& locs2, 
  const std::string& dtype)
{


  int n1 = locs1.rows();
  int n2 = locs2.rows();
  Eigen::MatrixXd distmat(n1, n2); 
  

  if(dtype=="GCD"){

    //#pragma omp parallel for collapse(2)
    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        distmat(i,j) = gcdistance(locs1(i,0), locs1(i,1), locs2(j,0), locs2(j,1));
      }
    }

  }else if(dtype=="Euclidean"){

    //#pragma omp parallel for collapse(2)
    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        distmat(i,j) = sqrt((locs1.row(i)-locs2.row(j)).array().square().sum());
      }
    }
  }else{
    Rcpp::stop("SP::pdist: distance type is not correctly specified.\n");
  }

  

  return distmat;
}


// @title Compute a distance matrix based on two sets of locations up to a cutoff distance
// 
// @description This function computes the distance matrix between pairs of
// locations. It should be used to compute distance between pairs of location 
// up to a short tapering range. In addition, this function is about 5 times faster  
// than the one without using bounding box to find the nearest points.
// 
// @param locs1 a matrix of locations
// @param locs2 a matrix of locations
// @param cutRange a cutoff distance specifies the tapering range
// @param dtype a string indicates which type of distance is used: great_circle (default),
// Euclidean
//
// @return a list of arguments for nonzero distances:
// \describe{
// \item{rowid}{a vector of row indeces of nonzero elements.} 
// \item{colid}{a vector of column indeces of nonzero elements.}
// \item{val}{a vector of nonzero elements.}
// \item{sp}{a sparse matrix of class `dgCMatrix'.}
//}
//
// @author Pulong Ma <mpulong@gmail.com>
//
Rcpp::List SP::tdist(const Eigen::MatrixXd& locs1, const Eigen::MatrixXd& locs2, 
              const double& cutRange, const std::string& dtype)
{
  
  // Purpose: This function computes great circle distance between pairs of
  // locations in Km. It uses haversine formula to compute these distances.
  // It should be used to compute distance between pairs of location up to a
  // short tapering range. In addition, this function is about 5 times faster  
  // than the one without using bounding box to find the nearest points.
  //
  // Input Arguments:
  //
  // (1) locs1: longitude and latitude of locations for which distance needs to be computed.
  // (2) locs2: longitude and latitude of locations for which distance needs to be computed.
  // (3) cutRange: tapering range.
  

  int counter = 0;
  int n1 = locs1.rows();
  int n2 = locs2.rows();

  int max_elem = (n1>50000) ? (int(1000*n1)) : (n1*n2);
  
  // arma::uvec rowid(max_elem);
  // arma::uvec colid(max_elem);
  // arma::vec d(max_elem);

  Eigen::VectorXd rowid(max_elem), colid(max_elem), d(max_elem);

  
  // arma::uvec col_ptr = arma::zeros<arma::uvec>(n2+1);


  if(dtype=="GCD"){

    double earthRadius = 6371.0087714; //WGS84 mean radius in km 

    
    // Variables to set circular boundaries.
    double deltaLat, deltaLon, maxLat, minLat, maxLon, minLon, lonTemp1;
    double locDistance;
    
    deltaLat = cutRange / earthRadius;
    
    
    for (int i = 0; i < n2; i++)
    {
      
      
      // Compute bounding box coordinates in which the locations
      // within the tapering range would be located
      deltaLon = arcsine(sin(deltaLat) / cos(locs2(i,1)*M_PI/180.0));
      bounds(locs2(i,0)*M_PI/180.0, locs2(i,1)*M_PI/180.0, minLon, maxLon, minLat, maxLat, deltaLon, deltaLat);
      
      for (int j = 0; j < n1; j++)
      {
        //Adjust for 180 degree meridian.
        lonTemp1 = lonAdjust(locs2(i,0)*M_PI/180.0, locs1(j,0)*M_PI/180.0);
        //lonTemp2 = lonAdjust(locs2(i,0)*M_PI/180.0, locs1(j,0)*M_PI/180.0);
        
        //Bounding box for finding distances.
        if ((locs1(j,1)*M_PI/180.0 <= maxLat) && (locs1(j,1)*M_PI/180.0 >= minLat) && 
            (lonTemp1 <= maxLon) && (lonTemp1 >= minLon))
        {
          //Row and column index of the output sparse matrix
          //Compute great circle distance;
          locDistance = gcdistance(locs2(i,0), locs2(i,1), locs1(j,0), locs1(j,1));
          if (locDistance < cutRange && locDistance > 0)
          {
            rowid(counter) = j;
            colid(counter) = i;
            d(counter) = locDistance;
            counter += 1;
          }
        }
      }
    }

  }else if(dtype=="Euclidean"){

   // Variables to set circular boundaries.
    double delta = cutRange;
    double locDistance;
    
    int dim = locs1.cols();
    // arma::rowvec dtemp = arma::zeros<arma::rowvec>(dim);
    // arma::rowvec minlimit = arma::zeros<arma::rowvec>(dim);
    // arma::rowvec maxlimit = arma::zeros<arma::rowvec>(dim);

    Eigen::VectorXd dtemp(dim), minlimit(dim), maxlimit(dim);
    //Eigen::VectorXi d1(dim);
    Eigen::Matrix<bool, Eigen::Dynamic, 1> d1(dim);

    for (int i = 0; i < n2; i++)
    {
      
      // col_ptr(i) = counter;
      
      // Compute bounding box coordinates in which the locations
      // within the tapering range would be located
      //eucbounds(locs2(i,0), locs2(i,1), minLon, maxLon, minLat, maxLat, deltaLon, deltaLat);
      minlimit = locs2.row(i).array() - delta;
      maxlimit = locs2.row(i).array() + delta;

      for (int j = 0; j < n1; j++)
      {      
        //Bounding box for finding distances.
        dtemp = locs2.row(i) - locs1.row(j);
        d1 = dtemp.array() < delta;

        if (d1.all())
        {
          //Row and column index of the output sparse matrix
          //Compute euclidean distance;
          
          locDistance = dtemp.squaredNorm();
          if (locDistance < cutRange && locDistance > 0)
          {
            rowid(counter) = j;
            colid(counter) = i;
            d(counter) = locDistance;
            counter += 1;
          }
        }
      }
    }
    

  }
  
  // col_ptr(n2) = counter;
  
  // arma::sp_mat distmat(rowid.head(counter), col_ptr, d.head(counter), n1, n2);
  Eigen::SparseMatrix<double> distmat(n1, n2);
  distmat.reserve(counter);
  for(int i=0; i<counter; i++){
    distmat.insert(rowid(i), colid(i)) = d(i);
  }
  distmat.makeCompressed();

  return Rcpp::List::create(Rcpp::_["rowid"] = rowid.head(counter), 
                            Rcpp::_["colid"] = colid.head(counter), 
                            Rcpp::_["val"] = d.head(counter),
                            Rcpp::_["spmat"] = distmat
  );    
  
  
}



/****************************************************************************************/
/****************************************************************************************/






Eigen::MatrixXd SP::iso_kernel(const Eigen::MatrixXd& d, const double& range, const double& tail, 
  const double& nu, const std::string& family){

  Eigen::MatrixXd cormat(d.rows(), d.cols());

  // double tail, nu, phi;

  if(family=="CH"){
    cormat = CH(d, range, tail, nu);

  }else if(family=="matern"){

    cormat = matern(d, range, nu);

  // }else if(family=="exp"){
  //   cormat = matern(d, range, 0.5);

  }else if(family=="gauss"){
    cormat = powexp(d, range, 2.0);

  }else if(family=="powexp"){
    cormat = powexp(d, range, nu);   
     
  }else if(family=="cauchy"){
    cormat = cauchy(d, range, tail, nu);

  }else{
    Rcpp::stop("The family of covariance functions is not yet supported!\n");
  }


  return cormat;

}




// @title A wraper to build different kinds of covariance matrix
// 
// @description This function wraps existing built-in routines to construct a covariance 
// matrix based on data type, covariance type, and distance type. The constructed 
// covariance matrix can be directly used for GaSP fitting and and prediction for spatial 
// data, spatio-temporal data, and computer experiments. 
//
// @param input1 a matrix of inputs
// @param input2 a matrix of inputs
// 
// @param param a list including values for regression parameters, covariance parameters, 
// and nugget variance parameter.
// The specification of \strong{param} should depend on the covariance model. 
// \itemize{
// \item{The regression parameters are denoted by \strong{b}.}
// \item The marginal variance or partial sill is denoted by \strong{sig2}. 
// \item{The nugget variance parameter is denoted by \strong{nugget} for all covariance models. 
// }
// \item{For the Confluent Hypergeometric class, \strong{range} is used to denote the range parameter \eqn{\beta}. 
// \strong{alpha} is used to denote the tail decay parameter \eqn{\alpha}. \strong{nu} is used to denote the 
// smoothness parameter \eqn{\nu}.}
// \item{For the generalized Cauchy class, \strong{range} is used to denote the range parameter \eqn{\phi}. 
// \strong{alpha} is used to denote the tail decay parameter \eqn{\alpha}. \strong{nu} is used to denote the 
// smoothness parameter \eqn{\nu}.}
// \item{For the Matérn class, \strong{range} is used to denote the range parameter \eqn{\phi}. 
// \strong{nu} is used to denote the smoothness parameter \eqn{\nu}. When \eqn{\nu=0.5}, the 
// Matérn class corresponds to the exponential covariance.}  
// \item{For the powered-exponential class, \strong{range} is used to denote the range parameter \eqn{\phi}.
// \strong{nu} is used to denote the smoothness parameter. When \eqn{\nu=2}, the powered-exponential class
// corresponds to the Gaussian covariance.}
// }
// @param covmodel a string indicating the type of covariance class.
// The following correlation functions are implemented:
// \describe{
// \item{CH}{The Confluent Hypergeometric correlation function is given by 
// \deqn{C(h) = \frac{\Gamma(\nu+\alpha)}{\Gamma(\nu)} 
// \mathcal{U}(\alpha, 1-\nu, \nu h^2/\beta),}
// where \eqn{\alpha} is the tail decay parameter. \eqn{\beta} is the range parameter.
// \eqn{\nu} is the smoothness parameter. \eqn{\mathcal{U}(\cdot)} is the confluent hypergeometric
// function of the second kind. For details about this covariance, 
// see Ma and Bhadra (2023; \doi{10.1080/01621459.2022.2027775}).  
// }
// \item{cauchy}{The generalized Cauchy covariance is given by
// \deqn{C(h) = \left\{ 1 + \left( \frac{h}{\phi} \right)^{\alpha}  
//             \right\}^{-\nu/\alpha},}
// where \eqn{\phi} is the range parameter. \eqn{\alpha} is the tail decay parameter.
// \eqn{\nu} is the smoothness parameter.
//}
//
// \item{exp}{The exponential correlation function is given by 
// \deqn{C(h)=\exp(-h/\phi),}
// where \eqn{\phi} is the range parameter.
// }
// \item{matern_3_2}{The Matérn correlation function with \eqn{\nu=1.5}.
// }
// \item{matern_5_2}{The Matérn correlation function with \eqn{\nu=2.5}.
// }
// \item{matern}{The Matérn correlation function is given by
// \deqn{C(h)=\frac{2^{1-\nu}}{\Gamma(\nu)} \left(\sqrt{2\nu} \frac{h}{\phi} \right)^{\nu} 
// \mathcal{K}_{\nu}\left( \sqrt{2\nu} \frac{h}{\phi} \right),}
// where \eqn{\phi} is the range parameter. \eqn{\nu} is the smoothness parameter. 
// \eqn{\mathcal{K}_{\nu}(\cdot)} is the modified Bessel function of the second kind of order \eqn{\nu}.
// }
//
//
// \item{powexp}{The powered-exponential correlation function is given by
//                \deqn{C(h)=\exp\left\{-\left(\frac{h}{\phi}\right)^{\nu}\right\},}
// where \eqn{\phi} is the range parameter. \eqn{\nu} is the smoothness parameter.
// }
// \item{gauss}{The Gaussian correlation function is given by 
// \deqn{C(h)=\exp\left(-\frac{h^2}{\phi^2}\right),}
// where \eqn{\phi} is the range parameter.
// }
// }
// @param type a string indicating the type of data
// @param dtype a string indicating the type of distance
// 
// @return a matrix of covariance
// @seealso \code{\link{gp}}
// @author Pulong Ma <mpulong@gmail.com>
// 
Eigen::MatrixXd SP::iso_kernel0(const Eigen::MatrixXd& input1, const Eigen::MatrixXd& input2, 
                const double& range, const double& tail, const double& nu, 
                const std::string& family, const std::string& dtype){

  int n1 = input1.rows();
  int n2 = input2.rows();

  Eigen::MatrixXd covmat(n1, n2);

  Eigen::MatrixXd distsp = SP::pdist(input1, input2, dtype);

  covmat = SP::iso_kernel(distsp, range, tail, nu, family);    

  return covmat;
}

/****************************************************************************************/
/****************************************************************************************/

Rcpp::List SP::deriv_iso_kernel(const Eigen::MatrixXd& d, const double& range, const double& tail, const double& nu, const std::string& family){

  int n1, n2;
  n1 = d.rows();
  n2 = d.cols();
  Eigen::MatrixXd covmat(n1,n2);


  Rcpp::List dR(3);
  int count;

  if(family=="CH"){
    dR[0] = CH_deriv_range(d, range, tail, nu);
    dR[1] = CH_deriv_tail(d, range, tail, nu);
    dR[2] = CH_deriv_nu(d, range, tail, nu);
    count = 3;
  }else if(family=="matern"){
    dR[0] = matern_deriv_range(d, range, nu);
    dR[1] = R_NilValue;
    count = 2;
  // }else if(family=="exp"){
  //   dR[0] = matern_deriv_range(d, range, 0.5);
  //   dR[1] = R_NilValue;
  //   count = 2;
  // }else if(family=="matern_3_2"){
  //   dR[0] = matern_deriv_range(d, range, 1.5); 
  //   dR[1] = R_NilValue;
  //   count = 2;  
  // }else if(family=="matern_5_2"){
  //   dR[0] = matern_deriv_range(d, range, 2.5); 
  //   dR[1] = R_NilValue;
  //   count = 2;   
  }else if(family=="gauss"){
    dR[0] = powexp_deriv_range(d, range, 2.0); 
    dR[1] = R_NilValue;
    count = 2;  
  }else if(family=="powexp"){
    dR[0] = powexp_deriv_range(d, range, nu);
    dR[1] = R_NilValue;
    count = 2;
  }else if(family=="cauchy"){
    dR[0] = cauchy_deriv_range(d, range, tail, nu);
    dR[1] = cauchy_deriv_tail(d, range, tail, nu);
    dR[2] = R_NilValue;
    count = 3;
  }else{
    Rcpp::stop("The family of covariance functions is not yet supported!\n");
  }

  Rcpp::List dRR(count);
  for(int k=0; k<count; k++){
    dRR[k] = dR[k];
  }

  return dRR;
}




/****************************************************************************************/
/****************************************************************************************/

Eigen::MatrixXd SP::FisherInfo(const Eigen::MatrixXd& d, const double& sig2,
  const Eigen::VectorXd& range, const Eigen::VectorXd& tail,
  const Eigen::VectorXd& nu, const double& nugget, 
  const std::string& family){


  // double sig2 = 1.0;
  // if(par.containsElementNamed("sig2")){
  //   sig2 = Rcpp::as<double>(par["sig2"]);
  // }

  // Eigen::VectorXd range;
  // if(par.containsElementNamed("range")){
  //   range = Rcpp::as<Eigen::VectorXd>(par["range"]);
  // }else{
  //   Rcpp::stop("SP::FisherInfo: No range parameter value is specified.\n");
  // }

  // Eigen::VectorXd tail;
  // if(par.containsElementNamed("tail")){
  //   tail = Rcpp::as<Eigen::VectorXd>(par["tail"]);
  // }else{
  //   tail = 0.5*Eigen::VectorXd::Ones(1); 
  // }

  // Eigen::VectorXd nu;
  // if(par.containsElementNamed("nu")){
  //   nu = Rcpp::as<Eigen::VectorXd>(par["nu"]);
  // }else{
  //   nu = 0.5*Eigen::VectorXd::Ones(1);
  // }
  
  // double nugget = 0;
  // if(par.containsElementNamed("nugget")){
  //   nugget = Rcpp::as<double>(par["nugget"]);
  // }

  int n = d.rows();

  Eigen::MatrixXd R = SP::iso_kernel(d, range(0), tail(0), nu(0), family);
  R.diagonal().array() += nugget;
  Eigen::MatrixXd E(n,n);
  E.setZero();
  E.diagonal().array() = 1.0;

  Eigen::MatrixXd RInv = R.llt().solve(E);

  Rcpp::List dR = SP::deriv_iso_kernel(d, range(0), tail(0), nu(0), family);
  int ncorpar = dR.size();
  Rcpp::List U(ncorpar+3); // add derivative w.r.t. sig2, nugget, nu 
  U[0] = 1.0/sig2 * E;
  for(int k=0; k<ncorpar; k++){
    U[k+1] = RInv*Rcpp::as<Eigen::MatrixXd>(dR[k]);
  }
  U[ncorpar+1] = RInv;


  Eigen::MatrixXd Ui(n,n), Uj(n,n);
  Eigen::MatrixXd I(ncorpar+3,ncorpar+3);
  I.setZero();

  if(family=="CH"){
    U[ncorpar+2] = CH_deriv_nu(d, range(0), tail(0), nu(0));
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






/****************************************************************************************/
/****************************************************************************************/

// Integrated/Marginal loglikelihood L(theta; y)
// theta contains the correlation parameter on the 
double SP::MLoglik(const double& range, const double& tail, const double& nu, const double& nugget,
 const Eigen::MatrixXd& y, const Eigen::MatrixXd& H, const Eigen::MatrixXd& d, 
  const Rcpp::List& covmodel){

  std::string family = Rcpp::as<std::string>(covmodel["family"]);

  int n = y.rows();
  int q = y.cols();
  int p = H.cols();
  Eigen::MatrixXd R(n,n), HRH(p,p);



  R = SP::iso_kernel(d, range, tail, nu, family);

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



// Derivative of the marginal likelihood w.r.t. each parameter 
Eigen::VectorXd SP::gradient_MLoglik(const double& range, const double& tail, 
  const double& nu, const double& nugget, const Eigen::MatrixXd& y, 
  const Eigen::MatrixXd& H, const Eigen::MatrixXd& d, const Rcpp::List& covmodel,
  const bool& smoothness_est){

  std::string family = Rcpp::as<std::string>(covmodel["family"]);

  int n = y.rows();
  int q = y.cols();
  int p = H.cols();
  Eigen::MatrixXd R(n,n), RInv(n,n), Q(n,n), RH(n,p), HRH(p,p);
  Eigen::VectorXd Ry(n);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH;


  R = SP::iso_kernel(d, range, tail, nu, family);
  R.diagonal().array() += nugget;

  ldltR.compute(R);
  RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
  Ry = RInv * y;
  
  RH = RInv * H;
  HRH = H.transpose() * RH;
  ldltH.compute(HRH);

  Q = RInv - RH*ldltH.solve(RH.transpose());
  Eigen::MatrixXd S2 = y.transpose()*Q*y; 


  Eigen::MatrixXd O(n,n), Qy(n,q);

  Qy = Q*y;
  Eigen::MatrixXd SyQ = S2.llt().solve(Qy.transpose());
  O = Q - Qy * SyQ;

  int len; // record the number of parameters

  Rcpp::List dR = SP::deriv_iso_kernel(d, range, tail, nu, family);
  len = dR.size() - 1;
  
  Eigen::VectorXd dloglik(len+2);
  int count;

  for(int k=0; k<len; k++){
    dloglik[k] = -0.5*(q-n+p)*(Q*(Rcpp::as<Eigen::MatrixXd>(dR[k]))).trace() - 0.5*(n-p)*(O*(Rcpp::as<Eigen::MatrixXd>(dR[k]))).trace();
  }
  count = len;


  // estimate nugget
    // compute the gradient w.r.t. nugget
    dloglik[len] = -0.5*(q-n+p)*Q.trace() - 0.5*(n-p)*O.trace();
    count += 1;

  if(smoothness_est){
    dloglik[len+1] = -0.5*(q-n+p)*(Q*(Rcpp::as<Eigen::MatrixXd>(dR[len]))).trace() - 0.5*(n-p)*(O*(Rcpp::as<Eigen::MatrixXd>(dR[len]))).trace();
    count += 1;
  }
  

  return dloglik.head(count);

}


/****************************************************************************************/
/****************************************************************************************/
/*********** profile log-likelihood function ************/
// @title Compute the profile log-likelihood 
// @description This function computes the profile loglikelihood function.
//
// @param par a list of parameters. 
// @param y a numerical vector of output
// @param d an \code{SEXP} type object of distance.
// @param covmodel a list of two strings containing: 
// \itemize{
//  \item{family}{\code{CH}(default), \code{matern}, \code{exp}, \code{matern_3_2}, \code{matern_5_2},
//  \code{gauss}, \code{cauchy}}
// \item{form}{\code{isotropic}(default)}
// }
// @param type a string indicating the type of data
// @param bound Default value is \code{NULL}. Otherwise, it should be a list
// containing the following elements depending on the covariance class:
// \itemize{
// \item{For nugget parameter \strong{nugget}, it lies in the interval \eqn{(0, 1)}.
// It is a list containing lower bound \strong{lb} and 
// upper bound \strong{ub} with default value 
// \code{{nugget}=list{lb=0, ub=1}}.}
// \item{For the Confluent Hypergeometric covariance class, correlation parameters consis of
// \strong{range} and \strong{alpha}. \strong{range} is a list containing
// lower bound \strong{lb} and upper bound \strong{ub} with default value
// \code{{range}=list{lb=1e-20, ub=1e10}}. \strong{alpha} is a list
// containing lower bound \strong{lb} and upper bound \strong{ub} with 
// default value \code{{alpha}=list{lb=1e-5, ub=6}}.}
// \item{For the Matérn covariance, exponential covariance, Gaussian 
// covariance, and powered-exponential covariance, the range parameter 
//  has suppport \eqn{(0, \infty)}. The log inverse range parameterization
//  is used: \eqn{\xi:=-\log(\phi)}. There is no need to specify \strong{bound}.}
//  \item{For Cauchy covariance, \strong{bound} is specified for the 
//  tail decay parameter \strong{alpha}. \strong{alpha} is a list containing
//  lower bound \strong{lb} and upper bound \strong{ub} with default value
//  \code{{alpha}=list{lb=1e-5, ub=2}}.}
// }
// @param nugget_est a logical value. If it is \code{TRUE}, the nugget parameter 
// will be estimated; otherwise the nugget is not included in the covariance
// model.
// 
double SP::PLoglik(const double& range, const double& tail, const double& nu, const double& nugget,
 const Eigen::MatrixXd& y, const Eigen::MatrixXd& H, const Eigen::MatrixXd& d, const Rcpp::List& covmodel){

 double loglik;
 loglik = SP::MLoglik(range, tail, nu, nugget, y, H, d, covmodel);

  return loglik;
}
/****************************************************************************************/
/****************************************************************************************/


double SP::reference_prior(const Eigen::MatrixXd& H, const Eigen::MatrixXd& d, 
    const double& range, const double& tail, const double& nu, const double& nugget, 
    const Rcpp::List& covmodel, const bool& smoothness_est){

    std::string family = Rcpp::as<std::string>(covmodel["family"]);


  int n=H.rows();
  int p=H.cols();
  
  Eigen::MatrixXd R(n,n), RInv(n,n), Q(n,n), RH(n,p), HRH(p,p);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH;
  
  Rcpp::List dR;

  R = SP::iso_kernel(d, range, tail, nu, family);
  dR = SP::deriv_iso_kernel(d, range, tail, nu, family);

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


/*****************************************************************************************/
/*****************************************************************************************/
Eigen::MatrixXd SP::simulate(const Eigen::MatrixXd& input, const Eigen::MatrixXd& H, 
const Eigen::VectorXd& coeff, const double& sig2, const double& range, const double& tail, const double& nu, const double& nugget, const Rcpp::List& covmodel, 
  const int& nsample, const std::string& dtype){

  int n = input.rows();
  Eigen::MatrixXd d = SP::pdist(input, input, dtype);

  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);
  Eigen::MatrixXd R(n,n);

  R = SP::iso_kernel(d, range, tail, nu, family);


  R.diagonal().array() += nugget;

  R = sig2 * R;
  Eigen::MatrixXd L = R.llt().matrixL();


  //int p = H.cols();

  Eigen::MatrixXd ysim(n, nsample);

#ifdef USE_R
    GetRNGstate();
#endif


  for(int i=0; i<nsample; i++){
    ysim.col(i) = H*coeff + L * Rcpp::as<Eigen::VectorXd>(Rcpp::rnorm(n, 0.0, 1.0));
  }


#ifdef USE_R
  PutRNGstate();
#endif 

  return ysim;
}

/*****************************************************************************************/
/*****************************************************************************************/
// Compute mean and variance in the Predictive distribution 
Rcpp::List SP::predict(const Eigen::MatrixXd& y, const Eigen::MatrixXd& H, 
  const Eigen::MatrixXd& input, const Eigen::MatrixXd& input_new, 
  const Eigen::MatrixXd& Hnew, const double& range, const double& tail, const double& nu,
  const double& nugget, const Rcpp::List& covmodel, const std::string& dtype){

  int n = y.rows();
  int q = y.cols();
  int m = input_new.rows();
  int p = H.cols();
  //int dim = input.cols();
  Eigen::MatrixXd R(n,n), RInv(n,n), RH(n,p), HRH(p,p), Rnew(n,m), HRHInv(p,p);
  Eigen::MatrixXd res(n,q), Ry(n,q), bhat(p,q);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH, ldlt;


  Eigen::MatrixXd sig2hat(q,q), predmean(m, q);

  Eigen::MatrixXd d = SP::pdist(input, input, dtype);
  Eigen::MatrixXd d0 = SP::pdist(input, input_new, dtype);

  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  R = SP::iso_kernel(d, range, tail, nu, family);
  Rnew = SP::iso_kernel(d0, range, tail, nu, family);

  R.diagonal().array() += nugget;

    for(int i=0; i<n; i++){
      for(int j=0; j<m; j++){
        if(d0(i,j)==0){
          Rnew(i,j) += nugget;
        }
      }
    }

  ldltR.compute(R);
  RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
  RH = RInv * H;
  HRH = H.transpose() * RH;
  ldltH.compute(HRH);
  Ry = RInv*y;

  bhat = ldltH.solve(H.transpose()*Ry);
  res = y - H*bhat;
  // Rnew = UQ::kernel(d0, par, covmodel);
  predmean = Hnew*bhat;
  predmean += Rnew.transpose()*(RInv*res);
  sig2hat = res.transpose()*RInv*res / (n-p);

  HRHInv = ldltH.solve(Eigen::MatrixXd::Identity(p,p));
  
  Eigen::VectorXd Rtmp(n), tmp(p), pred_corr(m);
  for(int k=0; k<m; k++){
    Rtmp = Rnew.col(k);
    tmp = Hnew.row(k) - RH.transpose()*Rtmp;
    pred_corr(k) = R(0,0) - Rtmp.dot(RInv*Rtmp) + tmp.dot(HRHInv*tmp);
  }


    double df = n-p;

    Eigen::MatrixXd sdmat(m,q), upper_mat(m,q), lower_mat(m,q);
    for(int j=0; j<q; j++){
      sdmat.col(j) = (pred_corr*sig2hat(j,j)).array().sqrt();
      lower_mat.col(j) = predmean.col(j) - sqrt(df/(df-2.0))*sdmat.col(j) * R::qt(0.975, n-p, true, false);
      upper_mat.col(j) = predmean.col(j) + sqrt(df/(df-2.0))*sdmat.col(j) * R::qt(0.975, n-p, true, false);      
    }
    return Rcpp::List::create(Rcpp::_["mean"] = predmean, 
                              Rcpp::_["sd"] = sdmat,
                              Rcpp::_["lower95"] = lower_mat,
                              Rcpp::_["upper95"] = upper_mat,
                              Rcpp::_["df"] = df,
                              Rcpp::_["input_corr"] = pred_corr,
                              Rcpp::_["output_cov"] = sig2hat
    );


}
/*****************************************************************************************/
/*****************************************************************************************/
// Simulate from the Predictive distribution 
Rcpp::List SP::simulate_predictive_dist(const Eigen::MatrixXd& y, 
  const Eigen::MatrixXd& H, const Eigen::MatrixXd& input,
  const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew, 
  const Eigen::VectorXd& range, const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, 
  const Eigen::VectorXd& nugget, const Rcpp::List& covmodel, const std::string& dtype){

  int n = y.rows();
  int q = y.cols();
  int m = input_new.rows();
  int p = H.cols();

  int nsample = range.rows();

  //int dim = input.cols();
  Eigen::MatrixXd R(n,n), RInv(n,n), RH(n,p), HRH(p,p), Rnew(n,m), HRHInv(p,p);
  Eigen::MatrixXd res(n,q), Ry(n,q), bhat(p,q);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH, ldlt;

  Eigen::MatrixXd sig2hat(q,q), predmean(m, q);

  Eigen::MatrixXd d = SP::pdist(input, input, dtype);
  Eigen::MatrixXd d0 = SP::pdist(input, input_new, dtype);

  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  double df = n-p;
  Eigen::VectorXd Rtmp(n), tmp(p), pred_corr(m);
  Eigen::MatrixXd ysim(m,q), L(q,q); 
  Rcpp::List ysim_sample(nsample);

#ifdef USE_R
    GetRNGstate();
#endif

  Progress prog(nsample, true);
  for(int it=0; it<nsample; it++){
    if(Progress::check_abort()){
      return R_NilValue;
    }
    prog.increment();

    R = SP::iso_kernel(d, range(it), tail(it), nu(it), family);
    Rnew = SP::iso_kernel(d0, range(it), tail(it), nu(it), family);

    R.diagonal().array() += nugget(it);
    for(int i=0; i<n; i++){
      for(int j=0; j<m; j++){
        if(d0(i,j)==0){
          Rnew(i,j) += nugget(it);
        }
      }
    }

    ldltR.compute(R);
    RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
    RH = RInv * H;
    HRH = H.transpose() * RH;
    ldltH.compute(HRH);
    Ry = RInv*y;

    bhat = ldltH.solve(H.transpose()*Ry);
    res = y - H*bhat;
    predmean = Hnew*bhat;
    predmean += Rnew.transpose()*(RInv*res);
    sig2hat = res.transpose() * RInv*res / df;

    HRHInv = ldltH.solve(Eigen::MatrixXd::Identity(p,p));

  // simulate from posterior predictive distribution
    for(int k=0; k<m; k++){
      Rtmp = Rnew.col(k);
      tmp = Hnew.row(k) - RH.transpose()*Rtmp;
      pred_corr(k) = R(0,0) - Rtmp.dot(RInv*Rtmp) + tmp.dot(HRHInv*tmp);
    }    

    L = sig2hat.llt().matrixL();
    for(int k=0; k<m; k++){
      ysim.row(k) =  sqrt(pred_corr(k)) * L * Rcpp::as<Eigen::VectorXd>(Rcpp::rnorm(q, 0.0, 1.0))
                  / sqrt(Rcpp::as<double>(Rcpp::rchisq(1, df)) / df);
    }
    ysim += predmean;

    ysim_sample[it] = ysim;
  }


#ifdef USE_R
  PutRNGstate();
#endif 


  return ysim_sample; 


}

/*****************************************************************************************/
/*****************************************************************************************/
// Conditional Simulation
Rcpp::List SP::condsim(const Eigen::MatrixXd& y, 
  const Eigen::MatrixXd& H, const Eigen::MatrixXd& input,
  const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew, 
  const double& range, const double& tail, const double& nu, 
  const double& nugget, const Rcpp::List& covmodel, const std::string& dtype, int nsample){

  int n = y.rows();
  int q = y.cols();
  int m = input_new.rows();
  int p = H.cols();


  //int dim = input.cols();
  Eigen::MatrixXd R(n,n), RInv(n,n), RH(n,p), HRH(p,p), Rnew(n,m), HRHInv(p,p);
  Eigen::MatrixXd res(n,q), Ry(n,q), bhat(p,q);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH, ldlt;

  Eigen::MatrixXd sig2hat(q,q), predmean(m, q);

  Eigen::MatrixXd d = SP::pdist(input, input, dtype);
  Eigen::MatrixXd d0 = SP::pdist(input, input_new, dtype);

  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  double df = n-p;
  Eigen::VectorXd Rtmp(n), tmp(p), pred_corr(m);
  Eigen::MatrixXd ysim(m,q), L(q,q); 
  Rcpp::List ysim_sample(nsample);

#ifdef USE_R
    GetRNGstate();
#endif


    R = SP::iso_kernel(d, range, tail, nu, family);
    R.diagonal().array() += nugget;
    Rnew = SP::iso_kernel(d0, range, tail, nu, family);

    for(int i=0; i<n; i++){
      for(int j=0; j<m; j++){
        if(d0(i,j)==0){
          Rnew(i,j) += nugget;
        }
      }
    }
    

    ldltR.compute(R);
    RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
    RH = RInv * H;
    HRH = H.transpose() * RH;
    ldltH.compute(HRH);
    Ry = RInv*y;

    bhat = ldltH.solve(H.transpose()*Ry);
    res = y - H*bhat;
    predmean = Hnew*bhat;
    predmean += Rnew.transpose()*(RInv*res);
    sig2hat = res.transpose() * RInv*res / df;

    HRHInv = ldltH.solve(Eigen::MatrixXd::Identity(p,p));

  // simulate from posterior predictive distribution
    for(int k=0; k<m; k++){
      Rtmp = Rnew.col(k);
      tmp = Hnew.row(k) - RH.transpose()*Rtmp;
      pred_corr(k) = R(0,0) - Rtmp.dot(RInv*Rtmp) + tmp.dot(HRHInv*tmp);
    }    

    L = sig2hat.llt().matrixL();

// simulate from posterior predictive distribution
  // Progress prog(nsample, true);
  for(int it=0; it<nsample; it++){
    // if(Progress::check_abort()){
    //   return R_NilValue;
    // }
    // prog.increment();

    for(int k=0; k<m; k++){
      ysim.row(k) =  sqrt(pred_corr(k)) * L * Rcpp::as<Eigen::VectorXd>(Rcpp::rnorm(q, 0.0, 1.0))
                  / sqrt(Rcpp::as<double>(Rcpp::rchisq(1, df)) / df);
    }
    ysim += predmean;

    ysim_sample[it] = ysim;
  }


#ifdef USE_R
  PutRNGstate();
#endif 


  return ysim_sample; 


}

/*****************************************************************************************/
/*****************************************************************************************/

// MCMC for isotropic covariance kernels with Random Walk on unconstrained parameter space
Rcpp::List SP::iso_MCMCOBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
  const Eigen::MatrixXd& input, const double& range, const double& tail, const double& nu, 
  const double& nugget, const Rcpp::List& covmodel, const bool& smoothness_est, const Rcpp::List& proposal,
  const int& nsample, const std::string& dtype, const bool& verbose){


  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  Eigen::MatrixXd d = SP::pdist(input, input, dtype);

  double Delta_range;
  if(proposal.containsElementNamed("range")){
    Delta_range = Rcpp::as<double>(proposal["range"]);
  }else{
    Delta_range = 0.1;
  }
  double Delta_tail=0.1, Delta_nugget=0.1, Delta_nu=0.1; // sd in the LN proposal distribution.

  if(proposal.containsElementNamed("tail")){
    Delta_tail = Rcpp::as<double>(proposal["tail"]);
  }else{
    Delta_tail = 0.1;
  }

  if(proposal.containsElementNamed("nugget")){
    Delta_nugget = Rcpp::as<double>(proposal["nugget"]);
  }else{
    Delta_nugget = 0.1;
  }
  
  if(proposal.containsElementNamed("nu")){
    Delta_nu = Rcpp::as<double>(proposal["nu"]);
  }else{
    Delta_nu = 0.1;
  }

  double loglik_curr = SP::MLoglik(range, tail, nu, nugget, output, H, d, covmodel);
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

  if(family=="cauchy" || family=="powexp"){
    nu_ub = 2.0;
  }

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
      loglik_prop = SP::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        range_sample[it] = range_curr;
        // accept_rate_range(it) = FALSE;
      }else{
        // log prior density of cauchy dist
        log_prior_curr = -log(1.0+range_curr*range_curr);
        log_prior_prop = -log(1.0+range_prop*range_prop);

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
      loglik_prop = SP::MLoglik(range_curr, tail_prop, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        tail_sample(it) = tail_curr;
        // accept_rate_tail[it]= FALSE;
      }else{
        // log prior density of cauchy dist
        log_prior_curr = -log(1.0+tail_curr*tail_curr);
        log_prior_prop = -log(1.0+tail_prop*tail_prop);

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
        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nugget_sample(it) = nugget_curr;
          // accept_rate_nugget[it]= FALSE;
        }else{
          // log prior density of cauchy dist
          log_prior_curr = -log(1.0+nugget_curr*nugget_curr);
          log_prior_prop = -log(1.0+nugget_prop*nugget_prop);

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

        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nu_sample(it) = nu_curr;
          // accept_rate_nu[it]= FALSE;
        }else{
          // log prior density 
          // log_prior_curr = -log(1.0+nu_curr*nu_curr);
          // log_prior_prop = -log(1.0+nu_prop*nu_prop);
          log_prior_curr = 0;
          log_prior_prop = 0;

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

    } 
  }else if(family=="cauchy"){ 
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
      loglik_prop = SP::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        range_sample[it] = range_curr;
        // accept_rate_range(it) = FALSE;
      }else{
        // log prior density of cauchy dist
        log_prior_curr = -log(1.0+range_curr*range_curr);
        log_prior_prop = -log(1.0+range_prop*range_prop);

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
      loglik_prop = SP::MLoglik(range_curr, tail_prop, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        tail_sample(it) = tail_curr;
        // accept_rate_tail[it]= FALSE;
      }else{
        // log prior density of cauchy dist
        log_prior_curr = -log(1.0+tail_curr*tail_curr);
        log_prior_prop = -log(1.0+tail_prop*tail_prop);

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
        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nugget_sample(it) = nugget_curr;
          // accept_rate_nugget[it]= FALSE;
        }else{
          // log prior density of cauchy dist
          log_prior_curr = -log(1.0+nugget_curr*nugget_curr);
          log_prior_prop = -log(1.0+nugget_prop*nugget_prop);

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
        nu_prop = ilogit(Rcpp::rnorm(1, logit(nu_curr, nu_lb, 2.0), Delta_nu)[0], nu_lb, 2.0);   

        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nu_sample(it) = nu_curr;
          // accept_rate_nu[it]= FALSE;
        }else{
          // log prior density 
          // log_prior_curr = -log(1.0+nu_curr*nu_curr);
          // log_prior_prop = -log(1.0+nu_prop*nu_prop);
          log_prior_curr = 0;
          log_prior_prop = 0;

          // log proposal density 
          log_pr_tran = 0;
          log_pr_rtran = 0;

          // Jacobian 
          // Jacobian_curr = log(nu_curr);
          // Jacobian_prop = log(nu_prop);
          Jacobian_curr = log(nu_curr-nu_lb) + log(2.0-nu_curr);
          Jacobian_prop = log(nu_prop-nu_lb) + log(2.0-nu_prop);

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

    } 


  }else if(family=="matern" || family=="powexp" || family=="gauss") {

    // begin MCMC sampling 
    for(int it=0; it<nsample; it++){
      if(Progress::check_abort()){
        return R_NilValue;
      }
      prog.increment();

      // loglik_curr = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // update range
      // generate proposal
      range_prop = exp(Rcpp::rnorm(1, log(range_curr), Delta_range)[0]);
      loglik_prop = SP::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        range_sample(it) = range_curr;
        // accept_rate_range[it]= FALSE;
      }else{

        // log prior density of cauchy dist
        log_prior_curr = -log(1.0+range_curr*range_curr);
        log_prior_prop = -log(1.0+range_prop*range_prop);

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
      // if(nugget_est){

        // generate proposal
        nugget_prop = exp(Rcpp::rnorm(1, log(nugget_curr), Delta_nugget)[0]);
        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);
        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nugget_sample(it) = nugget_curr;
          // accept_rate_nugget[it]= FALSE;
        }else{
          // log prior density of cauchy dist
          log_prior_curr = -log(1.0+nugget_curr*nugget_curr);
          log_prior_prop = -log(1.0+nugget_prop*nugget_prop);

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
            accept_rate_nugget[it] = TRUE;
            nugget_curr = nugget_prop;
            loglik_curr = loglik_prop;
          }
          nugget_sample(it) = nugget_curr;
        }

      // } 

      if(smoothness_est){

        // generate proposal
        // nu_prop = exp(Rcpp::rnorm(1, log(nu_curr), Delta_nu)[0]);
        nu_prop = ilogit(Rcpp::rnorm(1, logit(nu_curr, nu_lb, nu_ub), Delta_nu)[0], nu_lb, nu_ub);   

        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nu_sample(it) = nu_curr;
          // accept_rate_nu[it]= FALSE;
        }else{
          // log prior density of cauchy dist
          // log_prior_curr = -log(1.0+nu_curr*nu_curr);
          // log_prior_prop = -log(1.0+nu_prop*nu_prop);
          log_prior_curr = 0;
          log_prior_prop = 0;
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
            accept_rate_nu[it] = TRUE;
            nu_curr = nu_prop;
            loglik_curr = loglik_prop;
          }

          nu_sample(it) = nu_curr;
        } 
      }
    }    

  }else{
    Rcpp::stop("The family of covariance functions is not implemented yet.\n");
  }


/****************************************************************************/

  #ifdef USE_R
    PutRNGstate();
  #endif

  if(family=="CH"||family=="cauchy"){

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

  }else if(family=="matern" || family=="powexp" || family=="gauss"){

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
    Rcpp::stop("The family of covariance functions is not implemented yet.\n");
  }

}
/*******************************************************************************/
/*******************************************************************************/


// MCMC for isotropic covariance kernels with Random Walk on unconstrained parameter space
Rcpp::List SP::iso_MCMCSBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
  const Eigen::MatrixXd& input, const double& range, const double& tail, const double& nu, 
  const double& nugget, const Rcpp::List& covmodel, const bool& smoothness_est, const Rcpp::List& prior,
  const Rcpp::List& proposal, const int& nsample, const std::string& dtype, const bool& verbose){


  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  Eigen::MatrixXd d = SP::pdist(input, input, dtype);

  // tuning parameters in proposals 
  double Delta_range;
  if(proposal.containsElementNamed("range")){
    Delta_range = Rcpp::as<double>(proposal["range"]);
  }else{
    Delta_range = 0.1;
  }
  double Delta_tail=0.1, Delta_nugget=0.1, Delta_nu=0.1; // sd in the LN proposal distribution.

  if(proposal.containsElementNamed("tail")){
    Delta_tail = Rcpp::as<double>(proposal["tail"]);
  }else{
    Delta_tail = 0.1;
  }

  if(proposal.containsElementNamed("nugget")){
    Delta_nugget = Rcpp::as<double>(proposal["nugget"]);
  }else{
    Delta_nugget = 0.1;
  }
  
  if(proposal.containsElementNamed("nu")){
    Delta_nu = Rcpp::as<double>(proposal["nu"]);
  }else{
    Delta_nu = 0.1;
  }

  // tuning parameters in priors
  Rcpp::List range_prior; //beta(a, b, lb, ub) 
  double range_a=1.0, range_b=1.0;
  double range_lb=0, range_ub;
  range_ub = 3.0*d.maxCoeff(); 
  if(prior.containsElementNamed("range")){
    range_prior = Rcpp::as<Rcpp::List>(prior["range"]);
    if(range_prior.containsElementNamed("a")){
      range_a = Rcpp::as<double>(range_prior["a"]);
    }
    if(range_prior.containsElementNamed("b")){
      range_b = Rcpp::as<double>(range_prior["b"]);
    }
    if(range_prior.containsElementNamed("lb")){
      range_lb = Rcpp::as<double>(range_prior["lb"]);
    }
    if(range_prior.containsElementNamed("ub")){
      range_ub = Rcpp::as<double>(range_prior["ub"]);
    }    
  }

  Rcpp::List tail_prior; //beta(a, b, lb, ub) 
  double tail_a=1.0, tail_b=1.0;
  double tail_lb=0, tail_ub;
  tail_ub = 10.0; 
  if(prior.containsElementNamed("tail")){
    tail_prior = Rcpp::as<Rcpp::List>(prior["tail"]);
    if(tail_prior.containsElementNamed("a")){
      tail_a = Rcpp::as<double>(tail_prior["a"]);
    }
    if(tail_prior.containsElementNamed("b")){
      tail_b = Rcpp::as<double>(tail_prior["b"]);
    }
    if(tail_prior.containsElementNamed("lb")){
      tail_lb = Rcpp::as<double>(tail_prior["lb"]);
    }
    if(tail_prior.containsElementNamed("ub")){
      tail_ub = Rcpp::as<double>(tail_prior["ub"]);
    }    
  }

  Rcpp::List nugget_prior; //beta(a, b, lb, ub) 
  double nugget_a=1.0, nugget_b=1.0;
  double nugget_lb=0, nugget_ub;
  nugget_ub = 20.0; 
  if(prior.containsElementNamed("nugget")){
    nugget_prior = Rcpp::as<Rcpp::List>(prior["nugget"]);
    if(nugget_prior.containsElementNamed("a")){
      nugget_a = Rcpp::as<double>(nugget_prior["a"]);
    }
    if(nugget_prior.containsElementNamed("b")){
      nugget_b = Rcpp::as<double>(nugget_prior["b"]);
    }
    if(nugget_prior.containsElementNamed("lb")){
      nugget_lb = Rcpp::as<double>(nugget_prior["lb"]);
    }
    if(nugget_prior.containsElementNamed("ub")){
      nugget_ub = Rcpp::as<double>(nugget_prior["ub"]);
    }    
  }

  Rcpp::List nu_prior; //beta(a, b, lb, ub) 
  double nu_a=1.0, nu_b=1.0;
  double nu_lb=0.1, nu_ub=4.0;

  if(family=="cauchy" || family=="powexp"){
    nu_ub = 2.0;
  }

  if(prior.containsElementNamed("nu")){
    nu_prior = Rcpp::as<Rcpp::List>(prior["nu"]);
    if(nu_prior.containsElementNamed("a")){
      nu_a = Rcpp::as<double>(nu_prior["a"]);
    }
    if(nu_prior.containsElementNamed("b")){
      nu_b = Rcpp::as<double>(nu_prior["b"]);
    }
    if(nu_prior.containsElementNamed("lb")){
      nu_lb = Rcpp::as<double>(nu_prior["lb"]);
    }
    if(nu_prior.containsElementNamed("ub")){
      nu_ub = Rcpp::as<double>(nu_prior["ub"]);
      if(family=="cauchy" || family=="powexp"){
        if(nu_ub>2.0){
          nu_ub = 2.0;
        }
      }
    }    
  }


  double loglik_curr = SP::MLoglik(range, tail, nu, nugget, output, H, d, covmodel);
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
      range_prop = ilogit(Rcpp::rnorm(1, logit(range_curr, range_lb, range_ub), Delta_range)[0], range_lb, range_ub);
      loglik_prop = SP::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        range_sample(it) = range_curr;
        // accept_rate_range[it]= FALSE;
      }else{
        // log prior density of beta distribution 
        log_prior_curr = (range_a-1.0)*log(range_curr-range_lb) + (range_b-1.0)*log(range_ub-range_curr);
        log_prior_prop = (range_a-1.0)*log(range_prop-range_lb) + (range_b-1.0)*log(range_ub-range_prop);

        // difference of log proposal density is zero because RW normal propsal is used
        log_pr_tran =  0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(range_curr-range_lb) + log(range_ub-range_curr);
        Jacobian_prop = log(range_prop-range_lb) + log(range_ub-range_prop);

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
      tail_prop = ilogit(Rcpp::rnorm(1, logit(tail_curr, tail_lb, tail_ub), Delta_tail)[0], tail_lb, tail_ub);
      loglik_prop = SP::MLoglik(range_curr, tail_prop, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        tail_sample(it) = tail_curr;
        // accept_rate_tail[it]= FALSE;
      }else{
        // log prior density of cauchy dist
        log_prior_curr = (tail_a-1.0)*log(tail_curr-tail_lb) + (tail_b-1.0)*log(tail_ub-tail_curr);
        log_prior_prop = (tail_a-1.0)*log(tail_prop-tail_lb) + (tail_b-1.0)*log(tail_ub-tail_prop);

        // log proposal density 
        log_pr_tran = 0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(tail_curr-tail_lb) + log(tail_ub-tail_curr);
        Jacobian_prop = log(tail_prop-tail_lb) + log(tail_ub-tail_prop);

        MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                      + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_tail[it] = TRUE;
          tail_curr = tail_prop;
          loglik_curr = loglik_prop;
        }

        tail_sample(it) = tail_curr;
      
      }

        // generate proposal
        nugget_prop = ilogit(Rcpp::rnorm(1, logit(nugget_curr, nugget_lb, nugget_ub), Delta_nugget)[0], nugget_lb, nugget_ub);
        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        nugget_sample(it) = nugget_curr;
        // accept_rate_nugget[it]= FALSE;
      }else{
        // log prior density of cauchy dist
        log_prior_curr = (nugget_a-1.0)*log(nugget_curr-nugget_lb) + (nugget_b-1.0)*log(nugget_ub-nugget_curr);
        log_prior_prop = (nugget_a-1.0)*log(nugget_prop-nugget_lb) + (nugget_b-1.0)*log(nugget_ub-nugget_prop);

        // log proposal density 
        log_pr_tran = 0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(nugget_curr-nugget_lb) + log(nugget_ub-nugget_curr);
        Jacobian_prop = log(nugget_prop-nugget_lb) + log(nugget_ub-nugget_prop);

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
        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          range_sample(it) = range_curr;
          accept_rate_range[it]= FALSE;
        }else{
        // generate proposal
        nu_prop = ilogit(Rcpp::rnorm(1, logit(nu_curr, nu_lb, nu_ub), Delta_nu)[0], nu_lb, nu_ub);
        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nu_sample(it) = nu_curr;
          // accept_rate_nu[it]= FALSE;
        }else{
            // log prior density of beta dist
            log_prior_curr = (nu_a-1.0)*log(nu_curr-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_curr);
            log_prior_prop = (nu_a-1.0)*log(nu_prop-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_prop);

            // log proposal density 
            log_pr_tran = 0;
            log_pr_rtran = 0;

            // Jacobian 
            Jacobian_curr = log(nu_curr-nu_lb) + log(nu_ub-nu_curr);
            Jacobian_prop = log(nu_prop-nu_lb) + log(nu_ub-nu_prop);

            MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                            + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

            unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
            if(log(unif_rnd)<MH_ratio){ // accept 
              accept_rate_nu[it] = TRUE;
              nu_curr = nu_prop;
              loglik_curr = loglik_prop;
            }

            nu_sample(it) = nu_curr;
          }
        }
      }     

    } 

  }else if(family=="cauchy"){ 

    // begin MCMC sampling 
    for(int it=0; it<nsample; it++){
      if(Progress::check_abort()){
        return R_NilValue;
      }
      prog.increment();

      // update range 
      // loglik_curr = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // generate proposal
      range_prop = ilogit(Rcpp::rnorm(1, logit(range_curr, range_lb, range_ub), Delta_range)[0], range_lb, range_ub);
      loglik_prop = SP::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        range_sample(it) = range_curr;
        // accept_rate_range[it]= FALSE;
      }else{
        // log prior density of beta distribution 
        log_prior_curr = (range_a-1.0)*log(range_curr-range_lb) + (range_b-1.0)*log(range_ub-range_curr);
        log_prior_prop = (range_a-1.0)*log(range_prop-range_lb) + (range_b-1.0)*log(range_ub-range_prop);

        // difference of log proposal density is zero because RW normal propsal is used
        log_pr_tran =  0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(range_curr-range_lb) + log(range_ub-range_curr);
        Jacobian_prop = log(range_prop-range_lb) + log(range_ub-range_prop);

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
      tail_prop = ilogit(Rcpp::rnorm(1, logit(tail_curr, tail_lb, tail_ub), Delta_tail)[0], tail_lb, tail_ub);
      loglik_prop = SP::MLoglik(range_curr, tail_prop, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        tail_sample(it) = tail_curr;
        // accept_rate_tail[it]= FALSE;
      }else{
        // log prior density of cauchy dist
        log_prior_curr = (tail_a-1.0)*log(tail_curr-tail_lb) + (tail_b-1.0)*log(tail_ub-tail_curr);
        log_prior_prop = (tail_a-1.0)*log(tail_prop-tail_lb) + (tail_b-1.0)*log(tail_ub-tail_prop);

        // log proposal density 
        log_pr_tran = 0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(tail_curr-tail_lb) + log(tail_ub-tail_curr);
        Jacobian_prop = log(tail_prop-tail_lb) + log(tail_ub-tail_prop);

        MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                      + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_tail[it] = TRUE;
          tail_curr = tail_prop;
          loglik_curr = loglik_prop;
        }

        tail_sample(it) = tail_curr;
      
      }

        // generate proposal
        nugget_prop = ilogit(Rcpp::rnorm(1, logit(nugget_curr, nugget_lb, nugget_ub), Delta_nugget)[0], nugget_lb, nugget_ub);
        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        nugget_sample(it) = nugget_curr;
        // accept_rate_nugget[it]= FALSE;
      }else{
        // log prior density of cauchy dist
        log_prior_curr = (nugget_a-1.0)*log(nugget_curr-nugget_lb) + (nugget_b-1.0)*log(nugget_ub-nugget_curr);
        log_prior_prop = (nugget_a-1.0)*log(nugget_prop-nugget_lb) + (nugget_b-1.0)*log(nugget_ub-nugget_prop);

        // log proposal density 
        log_pr_tran = 0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(nugget_curr-nugget_lb) + log(nugget_ub-nugget_curr);
        Jacobian_prop = log(nugget_prop-nugget_lb) + log(nugget_ub-nugget_prop);

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
        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          range_sample(it) = range_curr;
          accept_rate_range[it]= FALSE;
        }else{
        // generate proposal
        nu_prop = ilogit(Rcpp::rnorm(1, logit(nu_curr, nu_lb, nu_ub), Delta_nu)[0], nu_lb, nu_ub);
        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nu_sample(it) = nu_curr;
          // accept_rate_nu[it]= FALSE;
        }else{
            // log prior density of beta dist
            log_prior_curr = (nu_a-1.0)*log(nu_curr-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_curr);
            log_prior_prop = (nu_a-1.0)*log(nu_prop-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_prop);

            // log proposal density 
            log_pr_tran = 0;
            log_pr_rtran = 0;

            // Jacobian 
            Jacobian_curr = log(nu_curr-nu_lb) + log(nu_ub-nu_curr);
            Jacobian_prop = log(nu_prop-nu_lb) + log(nu_ub-nu_prop);

            MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                            + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

            unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
            if(log(unif_rnd)<MH_ratio){ // accept 
              accept_rate_nu[it] = TRUE;
              nu_curr = nu_prop;
              loglik_curr = loglik_prop;
            }

            nu_sample(it) = nu_curr;
          }
        }
      }     

    } 


  }else if(family=="matern" || family=="powexp" || family=="gauss") {

    // begin MCMC sampling 
    for(int it=0; it<nsample; it++){
      if(Progress::check_abort()){
        return R_NilValue;
      }
      prog.increment();

      // loglik_curr = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // update range
      // generate proposal
      range_prop = ilogit(Rcpp::rnorm(1, logit(range_curr, range_lb, range_ub), Delta_range)[0], range_lb, range_ub);
      loglik_prop = SP::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // log prior density of beta distribution 
      log_prior_curr = (range_a-1.0)*log(range_curr-range_lb) + (range_b-1.0)*log(range_ub-range_curr);
      log_prior_prop = (range_a-1.0)*log(range_prop-range_lb) + (range_b-1.0)*log(range_ub-range_prop);

      // difference of log proposal density is zero because RW normal propsal is used
      log_pr_tran =  0;
      log_pr_rtran = 0;

      // Jacobian 
      Jacobian_curr = log(range_curr-range_lb) + log(range_ub-range_curr);
      Jacobian_prop = log(range_prop-range_lb) + log(range_ub-range_prop);

      MH_ratio = (loglik_prop - loglik_curr) + (log_prior_prop-log_prior_curr) 
                      + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

      unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
      if(log(unif_rnd)<MH_ratio){ // accept 
        accept_rate_range[it] = TRUE;
        range_curr = range_prop;
        loglik_curr = loglik_prop;
      }

      range_sample(it) = range_curr;

      // if(nugget_est){

        // generate proposal
        nugget_prop = ilogit(Rcpp::rnorm(1, logit(nugget_curr, nugget_lb, nugget_ub), Delta_nugget)[0], nugget_lb, nugget_ub);
        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

        // log prior density of beta dist
        log_prior_curr = (nugget_a-1.0)*log(nugget_curr-nugget_lb) + (nugget_b-1.0)*log(nugget_ub-nugget_curr);
        log_prior_prop = (nugget_a-1.0)*log(nugget_prop-nugget_lb) + (nugget_b-1.0)*log(nugget_ub-nugget_prop);

        // log proposal density 
        log_pr_tran = 0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(nugget_curr-tail_lb) + log(nugget_ub-nugget_curr);
        Jacobian_prop = log(nugget_prop-tail_lb) + log(nugget_ub-nugget_prop);


        MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_nugget[it] = TRUE;
          nugget_curr = nugget_prop;
          loglik_curr = loglik_prop;
        }

        nugget_sample(it) = nugget_curr;
      // } 

      if(smoothness_est){

        // generate proposal
        nu_prop = ilogit(Rcpp::rnorm(1, logit(nu_curr, nu_lb, nu_ub), Delta_nu)[0], nu_lb, nu_ub);
        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

        // log prior density of cauchy dist
        log_prior_curr = (nu_a-1.0)*log(nu_curr-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_curr);
        log_prior_prop = (nu_a-1.0)*log(nu_prop-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_prop);

        // log proposal density 
        log_pr_tran = 0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(nu_curr-nu_lb) + log(nu_ub-nu_curr);
        Jacobian_prop = log(nu_prop-nu_lb) + log(nu_ub-nu_prop);

        MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_nu[it] = TRUE;
          nu_curr = nu_prop;
          loglik_curr = loglik_prop;
        }

        nu_sample(it) = nu_curr;
      } 
    }    

  }else{
    Rcpp::stop("The family of covariance functions is not implemented yet.\n");
  }


/****************************************************************************/

  #ifdef USE_R
    PutRNGstate();
  #endif

  if(family=="CH"||family=="cauchy"){

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

  }else if(family=="matern" || family=="powexp" || family=="gauss"){

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
    Rcpp::stop("The family of covariance functions is not implemented yet.\n");
  }

}
/*******************************************************************************/
/*******************************************************************************/


// MCMC estimation + prediction for isotropic covariance kernels with Random Walk on unconstrained parameter space
Rcpp::List SP::iso_MCMCOBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
  const Eigen::MatrixXd& input, const double& range, const double& tail, const double& nu, 
  const double& nugget, const Rcpp::List& covmodel, const bool& smoothness_est, const Rcpp::List& proposal,
  const int& nsample, const std::string& dtype, const bool& verbose, const Eigen::MatrixXd& input_new, 
  const Eigen::MatrixXd& Hnew){


  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  Eigen::MatrixXd d = SP::pdist(input, input, dtype);

  double Delta_range;
  if(proposal.containsElementNamed("range")){
    Delta_range = Rcpp::as<double>(proposal["range"]);
  }else{
    Delta_range = 0.1;
  }
  double Delta_tail=0.1, Delta_nugget=0.1, Delta_nu=0.1; // sd in the LN proposal distribution.

  if(proposal.containsElementNamed("tail")){
    Delta_tail = Rcpp::as<double>(proposal["tail"]);
  }else{
    Delta_tail = 0.1;
  }

  if(proposal.containsElementNamed("nugget")){
    Delta_nugget = Rcpp::as<double>(proposal["nugget"]);
  }else{
    Delta_nugget = 0.1;
  }
  
  if(proposal.containsElementNamed("nu")){
    Delta_nu = Rcpp::as<double>(proposal["nu"]);
  }else{
    Delta_nu = 0.1;
  }

  double loglik_curr = SP::MLoglik(range, tail, nu, nugget, output, H, d, covmodel);
  double loglik_prop=0.0;
  
  // initial values for parameters 
  double range_curr, tail_curr, nu_curr, nugget_curr;
  range_curr = range;
  tail_curr = tail;
  nu_curr = nu; 
  nugget_curr = nugget;

  Eigen::VectorXd range_sample(nsample);
  Eigen::VectorXd tail_sample(nsample);
  Eigen::VectorXd nu_sample = nu_curr*Eigen::VectorXd::Ones(nsample);
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

  if(family=="cauchy" || family=="powexp"){
    nu_ub = 2.0;
  }

  /*********************** quantities related to prediction ****************/
  int n = output.rows();
  int q = output.cols();
  int m = input_new.rows();
  int p = H.cols();
  //int dim = input.cols();
  Eigen::MatrixXd R(n,n), RInv(n,n), RH(n,p), HRH(p,p), Rnew(n,m), HRHInv(p,p);
  Eigen::MatrixXd res(n,q), Ry(n,q), bhat(p,q);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH, ldlt;

  Eigen::MatrixXd sig2hat(q,q), predmean(m, q), L(q,q);

  //Eigen::MatrixXd d = SP::pdist(input, input, dtype);
  Eigen::MatrixXd d0 = SP::pdist(input, input_new, dtype);

  double df = n-p;
  Eigen::MatrixXd ysim(m,q);
  Eigen::VectorXd Rtmp(n), tmp(p), pred_corr(m);

  Rcpp::List ysim_sample(nsample);


  // create an empty list for abortion 
  // Rcpp::List abort_value = Rcpp::List::create(Rcpp::_["abort"]=R_NilValue);

  #ifdef USE_R
    GetRNGstate();
  #endif
  
  /****************************************************************************/
  Progress prog(nsample, verbose);
  if(family=="CH" || family=="cauchy"){ 

    // begin MCMC sampling 
    for(int it=0; it<nsample; it++){
      if(Progress::check_abort()){
        return R_NilValue;
      }
      prog.increment();

      /******************************* Parameter Estimation ***************************/
      // update range 
      // loglik_curr = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // generate proposal
      range_prop = exp(Rcpp::rnorm(1, log(range_curr), Delta_range)[0]);
      loglik_prop = SP::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        range_sample[it] = range_curr;
        // accept_rate_range(it) = FALSE;
      }else{
        // log prior density of cauchy dist
        log_prior_curr = -log(1.0+range_curr*range_curr);
        log_prior_prop = -log(1.0+range_prop*range_prop);

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
      loglik_prop = SP::MLoglik(range_curr, tail_prop, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        tail_sample(it) = tail_curr;
        // accept_rate_tail[it]= FALSE;
      }else{
        // log prior density of cauchy dist
        log_prior_curr = -log(1.0+tail_curr*tail_curr);
        log_prior_prop = -log(1.0+tail_prop*tail_prop);

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
        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nugget_sample(it) = nugget_curr;
          // accept_rate_nugget[it]= FALSE;
        }else{
          // log prior density of cauchy dist
          log_prior_curr = -log(1.0+nugget_curr*nugget_curr);
          log_prior_prop = -log(1.0+nugget_prop*nugget_prop);

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

        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nu_sample(it) = nu_curr;
          // accept_rate_nu[it]= FALSE;
        }else{
          // log prior density 
          // log_prior_curr = -log(1.0+nu_curr*nu_curr);
          // log_prior_prop = -log(1.0+nu_prop*nu_prop);
          log_prior_curr = 0;
          log_prior_prop = 0;

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

      /******************************************************************************/
      /******************************************************************************/
      
      /********************************* Prediction *********************************/ 
      //if(it>burnin){
        R = SP::iso_kernel(d, range_sample(it), tail_sample(it), nu_sample(it), family);
        Rnew = SP::iso_kernel(d0, range_sample(it), tail_sample(it), nu_sample(it), family);

        R.diagonal().array() += nugget_sample(it);

        ldltR.compute(R);
        RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
        RH = RInv * H;
        HRH = H.transpose() * RH;
        ldltH.compute(HRH);
        Ry = RInv*output;

        bhat = ldltH.solve(H.transpose()*Ry);
        res = output - H*bhat;
        predmean = Hnew*bhat;
        predmean += Rnew.transpose()*(RInv*res);
        sig2hat = res.transpose() * RInv*res / (n-p);

        HRHInv = ldltH.solve(Eigen::MatrixXd::Identity(p,p));

        for(int k=0; k<m; k++){
          Rtmp = Rnew.col(k);
          tmp = Hnew.row(k) - RH.transpose()*Rtmp;
          pred_corr(k) = R(0,0) - Rtmp.dot(RInv*Rtmp) + tmp.dot(HRHInv*tmp);
        }

        // simulate from posterior predictive distribution


        L = sig2hat.llt().matrixL(); // lower cholesky factor of sig2hat
        
        for(int k=0; k<m; k++){
          ysim.row(k) =  sqrt(pred_corr(k)) * L * Rcpp::as<Eigen::VectorXd>(Rcpp::rnorm(q, 0.0, 1.0))
                      / sqrt(Rcpp::as<double>(Rcpp::rchisq(1, df)) / df);
        }
        ysim += predmean;
        ysim_sample[it] = ysim;


      // // end prediction


      /******************************************************************************/
   

    } 


  }else if(family=="matern" || family=="powexp" || family=="gauss") {

    // begin MCMC sampling 
    for(int it=0; it<nsample; it++){
      if(Progress::check_abort()){
        return R_NilValue;
      }
      prog.increment();

      /******************************************************************************/
      /**************************** Parameter Estimation *******************************/
      // loglik_curr = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // update range
      // generate proposal
      range_prop = exp(Rcpp::rnorm(1, log(range_curr), Delta_range)[0]);
      loglik_prop = SP::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        range_sample(it) = range_curr;
        // accept_rate_range[it]= FALSE;
      }else{

        // log prior density of cauchy dist
        log_prior_curr = -log(1.0+range_curr*range_curr);
        log_prior_prop = -log(1.0+range_prop*range_prop);

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
      // if(nugget_est){

        // generate proposal
        nugget_prop = exp(Rcpp::rnorm(1, log(nugget_curr), Delta_nugget)[0]);
        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);
        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nugget_sample(it) = nugget_curr;
          // accept_rate_nugget[it]= FALSE;
        }else{
          // log prior density of cauchy dist
          log_prior_curr = -log(1.0+nugget_curr*nugget_curr);
          log_prior_prop = -log(1.0+nugget_prop*nugget_prop);

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
            accept_rate_nugget[it] = TRUE;
            nugget_curr = nugget_prop;
            loglik_curr = loglik_prop;
          }
          nugget_sample(it) = nugget_curr;
        }

      // } 

      if(smoothness_est){

        // generate proposal
        // nu_prop = exp(Rcpp::rnorm(1, log(nu_curr), Delta_nu)[0]);
        nu_prop = ilogit(Rcpp::rnorm(1, logit(nu_curr, nu_lb, nu_ub), Delta_nu)[0], nu_lb, nu_ub);   

        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nu_sample(it) = nu_curr;
          // accept_rate_nu[it]= FALSE;
        }else{
          // log prior density 
          // log_prior_curr = -log(1.0+nu_curr*nu_curr);
          // log_prior_prop = -log(1.0+nu_prop*nu_prop);
          log_prior_curr = 0;
          log_prior_prop = 0;

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
            accept_rate_nu[it] = TRUE;
            nu_curr = nu_prop;
            loglik_curr = loglik_prop;
          }

          nu_sample(it) = nu_curr;
        } 
      }

      /******************************************************************************/
      /******************************************************************************/
      
      /********************************* Prediction *********************************/ 
      //if(it>burnin){
        R = SP::iso_kernel(d, range_sample(it), tail_sample(it), nu_sample(it), family);
        Rnew = SP::iso_kernel(d0, range_sample(it), tail_sample(it), nu_sample(it), family);

        R.diagonal().array() += nugget_sample(it);

        ldltR.compute(R);
        RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
        RH = RInv * H;
        HRH = H.transpose() * RH;
        ldltH.compute(HRH);
        Ry = RInv*output;

        bhat = ldltH.solve(H.transpose()*Ry);
        res = output - H*bhat;
        predmean = Hnew*bhat;
        predmean += Rnew.transpose()*(RInv*res);
        sig2hat = res.transpose() * RInv*res / (n-p);

        HRHInv = ldltH.solve(Eigen::MatrixXd::Identity(p,p));

        for(int k=0; k<m; k++){
          Rtmp = Rnew.col(k);
          tmp = Hnew.row(k) - RH.transpose()*Rtmp;
          pred_corr(k) = R(0,0) - Rtmp.dot(RInv*Rtmp) + tmp.dot(HRHInv*tmp);
        }

        // simulate from posterior predictive distribution


        L = sig2hat.llt().matrixL(); // lower cholesky factor of sig2hat
        
        for(int k=0; k<m; k++){
          ysim.row(k) =  sqrt(pred_corr(k)) * L * Rcpp::as<Eigen::VectorXd>(Rcpp::rnorm(q, 0.0, 1.0))
                      / sqrt(Rcpp::as<double>(Rcpp::rchisq(1, df)) / df);
        }
        ysim += predmean;
        ysim_sample[it] = ysim;


      // // end prediction


      /******************************************************************************/

    }    

  }else{
    Rcpp::stop("The family of covariance functions is not implemented yet.\n");
  }


/****************************************************************************/

  #ifdef USE_R
    PutRNGstate();
  #endif

  if(family=="CH"||family=="cauchy"){

    if(smoothness_est){
      return Rcpp::List::create(Rcpp::_["range"]=range_sample, 
                                Rcpp::_["tail"]=tail_sample,
                                Rcpp::_["nugget"]=nugget_sample,
                                Rcpp::_["nu"]=nu_sample,
                                Rcpp::_["accept_rate_range"]=accept_rate_range,
                                Rcpp::_["accept_rate_tail"]=accept_rate_tail,
                                Rcpp::_["accept_rate_nugget"]=accept_rate_nugget,
                                Rcpp::_["accept_rate_nu"]=accept_rate_nu,
                                Rcpp::_["pred"] = ysim_sample
                                );      
    }else{
      return Rcpp::List::create(Rcpp::_["range"]=range_sample, 
                                Rcpp::_["tail"]=tail_sample,
                                Rcpp::_["nugget"]=nugget_sample,
                                Rcpp::_["accept_rate_range"]=accept_rate_range,
                                Rcpp::_["accept_rate_tail"]=accept_rate_tail,
                                Rcpp::_["accept_rate_nugget"]=accept_rate_nugget,
                                Rcpp::_["pred"] = ysim_sample);
    }

  }else if(family=="matern" || family=="powexp" || family=="gauss"){

    if(smoothness_est){
      return Rcpp::List::create(Rcpp::_["range"]=range_sample, 
                                Rcpp::_["nugget"]=nugget_sample,
                                Rcpp::_["nu"]=nu_sample,
                                Rcpp::_["accept_rate_range"]=accept_rate_range,
                                Rcpp::_["accept_rate_nugget"]=accept_rate_nugget,
                                Rcpp::_["accept_rate_nu"]=accept_rate_nu,
                                Rcpp::_["pred"] = ysim_sample
                                );      
    }else{
      return Rcpp::List::create(Rcpp::_["range"]=range_sample, 
                                Rcpp::_["nugget"]=nugget_sample,
                                Rcpp::_["accept_rate_range"]=accept_rate_range,
                                Rcpp::_["accept_rate_nugget"]=accept_rate_nugget,
                                Rcpp::_["pred"] = ysim_sample);
    }
  }else{
    Rcpp::stop("The family of covariance functions is not implemented yet.\n");
  }

}
/*******************************************************************************/
/*******************************************************************************/


// MCMC for isotropic covariance kernels with Random Walk on unconstrained parameter space
Rcpp::List SP::iso_MCMCSBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
  const Eigen::MatrixXd& input, const double& range, const double& tail, const double& nu, 
  const double& nugget, const Rcpp::List& covmodel, const bool& smoothness_est, const Rcpp::List& prior,
  const Rcpp::List& proposal, const int& nsample, const std::string& dtype, const bool& verbose, 
  const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew){


  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  Eigen::MatrixXd d = SP::pdist(input, input, dtype);

  // tuning parameters in proposals 
  double Delta_range;
  if(proposal.containsElementNamed("range")){
    Delta_range = Rcpp::as<double>(proposal["range"]);
  }else{
    Delta_range = 0.1;
  }
  double Delta_tail=0.1, Delta_nugget=0.1, Delta_nu=0.1; // sd in the LN proposal distribution.

  if(proposal.containsElementNamed("tail")){
    Delta_tail = Rcpp::as<double>(proposal["tail"]);
  }else{
    Delta_tail = 0.1;
  }

  if(proposal.containsElementNamed("nugget")){
    Delta_nugget = Rcpp::as<double>(proposal["nugget"]);
  }else{
    Delta_nugget = 0.1;
  }
  
  if(proposal.containsElementNamed("nu")){
    Delta_nu = Rcpp::as<double>(proposal["nu"]);
  }else{
    Delta_nu = 0.1;
  }

  // tuning parameters in priors
  Rcpp::List range_prior; //beta(a, b, lb, ub) 
  double range_a=1.0, range_b=1.0;
  double range_lb=0, range_ub;
  range_ub = 3.0*d.maxCoeff(); 
  if(prior.containsElementNamed("range")){
    range_prior = Rcpp::as<Rcpp::List>(prior["range"]);
    if(range_prior.containsElementNamed("a")){
      range_a = Rcpp::as<double>(range_prior["a"]);
    }
    if(range_prior.containsElementNamed("b")){
      range_b = Rcpp::as<double>(range_prior["b"]);
    }
    if(range_prior.containsElementNamed("lb")){
      range_lb = Rcpp::as<double>(range_prior["lb"]);
    }
    if(range_prior.containsElementNamed("ub")){
      range_ub = Rcpp::as<double>(range_prior["ub"]);
    }    
  }

  Rcpp::List tail_prior; //beta(a, b, lb, ub) 
  double tail_a=1.0, tail_b=1.0;
  double tail_lb=0, tail_ub;
  tail_ub = 5.0; 
  if(prior.containsElementNamed("tail")){
    tail_prior = Rcpp::as<Rcpp::List>(prior["tail"]);
    if(tail_prior.containsElementNamed("a")){
      tail_a = Rcpp::as<double>(tail_prior["a"]);
    }
    if(tail_prior.containsElementNamed("b")){
      tail_b = Rcpp::as<double>(tail_prior["b"]);
    }
    if(tail_prior.containsElementNamed("lb")){
      tail_lb = Rcpp::as<double>(tail_prior["lb"]);
    }
    if(tail_prior.containsElementNamed("ub")){
      tail_ub = Rcpp::as<double>(tail_prior["ub"]);
    }    
  }

  Rcpp::List nugget_prior; //beta(a, b, lb, ub) 
  double nugget_a=1.0, nugget_b=1.0;
  double nugget_lb=0, nugget_ub;
  nugget_ub = 20.0; 
  if(prior.containsElementNamed("nugget")){
    nugget_prior = Rcpp::as<Rcpp::List>(prior["nugget"]);
    if(nugget_prior.containsElementNamed("a")){
      nugget_a = Rcpp::as<double>(nugget_prior["a"]);
    }
    if(nugget_prior.containsElementNamed("b")){
      nugget_b = Rcpp::as<double>(nugget_prior["b"]);
    }
    if(nugget_prior.containsElementNamed("lb")){
      nugget_lb = Rcpp::as<double>(nugget_prior["lb"]);
    }
    if(nugget_prior.containsElementNamed("ub")){
      nugget_ub = Rcpp::as<double>(nugget_prior["ub"]);
    }    
  }

  Rcpp::List nu_prior; //beta(a, b, lb, ub) 
  double nu_a=1.0, nu_b=1.0;
  double nu_lb=0.1, nu_ub=4.0;
  if(family=="cauchy" || family=="powexp"){
    nu_ub = 2.0;
  }

  if(prior.containsElementNamed("nu")){
    nu_prior = Rcpp::as<Rcpp::List>(prior["nu"]);
    if(nu_prior.containsElementNamed("a")){
      nu_a = Rcpp::as<double>(nu_prior["a"]);
    }
    if(nu_prior.containsElementNamed("b")){
      nu_b = Rcpp::as<double>(nu_prior["b"]);
    }
    if(nu_prior.containsElementNamed("lb")){
      nu_lb = Rcpp::as<double>(nu_prior["lb"]);
    }
    if(nu_prior.containsElementNamed("ub")){
      nu_ub = Rcpp::as<double>(nu_prior["ub"]);
      if(family=="cauchy" || family=="powexp"){
        if(nu_ub>2.0){
          nu_ub = 2.0;
        }
      }
    }    
  }


  double loglik_curr = SP::MLoglik(range, tail, nu, nugget, output, H, d, covmodel);
  double loglik_prop=0.0;
  
  // initial values for parameters 
  double range_curr, tail_curr, nu_curr, nugget_curr;
  range_curr = range;
  tail_curr = tail;
  nu_curr = nu; 
  nugget_curr = nugget;

  Eigen::VectorXd range_sample(nsample);
  Eigen::VectorXd tail_sample(nsample);
  Eigen::VectorXd nu_sample=nu_curr*Eigen::VectorXd::Ones(nsample);
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

  /*********************** quantities related to prediction ****************/
  int n = output.rows();
  int q = output.cols();
  int m = input_new.rows();
  int p = H.cols();
  //int dim = input.cols();
  Eigen::MatrixXd R(n,n), RInv(n,n), RH(n,p), HRH(p,p), Rnew(n,m), HRHInv(p,p);
  Eigen::MatrixXd res(n,q), Ry(n,q), bhat(p,q);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH, ldlt;

  Eigen::MatrixXd sig2hat(q,q), predmean(m, q), L(q,q);

  // Eigen::MatrixXd d = SP::pdist(input, input, dtype);
  Eigen::MatrixXd d0 = SP::pdist(input, input_new, dtype);

  double df = n-p;
  Eigen::MatrixXd ysim(m,q);
  Eigen::VectorXd Rtmp(n), tmp(p), pred_corr(m);

  Rcpp::List ysim_sample(nsample);

  // create an empty list for abortion 
  // Rcpp::List abort_value = Rcpp::List::create(Rcpp::_["abort"]=R_NilValue);

  #ifdef USE_R
    GetRNGstate();
  #endif
  
  /****************************************************************************/
  Progress prog(nsample, verbose);
  if(family=="CH" || family=="cauchy"){ 

    // begin MCMC sampling 
    for(int it=0; it<nsample; it++){
      if(Progress::check_abort()){
        return R_NilValue;
      }
      prog.increment();

      /******************************************************************************/
      /******************************************************************************/

      // update range 
      // loglik_curr = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // generate proposal
      range_prop = ilogit(Rcpp::rnorm(1, logit(range_curr, range_lb, range_ub), Delta_range)[0], range_lb, range_ub);
      loglik_prop = SP::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        range_sample(it) = range_curr;
        // accept_rate_range[it]= FALSE;
      }else{
        // log prior density of beta distribution 
        log_prior_curr = (range_a-1.0)*log(range_curr-range_lb) + (range_b-1.0)*log(range_ub-range_curr);
        log_prior_prop = (range_a-1.0)*log(range_prop-range_lb) + (range_b-1.0)*log(range_ub-range_prop);

        // difference of log proposal density is zero because RW normal propsal is used
        log_pr_tran =  0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(range_curr-range_lb) + log(range_ub-range_curr);
        Jacobian_prop = log(range_prop-range_lb) + log(range_ub-range_prop);

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
      tail_prop = ilogit(Rcpp::rnorm(1, logit(tail_curr, tail_lb, tail_ub), Delta_tail)[0], tail_lb, tail_ub);
      loglik_prop = SP::MLoglik(range_curr, tail_prop, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        tail_sample(it) = tail_curr;
        // accept_rate_tail[it]= FALSE;
      }else{
        // log prior density of cauchy dist
        log_prior_curr = (tail_a-1.0)*log(tail_curr-tail_lb) + (tail_b-1.0)*log(tail_ub-tail_curr);
        log_prior_prop = (tail_a-1.0)*log(tail_prop-tail_lb) + (tail_b-1.0)*log(tail_ub-tail_prop);

        // log proposal density 
        log_pr_tran = 0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(tail_curr-tail_lb) + log(tail_ub-tail_curr);
        Jacobian_prop = log(tail_prop-tail_lb) + log(tail_ub-tail_prop);

        MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                      + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_tail[it] = TRUE;
          tail_curr = tail_prop;
          loglik_curr = loglik_prop;
        }

        tail_sample(it) = tail_curr;
      
      }

        // generate proposal
        nugget_prop = ilogit(Rcpp::rnorm(1, logit(nugget_curr, nugget_lb, nugget_ub), Delta_nugget)[0], nugget_lb, nugget_ub);
        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        nugget_sample(it) = nugget_curr;
        // accept_rate_nugget[it]= FALSE;
      }else{
        // log prior density of cauchy dist
        log_prior_curr = (nugget_a-1.0)*log(nugget_curr-nugget_lb) + (nugget_b-1.0)*log(nugget_ub-nugget_curr);
        log_prior_prop = (nugget_a-1.0)*log(nugget_prop-nugget_lb) + (nugget_b-1.0)*log(nugget_ub-nugget_prop);

        // log proposal density 
        log_pr_tran = 0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(nugget_curr-nugget_lb) + log(nugget_ub-nugget_curr);
        Jacobian_prop = log(nugget_prop-nugget_lb) + log(nugget_ub-nugget_prop);

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
        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          range_sample(it) = range_curr;
          // accept_rate_nu[it]= FALSE;
        }else{
        // generate proposal
        nu_prop = ilogit(Rcpp::rnorm(1, logit(nu_curr, nu_lb, nu_ub), Delta_nu)[0], nu_lb, nu_ub);
        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nu_sample(it) = nu_curr;
          // accept_rate_nu[it]= FALSE;
        }else{
            // log prior density of beta dist
            log_prior_curr = (nu_a-1.0)*log(nu_curr-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_curr);
            log_prior_prop = (nu_a-1.0)*log(nu_prop-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_prop);

            // log proposal density 
            log_pr_tran = 0;
            log_pr_rtran = 0;

            // Jacobian 
            Jacobian_curr = log(nu_curr-nu_lb) + log(nu_ub-nu_curr);
            Jacobian_prop = log(nu_prop-nu_lb) + log(nu_ub-nu_prop);

            MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                            + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

            unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
            if(log(unif_rnd)<MH_ratio){ // accept 
              accept_rate_nu[it] = TRUE;
              nu_curr = nu_prop;
              loglik_curr = loglik_prop;
            }

            nu_sample(it) = nu_curr;
          }
        }
      } 

      /******************************************************************************/
      /******************************************************************************/
      
      /********************************* Prediction *********************************/ 
      //if(it>burnin){
        R = SP::iso_kernel(d, range_sample(it), tail_sample(it), nu_sample(it), family);
        Rnew = SP::iso_kernel(d0, range_sample(it), tail_sample(it), nu_sample(it), family);

        R.diagonal().array() += nugget_sample(it);

        ldltR.compute(R);
        RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
        RH = RInv * H;
        HRH = H.transpose() * RH;
        ldltH.compute(HRH);
        Ry = RInv*output;

        bhat = ldltH.solve(H.transpose()*Ry);
        res = output - H*bhat;
        predmean = Hnew*bhat;
        predmean += Rnew.transpose()*(RInv*res);
        sig2hat = res.transpose() * RInv*res / (n-p);

        HRHInv = ldltH.solve(Eigen::MatrixXd::Identity(p,p));

        for(int k=0; k<m; k++){
          Rtmp = Rnew.col(k);
          tmp = Hnew.row(k) - RH.transpose()*Rtmp;
          pred_corr(k) = R(0,0) - Rtmp.dot(RInv*Rtmp) + tmp.dot(HRHInv*tmp);
        }

        // simulate from posterior predictive distribution


        L = sig2hat.llt().matrixL(); // lower cholesky factor of sig2hat
        
        for(int k=0; k<m; k++){
          ysim.row(k) =  sqrt(pred_corr(k)) * L * Rcpp::as<Eigen::VectorXd>(Rcpp::rnorm(q, 0.0, 1.0))
                      / sqrt(Rcpp::as<double>(Rcpp::rchisq(1, df)) / df);
        }
        ysim += predmean;
        ysim_sample[it] = ysim;


      // // end prediction


      /******************************************************************************/    

    } 


  }else if(family=="matern" || family=="powexp" || family=="gauss") {

    // begin MCMC sampling 
    for(int it=0; it<nsample; it++){
      if(Progress::check_abort()){
        return R_NilValue;
      }
      prog.increment();

      /******************************************************************************/
      /******************************************************************************/

      // loglik_curr = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // update range
      // generate proposal
      range_prop = ilogit(Rcpp::rnorm(1, logit(range_curr, range_lb, range_ub), Delta_range)[0], range_lb, range_ub);
      loglik_prop = SP::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // log prior density of beta distribution 
      log_prior_curr = (range_a-1.0)*log(range_curr-range_lb) + (range_b-1.0)*log(range_ub-range_curr);
      log_prior_prop = (range_a-1.0)*log(range_prop-range_lb) + (range_b-1.0)*log(range_ub-range_prop);

      // difference of log proposal density is zero because RW normal propsal is used
      log_pr_tran =  0;
      log_pr_rtran = 0;

      // Jacobian 
      Jacobian_curr = log(range_curr-range_lb) + log(range_ub-range_curr);
      Jacobian_prop = log(range_prop-range_lb) + log(range_ub-range_prop);

      MH_ratio = (loglik_prop - loglik_curr) + (log_prior_prop-log_prior_curr) 
                      + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

      unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
      if(log(unif_rnd)<MH_ratio){ // accept 
        accept_rate_range[it] = TRUE;
        range_curr = range_prop;
        loglik_curr = loglik_prop;
      }

      range_sample(it) = range_curr;

      // if(nugget_est){

        // generate proposal
        nugget_prop = ilogit(Rcpp::rnorm(1, logit(nugget_curr, nugget_lb, nugget_ub), Delta_nugget)[0], nugget_lb, nugget_ub);
        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

        // log prior density of beta dist
        log_prior_curr = (nugget_a-1.0)*log(nugget_curr-nugget_lb) + (nugget_b-1.0)*log(nugget_ub-nugget_curr);
        log_prior_prop = (nugget_a-1.0)*log(nugget_prop-nugget_lb) + (nugget_b-1.0)*log(nugget_ub-nugget_prop);

        // log proposal density 
        log_pr_tran = 0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(nugget_curr-tail_lb) + log(nugget_ub-nugget_curr);
        Jacobian_prop = log(nugget_prop-tail_lb) + log(nugget_ub-nugget_prop);


        MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_nugget[it] = TRUE;
          nugget_curr = nugget_prop;
          loglik_curr = loglik_prop;
        }

        nugget_sample(it) = nugget_curr;
      // } 

      if(smoothness_est){

        // generate proposal
        nu_prop = ilogit(Rcpp::rnorm(1, logit(nu_curr, nu_lb, nu_ub), Delta_nu)[0], nu_lb, nu_ub);
        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

        // log prior density of cauchy dist
        log_prior_curr = (nu_a-1.0)*log(nu_curr-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_curr);
        log_prior_prop = (nu_a-1.0)*log(nu_prop-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_prop);

        // log proposal density 
        log_pr_tran = 0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(nu_curr-nu_lb) + log(nu_ub-nu_curr);
        Jacobian_prop = log(nu_prop-nu_lb) + log(nu_ub-nu_prop);

        MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_nu[it] = TRUE;
          nu_curr = nu_prop;
          loglik_curr = loglik_prop;
        }

        nu_sample(it) = nu_curr;
      } 

      /******************************************************************************/
      /******************************************************************************/
      
      /********************************* Prediction *********************************/ 
      //if(it>burnin){
        R = SP::iso_kernel(d, range_sample(it), tail_sample(it), nu_sample(it), family);
        Rnew = SP::iso_kernel(d0, range_sample(it), tail_sample(it), nu_sample(it), family);

        R.diagonal().array() += nugget_sample(it);

        ldltR.compute(R);
        RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
        RH = RInv * H;
        HRH = H.transpose() * RH;
        ldltH.compute(HRH);
        Ry = RInv*output;

        bhat = ldltH.solve(H.transpose()*Ry);
        res = output - H*bhat;
        predmean = Hnew*bhat;
        predmean += Rnew.transpose()*(RInv*res);
        sig2hat = res.transpose() * RInv*res / (n-p);

        HRHInv = ldltH.solve(Eigen::MatrixXd::Identity(p,p));

        for(int k=0; k<m; k++){
          Rtmp = Rnew.col(k);
          tmp = Hnew.row(k) - RH.transpose()*Rtmp;
          pred_corr(k) = R(0,0) - Rtmp.dot(RInv*Rtmp) + tmp.dot(HRHInv*tmp);
        }

        // simulate from posterior predictive distribution


        L = sig2hat.llt().matrixL(); // lower cholesky factor of sig2hat
        
        for(int k=0; k<m; k++){
          ysim.row(k) =  sqrt(pred_corr(k)) * L * Rcpp::as<Eigen::VectorXd>(Rcpp::rnorm(q, 0.0, 1.0))
                      / sqrt(Rcpp::as<double>(Rcpp::rchisq(1, df)) / df);
        }
        ysim += predmean;
        ysim_sample[it] = ysim;


      // // end prediction


      /******************************************************************************/

    }    

  }else{
    Rcpp::stop("The family of covariance functions is not implemented yet.\n");
  }


/****************************************************************************/

  #ifdef USE_R
    PutRNGstate();
  #endif

  if(family=="CH"||family=="cauchy"){

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

  }else if(family=="matern" || family=="powexp" || family=="gauss"){

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
    Rcpp::stop("The family of covariance functions is not implemented yet.\n");
  }

}
/*******************************************************************************/
/*******************************************************************************/




// Reference priors on range, tail, nugget parameters. Uniform(0, 4) prior on smoothness parameter
Rcpp::List SP::iso_MCMCOBayes_Ref(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
const Eigen::MatrixXd& input, const double& range, const double& tail, const double& nu, 
  const double& nugget, const Rcpp::List& covmodel, const bool& smoothness_est, const Rcpp::List& proposal,
  const int & nsample, const std::string & dtype, const bool& verbose){


  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);


  Eigen::MatrixXd d = SP::pdist(input, input, dtype);

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

  double loglik_curr = SP::MLoglik(range, tail, nu, nugget, output, H, d, covmodel);
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

  if(family=="cauchy" || family=="powexp"){
    nu_ub = 2.0;
  }

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
      loglik_prop = SP::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        range_sample[it] = range_curr;
        // accept_rate_range(it) = FALSE;
      }else{
        // log prior density of cauchy dist
        // log_prior_curr = -log(1.0+range_curr*range_curr);
        // log_prior_prop = -log(1.0+range_prop*range_prop);
        log_prior_curr = SP::reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_curr, covmodel, false);
        log_prior_prop = SP::reference_prior(H, d, range_prop, tail_curr, nu_curr, nugget_curr, covmodel, false);

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
      loglik_prop = SP::MLoglik(range_curr, tail_prop, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        tail_sample(it) = tail_curr;
        // accept_rate_tail[it]= FALSE;
      }else{
        // log prior density of cauchy dist
        // log_prior_curr = -log(1.0+tail_curr*tail_curr);
        // log_prior_prop = -log(1.0+tail_prop*tail_prop);
        log_prior_curr = SP::reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_curr, covmodel, false);
        log_prior_prop = SP::reference_prior(H, d, range_curr, tail_prop, nu_curr, nugget_curr, covmodel, false);

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
        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nugget_sample(it) = nugget_curr;
          // accept_rate_nugget[it]= FALSE;
        }else{
          // log prior density of cauchy dist
          // log_prior_curr = -log(1.0+nugget_curr*nugget_curr);
          // log_prior_prop = -log(1.0+nugget_prop*nugget_prop);
          log_prior_curr = SP::reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_curr, covmodel, false);
          log_prior_prop = SP::reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_prop, covmodel, false);

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

        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

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
      loglik_prop = SP::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        range_sample[it] = range_curr;
        // accept_rate_range(it) = FALSE;
      }else{
        // log prior density of cauchy dist
        // log_prior_curr = -log(1.0+range_curr*range_curr);
        // log_prior_prop = -log(1.0+range_prop*range_prop);
        log_prior_curr = SP::reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_curr, covmodel, false);
        log_prior_prop = SP::reference_prior(H, d, range_prop, tail_curr, nu_curr, nugget_curr, covmodel, false);

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
        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nugget_sample(it) = nugget_curr;
          // accept_rate_nugget[it]= FALSE;
        }else{
          // log prior density of cauchy dist
          // log_prior_curr = -log(1.0+nugget_curr*nugget_curr);
          // log_prior_prop = -log(1.0+nugget_prop*nugget_prop);
          log_prior_curr = SP::reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_curr, covmodel, false);
          log_prior_prop = SP::reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_prop, covmodel, false);

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
        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

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



Rcpp::List SP::iso_MCMCOBayes_Ref(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
const Eigen::MatrixXd& input, const double& range, const double& tail, const double& nu, 
  const double& nugget, const Rcpp::List& covmodel, const bool& smoothness_est, const Rcpp::List& proposal,
  const int& nsample, const std::string& dtype, const bool& verbose, const Eigen::MatrixXd& input_new, 
  const Eigen::MatrixXd& Hnew){


  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);


  Eigen::MatrixXd d = SP::pdist(input, input, dtype);

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

  double loglik_curr = SP::MLoglik(range, tail, nu, nugget, output, H, d, covmodel);
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
  if(family=="cauchy" || family=="powexp"){
    nu_ub = 2.0;
  }
  /*********************** quantities related to prediction ****************/
  int n = output.rows();
  int q = output.cols();
  int m = input_new.rows();
  int p = H.cols();
  //int dim = input.cols();
  Eigen::MatrixXd R(n,n), RInv(n,n), RH(n,p), HRH(p,p), Rnew(n,m), HRHInv(p,p);
  Eigen::MatrixXd res(n,q), Ry(n,q), bhat(p,q);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH, ldlt;

  Eigen::MatrixXd sig2hat(q,q), predmean(m, q), L(q,q);

  //Eigen::MatrixXd d = SP::pdist(input, input, dtype);
  Eigen::MatrixXd d0 = SP::pdist(input, input_new, dtype);

  double df = n-p;
  Eigen::MatrixXd ysim(m,q);
  Eigen::VectorXd Rtmp(n), tmp(p), pred_corr(m);

  Rcpp::List ysim_sample(nsample);

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
      loglik_prop = SP::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        range_sample[it] = range_curr;
        // accept_rate_range(it) = FALSE;
      }else{
        // log prior density of cauchy dist
        // log_prior_curr = -log(1.0+range_curr*range_curr);
        // log_prior_prop = -log(1.0+range_prop*range_prop);
        log_prior_curr = SP::reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_curr, covmodel, false);
        log_prior_prop = SP::reference_prior(H, d, range_prop, tail_curr, nu_curr, nugget_curr, covmodel, false);

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
      loglik_prop = SP::MLoglik(range_curr, tail_prop, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        tail_sample(it) = tail_curr;
        // accept_rate_tail[it]= FALSE;
      }else{
        // log prior density of cauchy dist
        // log_prior_curr = -log(1.0+tail_curr*tail_curr);
        // log_prior_prop = -log(1.0+tail_prop*tail_prop);
        log_prior_curr = SP::reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_curr, covmodel, false);
        log_prior_prop = SP::reference_prior(H, d, range_curr, tail_prop, nu_curr, nugget_curr, covmodel, false);

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
        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nugget_sample(it) = nugget_curr;
          // accept_rate_nugget[it]= FALSE;
        }else{
          // log prior density of cauchy dist
          // log_prior_curr = -log(1.0+nugget_curr*nugget_curr);
          // log_prior_prop = -log(1.0+nugget_prop*nugget_prop);
          log_prior_curr = SP::reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_curr, covmodel, false);
          log_prior_prop = SP::reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_prop, covmodel, false);

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

        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

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

      /********************************* Prediction *********************************/ 
      //if(it>burnin){
        R = SP::iso_kernel(d, range_sample(it), tail_sample(it), nu_sample(it), family);
        Rnew = SP::iso_kernel(d0, range_sample(it), tail_sample(it), nu_sample(it), family);

        R.diagonal().array() += nugget_sample(it);

        ldltR.compute(R);
        RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
        RH = RInv * H;
        HRH = H.transpose() * RH;
        ldltH.compute(HRH);
        Ry = RInv*output;

        bhat = ldltH.solve(H.transpose()*Ry);
        res = output - H*bhat;
        predmean = Hnew*bhat;
        predmean += Rnew.transpose()*(RInv*res);
        sig2hat = res.transpose() * RInv*res / (n-p);

        HRHInv = ldltH.solve(Eigen::MatrixXd::Identity(p,p));

        for(int k=0; k<m; k++){
          Rtmp = Rnew.col(k);
          tmp = Hnew.row(k) - RH.transpose()*Rtmp;
          pred_corr(k) = R(0,0) - Rtmp.dot(RInv*Rtmp) + tmp.dot(HRHInv*tmp);
        }

        // simulate from posterior predictive distribution


        L = sig2hat.llt().matrixL(); // lower cholesky factor of sig2hat
        
        for(int k=0; k<m; k++){
          ysim.row(k) =  sqrt(pred_corr(k)) * L * Rcpp::as<Eigen::VectorXd>(Rcpp::rnorm(q, 0.0, 1.0))
                      / sqrt(Rcpp::as<double>(Rcpp::rchisq(1, df)) / df);
        }
        ysim += predmean;
        ysim_sample[it] = ysim;


      // // end prediction


      /******************************************************************************/


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
      loglik_prop = SP::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // prior rejection
      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        range_sample[it] = range_curr;
        // accept_rate_range(it) = FALSE;
      }else{
        // log prior density of cauchy dist
        // log_prior_curr = -log(1.0+range_curr*range_curr);
        // log_prior_prop = -log(1.0+range_prop*range_prop);
        log_prior_curr = SP::reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_curr, covmodel, false);
        log_prior_prop = SP::reference_prior(H, d, range_prop, tail_curr, nu_curr, nugget_curr, covmodel, false);

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
        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

        // prior rejection
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nugget_sample(it) = nugget_curr;
          // accept_rate_nugget[it]= FALSE;
        }else{
          // log prior density of cauchy dist
          // log_prior_curr = -log(1.0+nugget_curr*nugget_curr);
          // log_prior_prop = -log(1.0+nugget_prop*nugget_prop);
          log_prior_curr = SP::reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_curr, covmodel, false);
          log_prior_prop = SP::reference_prior(H, d, range_curr, tail_curr, nu_curr, nugget_prop, covmodel, false);

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
        loglik_prop = SP::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

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

      /******************************************************************************/
      /******************************************************************************/
      
      /********************************* Prediction *********************************/ 
      //if(it>burnin){
        R = SP::iso_kernel(d, range_sample(it), tail_sample(it), nu_sample(it), family);
        Rnew = SP::iso_kernel(d0, range_sample(it), tail_sample(it), nu_sample(it), family);

        R.diagonal().array() += nugget_sample(it);

        ldltR.compute(R);
        RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
        RH = RInv * H;
        HRH = H.transpose() * RH;
        ldltH.compute(HRH);
        Ry = RInv*output;

        bhat = ldltH.solve(H.transpose()*Ry);
        res = output - H*bhat;
        predmean = Hnew*bhat;
        predmean += Rnew.transpose()*(RInv*res);
        sig2hat = res.transpose() * RInv*res / (n-p);

        HRHInv = ldltH.solve(Eigen::MatrixXd::Identity(p,p));

        for(int k=0; k<m; k++){
          Rtmp = Rnew.col(k);
          tmp = Hnew.row(k) - RH.transpose()*Rtmp;
          pred_corr(k) = R(0,0) - Rtmp.dot(RInv*Rtmp) + tmp.dot(HRHInv*tmp);
        }

        // simulate from posterior predictive distribution


        L = sig2hat.llt().matrixL(); // lower cholesky factor of sig2hat
        
        for(int k=0; k<m; k++){
          ysim.row(k) =  sqrt(pred_corr(k)) * L * Rcpp::as<Eigen::VectorXd>(Rcpp::rnorm(q, 0.0, 1.0))
                      / sqrt(Rcpp::as<double>(Rcpp::rchisq(1, df)) / df);
        }
        ysim += predmean;
        ysim_sample[it] = ysim;


      // // end prediction


      /******************************************************************************/


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
                                Rcpp::_["accept_rate_nu"]=accept_rate_nu,
                                Rcpp::_["pred"] = ysim_sample
                                );      
    }else{
      return Rcpp::List::create(Rcpp::_["range"]=range_sample, 
                                Rcpp::_["tail"]=tail_sample,
                                Rcpp::_["nugget"]=nugget_sample,
                                Rcpp::_["accept_rate_range"]=accept_rate_range,
                                Rcpp::_["accept_rate_tail"]=accept_rate_tail,
                                Rcpp::_["accept_rate_nugget"]=accept_rate_nugget,
                                Rcpp::_["pred"] = ysim_sample);
    }

  }else if(family=="matern"){

    if(smoothness_est){
      return Rcpp::List::create(Rcpp::_["range"]=range_sample, 
                                Rcpp::_["nugget"]=nugget_sample,
                                Rcpp::_["nu"]=nu_sample,
                                Rcpp::_["accept_rate_range"]=accept_rate_range,
                                Rcpp::_["accept_rate_nugget"]=accept_rate_nugget,
                                Rcpp::_["accept_rate_nu"]=accept_rate_nu,
                                Rcpp::_["pred"] = ysim_sample
                                );      
    }else{
      return Rcpp::List::create(Rcpp::_["range"]=range_sample, 
                                Rcpp::_["nugget"]=nugget_sample,
                                Rcpp::_["accept_rate_range"]=accept_rate_range,
                                Rcpp::_["accept_rate_nugget"]=accept_rate_nugget,
                                Rcpp::_["pred"] = ysim_sample);
    }
  }else{
    Rcpp::stop("The MCMC algorithm for the specified covariance family is not implemented.\n");
  }


}


/***********************************************************************/
/*************** Model Comparison ******************/
Rcpp::List SP::model_evaluation(const Eigen::MatrixXd& output, 
  const Eigen::MatrixXd& input, const Eigen::MatrixXd& H, 
  const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
  const Eigen::VectorXd& nu, const Eigen::VectorXd& nugget,
  const Rcpp::List& covmodel,
  const Eigen::MatrixXd& output_new, const Eigen::MatrixXd& input_new, 
  const Eigen::MatrixXd& Hnew, const std::string& dtype, 
  const bool& pointwise, const bool& joint){

    std::string family = Rcpp::as<std::string>(covmodel["family"]);

  int n = output.rows();
  // int q = output.cols();
  int m = input_new.rows();
  int nsample = range.rows();
  int p = H.cols();

  Eigen::MatrixXd d = SP::pdist(input, input, dtype);
  Eigen::MatrixXd dnew = SP::pdist(input_new, input_new, dtype);
  Eigen::MatrixXd dcross = SP::pdist(input, input_new, dtype);
  Eigen::MatrixXd R(n,n), L(n,n),  LH(n,p);
  Eigen::MatrixXd HRH(p,p), HRHchol(p,p),  HRHcholH(p,n);
  Eigen::VectorXd coeff_hat(p), Ly(n), HRy(p);
  double S2, sig2hat;

  Eigen::LLT<Eigen::MatrixXd> lltR, lltH, lltRnew;


  Eigen::MatrixXd Rnew(m,m), Rnewchol(m,m), Rcross(n,m), Rtemp(m,p), Htemp(p,p), RoRp(n,m);
  Eigen::VectorXd predmean(m), res(m), pred_corr(m), Rres(m);

  double SSE;

  Eigen::VectorXd lpd_joint(nsample), lpd_ptw(nsample);

  double con = gsl_sf_lngamma((n-p+m)/2.0) - gsl_sf_lngamma((n-p)/2.0) 
         - (m/2.0)*log((n-p)/2.0) - (m/2.0)*log(M_PI);

  lpd_joint.array() = con;
  lpd_ptw.array() = con;

  Eigen::VectorXd deviance(nsample);
  deviance.array() = gsl_sf_lngamma((n-p)/2.0);

  Progress prog(nsample, true);
  for(int it=0; it<nsample; it++){

    if(Progress::check_abort()){
      return R_NilValue;
    }
    prog.increment();

    R = SP::iso_kernel(d, range(it), tail(it), nu(it), family);
    R.diagonal().array() += nugget(it);
    lltR.compute(R);
    L = lltR.matrixL();
    Ly = L.triangularView<Eigen::Lower>().solve(output);
    LH = L.triangularView<Eigen::Lower>().solve(H);
    HRy = LH.transpose()*Ly;
    HRH = LH.transpose()*LH;


    Rcross = SP::iso_kernel(dcross, range(it), tail(it), nu(it), family);

    lltH.compute(HRH);
    coeff_hat = lltH.solve(HRy);

    // compute posterior predictive mean and variance 
    predmean = Hnew * coeff_hat;
    predmean += Rcross.transpose()*lltR.solve(output-H*coeff_hat);

    res = output_new - predmean;
    S2 = Ly.squaredNorm() - HRy.dot(coeff_hat);
    sig2hat = S2 / (n-p);

    // compute log predictive density (pointwise and joint)
    RoRp = lltR.solve(Rcross);
    Rtemp = Hnew - RoRp.transpose()*H;
    HRHchol= lltH.matrixL();
    Htemp = HRHchol.triangularView<Eigen::Lower>().solve(Rtemp);

    if(joint){
      Rnew = SP::iso_kernel(dnew, range(it), tail(it), nu(it), family);
      Rnew.diagonal().array() += nugget(it);
      Rnew += -Rcross.transpose()*RoRp + Htemp*Htemp.transpose();

      lltRnew.compute(sig2hat*Rnew);
      Rnewchol = lltRnew.matrixL();
      Rres = Rnewchol.triangularView<Eigen::Lower>().solve(res);
      SSE = Rres.squaredNorm() / (n-p);
      lpd_joint(it) += -Rnewchol.diagonal().array().log().sum() 
               - ((n-p+m)/2.0)*log(1.0+SSE);

    }

    if(pointwise){

      for(int k=0; k<m; k++){
        pred_corr(k) = R(0,0) - Rcross.col(k).transpose()*RoRp.col(k) 
                + Htemp.row(k).squaredNorm();
      }
      pred_corr = sig2hat * pred_corr;
      
      
      SSE = (res.array()*pred_corr.array().inverse() * res.array()).sum() / (n-p);
      lpd_ptw(it) += -0.5*pred_corr.array().log().sum() 
               - ((n-p+m)/2.0)*log(1.0+SSE);

    }

    // compute deviance function for each sample 
    L = lltR.matrixL();
    HRHchol= lltH.matrixL();
    deviance(it) += -L.diagonal().array().log().sum() 
             -HRHchol.diagonal().array().log().sum()
             -((n-p)/2.0) * log(S2);

    deviance(it) *= -2.0; 


    
  }


  // compute DIC: 
  double range_mean = range.mean();
  double tail_mean = tail.mean();
  double nu_mean = nu.mean();
  double nugget_mean = nugget.mean();

  R = SP::iso_kernel(d, range_mean, tail_mean, nu_mean, family);
  R.diagonal().array() += nugget_mean;

  lltR.compute(R);
  L = lltR.matrixL();
  Ly = L.triangularView<Eigen::Lower>().solve(output);
  LH = L.triangularView<Eigen::Lower>().solve(H);
  HRy = LH.transpose()*Ly;
  HRH = LH.transpose()*LH;

  lltH.compute(HRH);
  coeff_hat = lltH.solve(HRy);
  S2 = Ly.squaredNorm() - HRy.dot(coeff_hat);

  double deviance_Bayes = gsl_sf_lngamma((n-p)/2.0);
  L = lltR.matrixL();
  HRHchol = lltH.matrixL();
  deviance_Bayes += -L.diagonal().array().log().sum() 
             -HRHchol.diagonal().array().log().sum()
             -((n-p)/2.0) * log(S2);
  deviance_Bayes *= -2.0;
  double p_D = deviance.mean() - deviance_Bayes; // effective number of parameters
  double DIC = deviance.mean() + p_D;

  // Eigen::VectorXd Result(4);

  // Result(0) = p_D;
  // Result(1) = DIC;

  // Result(2) = lpd_ptw.mean();
  // Result(3) = lpd_joint.mean();

  return Rcpp::List::create(Rcpp::_["pD"]=p_D,
                            Rcpp::_["DIC"]=DIC,
                            Rcpp::_["lppd"]=lpd_ptw.mean(),
                            Rcpp::_["ljpd"]=lpd_joint.mean());

}




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

// using namespace Rcpp;


class UQ
{
  public:
    // Default constructor.

    // basic functions 
    Eigen::MatrixXd pdist(const Eigen::MatrixXd& locs1, const Eigen::MatrixXd& locs2);
    Rcpp::List tdist(const Eigen::MatrixXd& locs1, const Eigen::MatrixXd& locs2, const double& cutRange);
    Rcpp::List adist(const Eigen::MatrixXd& input1, const Eigen::MatrixXd& input2);
    
    Eigen::MatrixXd tensor_kernel0(const Eigen::MatrixXd& input1, const Eigen::MatrixXd& input2, 
      const Eigen::VectorXd& range, const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, 
      const std::string& family);
    Eigen::MatrixXd tensor_kernel(const Rcpp::List& d, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
      const Eigen::VectorXd& nu, const std::string& family);
    Rcpp::List deriv_tensor_kernel(const Rcpp::List& d, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
      const Eigen::VectorXd& nu, const std::string& family);
    
    Eigen::MatrixXd ARD_kernel(const Rcpp::List& d, const Eigen::VectorXd& range, const double& tail, 
      const double& nu, const std::string& family);
    Rcpp::List deriv_ARD_kernel(const Rcpp::List& d, const Eigen::VectorXd& range, const double& tail, 
      const double& nu, const std::string& family);

    // helper
    // Eigen::VectorXd logit(Eigen::VectorXd x, double lb, double ub);
    // Eigen::VectorXd ilogit(Eigen::VectorXd x, double lb, double ub);
    /**************************************************************************************/

    /**************************************************************************************/
    // The following routines are written for 
    // (1) univariate GP models,
    // (2) multivariate GP models with separable covariance structures. 
    // In both (1) and (2), Jeffreys' prior is assumed for location-scale parameters,
    // that is, p(b, SIGMA) propto det(SIGMA)^(-1/2). 
    /*************************************************************************************/
    // likelihood 
    double MLoglik(const Eigen::VectorXd& range, const Eigen::VectorXd& tail, const Eigen::VectorXd& nu,  
      const double& nugget, const Eigen::MatrixXd& y, const Eigen::MatrixXd& H, const Rcpp::List& d, 
      const Rcpp::List& covmodel);
    Eigen::VectorXd gradient_MLoglik(const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
      const Eigen::VectorXd& nu, const double& nugget, const Eigen::MatrixXd& y,  
      const Eigen::MatrixXd& H, const Rcpp::List& d, const Rcpp::List& covmodel, const bool& smoothness_est);
    double PLoglik(const Eigen::VectorXd& range, const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, 
      const double& nugget, const Eigen::MatrixXd& y, const Eigen::MatrixXd& H, const Rcpp::List& d, 
          const Rcpp::List& covmodel);
    
    // prior 
    // double reference_prior(const Eigen::MatrixXd& y, const Eigen::MatrixXd& H, Rcpp::List& d, 
    //   const Eigen::VectorXd& range, const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, 
    //   const double& nugget, Rcpp::List& covmodel, bool smoothness_est); 

    // posterior 
    double mposterior_ref_range(const Eigen::MatrixXd& y, const Eigen::MatrixXd& H, Rcpp::List& d, 
      const Eigen::VectorXd& range, const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, 
      const double& nugget, const Rcpp::List& covmodel, const bool& nugget_est, const bool& smoothness_est); // gamma parametrization 
    double mposterior_cauchy_range(const Eigen::MatrixXd& y, const Eigen::MatrixXd& H, const Rcpp::List& d, 
      const Eigen::VectorXd& range, const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, 
      const double& nugget, const Rcpp::List& covmodel, const bool& nugget_est, const bool& smoothness_est); // gamma parametrization 
    Eigen::VectorXd grad_mposterior_cauchy_range(const Eigen::MatrixXd& y, const Eigen::MatrixXd& H, 
      Rcpp::List& d, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, 
      const double& nugget, const Rcpp::List& covmodel, const bool& smoothness_est);

    // prediction and simulation
    Eigen::MatrixXd simulate(const Eigen::MatrixXd& input, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& coeff, const double& sig2, const Eigen::VectorXd& range, 
      const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, const double& nugget, 
      const Rcpp::List& covmodel, const int& nsample);
    Rcpp::List predict(const Eigen::MatrixXd& y, const Eigen::MatrixXd& H, const Eigen::MatrixXd& input, 
      const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew, const Eigen::VectorXd& range, 
      const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, const double& nugget,
      const Rcpp::List& covmodel); 
    // Rcpp::List condsim(Eigen::MatrixXd& y,  Eigen::MatrixXd& H, Eigen::MatrixXd& input, Eigen::MatrixXd& input_new, 
    // Eigen::MatrixXd& Hnew, Rcpp::List& par, Rcpp::List& covmodel, bool nugget_est);
    Rcpp::List tensor_simulate_predictive_dist(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& input, const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew, 
      const Eigen::MatrixXd& range, const Eigen::MatrixXd& tail, const Eigen::MatrixXd& nu, 
      const Eigen::VectorXd& nugget, const Rcpp::List& covmodel);

    Rcpp::List ARD_simulate_predictive_dist(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& input, const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew, 
      const Eigen::MatrixXd& range, const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, 
      const Eigen::VectorXd& nugget, const Rcpp::List& covmodel);

    // conditional simulation
    Rcpp::List tensor_condsim(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& input, const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew, 
      const Eigen::VectorXd& range, const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, 
      const double& nugget, const Rcpp::List& covmodel, int nsample);

    Rcpp::List ARD_condsim(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& input, const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew, 
      const Eigen::VectorXd& range, const double& tail, const double& nu, 
      const double& nugget, const Rcpp::List& covmodel, int nsample);

    // MCMC algorithm
    // LogNormal on constrained parameter space
    // Rcpp::List tensor_MCMC_LN(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, const Eigen::MatrixXd& input, const Rcpp::List& par_curr, Rcpp::List& covmodel, 
    //   bool nugget_est, Rcpp::List& proposal, int nsample, bool verbose);
    // Rcpp::List ARD_MCMC_LN(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, const Eigen::MatrixXd& input, const Rcpp::List& par_curr, Rcpp::List& covmodel, 
    //       bool nugget_est, Rcpp::List& proposal, int nsample, bool verbose);
    // Random walk on unconstrained parameter space 
    Rcpp::List tensor_MCMCOBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& input, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
      const Eigen::VectorXd& nu, const double& nugget, const Rcpp::List& covmodel, 
      const bool& smoothness_est, const Rcpp::List& proposal, const int& nsample, const bool& verbose);
    Rcpp::List ARD_MCMCOBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& input, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
      const Eigen::VectorXd& nu, const double& nugget, const Rcpp::List& covmodel, 
      const bool& smoothness_est, const Rcpp::List& proposal, const int& nsample, const bool& verbose);

    Rcpp::List tensor_MCMCSBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& input, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
      const Eigen::VectorXd& nu, const double& nugget, const Rcpp::List& covmodel, 
      const bool& smoothness_est, const Rcpp::List& prior, const Rcpp::List& proposal, const int& nsample, const bool& verbose);
    Rcpp::List ARD_MCMCSBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& input, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
      const Eigen::VectorXd& nu, const double& nugget, const Rcpp::List& covmodel, 
      const bool& smoothness_est, const Rcpp::List& prior, const Rcpp::List& proposal, const int& nsample, const bool& verbose);

    // MCMC + prediction
    Rcpp::List ARD_MCMCOBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& input, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
      const Eigen::VectorXd& nu, const double& nugget, const Rcpp::List& covmodel, 
      const bool& smoothness_est, const Rcpp::List& proposal, const int& nsample, const bool& verbose,
      const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew);

    Rcpp::List ARD_MCMCSBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& input, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
      const Eigen::VectorXd& nu, const double& nugget, const Rcpp::List& covmodel, 
      const bool& smoothness_est, const Rcpp::List& prior, const Rcpp::List& proposal, const int& nsample, const bool& verbose,
      const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew);

    Rcpp::List tensor_MCMCOBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& input, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
      const Eigen::VectorXd& nu, const double& nugget, const Rcpp::List& covmodel, 
      const bool& smoothness_est, const Rcpp::List& proposal, const int& nsample, const bool& verbose,
      const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew);

    Rcpp::List tensor_MCMCSBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& input, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
      const Eigen::VectorXd& nu, const double& nugget, const Rcpp::List& covmodel, 
      const bool& smoothness_est, const Rcpp::List& prior, const Rcpp::List& proposal, const int& nsample, const bool& verbose,
      const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew);

    Rcpp::List tensor_model_evaluation(const Eigen::MatrixXd& output, 
      const Eigen::MatrixXd& input, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& range, const Eigen::MatrixXd& tail, 
      const Eigen::MatrixXd& nu, const Eigen::VectorXd& nugget,
      const Rcpp::List& covmodel,
      const Eigen::MatrixXd& output_new, const Eigen::MatrixXd& input_new, 
      const Eigen::MatrixXd& Hnew, const std::string& dtype, 
      const bool& pointwise, const bool& joint);
    Rcpp::List ARD_model_evaluation(const Eigen::MatrixXd& output, 
      const Eigen::MatrixXd& input, const Eigen::MatrixXd& H, 
      const Eigen::MatrixXd& range, const Eigen::VectorXd& tail, 
      const Eigen::VectorXd& nu, const Eigen::VectorXd& nugget,
      const Rcpp::List& covmodel,
      const Eigen::MatrixXd& output_new, const Eigen::MatrixXd& input_new, 
      const Eigen::MatrixXd& Hnew, const std::string& dtype, 
      const bool& pointwise, const bool& joint);
    /**************************************************************************************/

    // constructor 
    UQ()
    {
      
    }
    // destructor 
    ~UQ()
    {
      
    }
};


/****************************************************************************************/
/****************************************************************************************/

/****************************************************************************************/
/* Distance Matrices  */
/****************************************************************************************/

// @title Compute a distance matrix based on two sets of locations
// 
// @description This function computes the distance matrix based on two sets of locations.
// @param locs1 a matrix of locations
// @param locs2 a matrix of locations
// 
// @author Pulong Ma <mpulong@gmail.com>
//
// 
Eigen::MatrixXd  UQ::pdist(const Eigen::MatrixXd& locs1, const Eigen::MatrixXd& locs2)
{


  int n1 = locs1.rows();
  int n2 = locs2.rows();
  Eigen::MatrixXd distmat(n1, n2); 
  

  //#pragma omp parallel for collapse(2)
  for(int i=0; i<n1; i++){
    for(int j=0; j<n2; j++){
      distmat(i,j) = sqrt((locs1.row(i)-locs2.row(j)).array().square().sum());
    }
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
//
// @return a Rcpp::List of arguments for nonzero distances:
// \describe{
// \item{rowid}{a vector of row indeces of nonzero elements.} 
// \item{colid}{a vector of column indeces of nonzero elements.}
// \item{val}{a vector of nonzero elements.}
// \item{sCHt}{a sparse matrix of class `dgCMatrix'.}
//}
//
// @author Pulong Ma <mpulong@gmail.com>
//

Rcpp::List UQ::tdist(const Eigen::MatrixXd& locs1, const Eigen::MatrixXd& locs2, 
               const double& cutRange)
{
  
  // Purpose: This function computes distance between pairs of location up to a
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



// @title Compute a distance matrix along each dimension based on two sets of inputs 
// 
// @description This function computes the distance for each input dimension and returns
// a Rcpp::List. Each element in the Rcpp::List contains the distance matrix for a specific dimension.
//
// @param input1 a matrix of inputs
// @param input2 a matrix of inputs
// 
// @return a Rcpp::List of distance matrix
//
// @author Pulong Ma <mpulong@gmail.com>
//

Rcpp::List UQ::adist(const Eigen::MatrixXd& input1, const Eigen::MatrixXd& input2){ 
  
  int Dim_x = input1.cols();

  Rcpp::List dist(Dim_x);


  for(int k=0; k<Dim_x; k++){
    dist[k] = UQ::pdist(input1.col(k), input2.col(k));
  }


  return dist;
}


/****************************************************************************************/
/****************************************************************************************/



/****************************************************************************************/
/* Compute tensor covariance kernels with different forms with inputs*/
/****************************************************************************************/

Eigen::MatrixXd UQ::tensor_kernel0(const Eigen::MatrixXd& input1, const Eigen::MatrixXd& input2, 
  const Eigen::VectorXd& range, const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, 
  const std::string& family){

  int n1 = input1.rows();
  int n2 = input2.rows();
  int Dim = input1.cols();

  Eigen::MatrixXd covmat(n1, n2);

  Rcpp::List dist(Dim);
  dist = UQ::adist(input1, input2);

  
  Eigen::MatrixXd distemp(n1,n2), cortemp(n1,n2);

  if(family=="CH"){

    covmat.setOnes();
    for(int i=0; i<Dim; i++){
      distemp = Rcpp::as<Eigen::MatrixXd>(dist[i]);
      cortemp = CH(distemp, range(i), tail(i), nu(i));
      covmat.array() *= cortemp.array();
    }

  }else if(family=="matern"){

    covmat.setOnes();
    for(int i=0; i<Dim; i++){
      distemp = Rcpp::as<Eigen::MatrixXd>(dist[i]);
      cortemp = matern(distemp, range(i), nu(i));
      covmat.array() *= cortemp.array();
    }

  // }else if(family=="exp"){
  //   covmat.setOnes();
  //   for(int i=0; i<Dim; i++){
  //     distemp = Rcpp::as<Eigen::MatrixXd>(dist[i]);
  //     cortemp = matern(distemp, range(i), 0.5);
  //     covmat.array() *= cortemp.array();
  //   }

  // }else if(family=="matern_3_2"){
  //   covmat.setOnes();
  //   for(int i=0; i<Dim; i++){
  //     distemp = Rcpp::as<Eigen::MatrixXd>(dist[i]);
  //     cortemp = matern(distemp, range(i), 1.5);
  //     covmat.array() *= cortemp.array();
  //   }

  // }else if(family=="matern_5_2"){
  //   covmat.setOnes();
  //   for(int i=0; i<Dim; i++){
  //     distemp = Rcpp::as<Eigen::MatrixXd>(dist[i]);
  //     cortemp = matern(distemp, range(i), 2.5);
  //     covmat.array() *= cortemp.array();
  //   }
    
  }else if(family=="gauss"){
    covmat.setOnes();
    for(int i=0; i<Dim; i++){
      distemp = Rcpp::as<Eigen::MatrixXd>(dist[i]);
      cortemp = powexp(distemp, range(i), 2.0);
      covmat.array() *= cortemp.array();
    }

  }else if(family=="powexp"){

    covmat.setOnes();
    for(int i=0; i<Dim; i++){
      distemp = Rcpp::as<Eigen::MatrixXd>(dist[i]);
      cortemp = powexp(distemp, range(i), nu(i));
      covmat.array() *= cortemp.array();
    }
     
  }else if(family=="cauchy"){

    covmat.setOnes();
    for(int i=0; i<Dim; i++){
      distemp = Rcpp::as<Eigen::MatrixXd>(dist[i]);
      cortemp = cauchy(distemp, range(i), tail(i), nu(i));
      covmat.array() *= cortemp.array();
    }
     
  }else{
    Rcpp::stop("The family of covariance functions is not yet supported!\n");
  }      



  return covmat;
}

/****************************************************************************************/
/****************************************************************************************/


/****************************************************************************************/
/****************************************************************************************/
Eigen::MatrixXd UQ::tensor_kernel(const Rcpp::List& d, const Eigen::VectorXd& range, 
  const Eigen::VectorXd& tail, const Eigen::VectorXd& smoothness, const std::string& family){

  int n1, n2, Dim;

  Eigen::MatrixXd dtemp, covmat, covtemp;

  Dim = d.size();
  dtemp = Rcpp::as<Eigen::MatrixXd>(d[0]);
  n1 = dtemp.rows();
  n2 = dtemp.cols();
  covmat = Eigen::MatrixXd::Ones(n1, n2);

  Eigen::VectorXd nu(Dim);
  if(smoothness.size()==1){
    nu = smoothness(0) * Eigen::VectorXd::Ones(Dim);
  }else{
    nu = smoothness;
  }
  
  Eigen::MatrixXd distemp(n1,n2), cortemp(n1,n2);

  if(family=="CH"){

    covmat.setOnes();
    for(int i=0; i<Dim; i++){
      distemp = Rcpp::as<Eigen::MatrixXd>(d[i]);
      cortemp = CH(distemp, range(i), tail(i), nu(i));
      covmat.array() *= cortemp.array();
    }

  }else if(family=="matern"){
    covmat.setOnes();
    for(int i=0; i<Dim; i++){
      distemp = Rcpp::as<Eigen::MatrixXd>(d[i]);
      cortemp = matern(distemp, range(i), nu(i));
      covmat.array() *= cortemp.array();
    }

  // }else if(family=="exp"){

  //   covmat.setOnes();
  //   for(int i=0; i<Dim; i++){
  //     distemp = Rcpp::as<Eigen::MatrixXd>(d[i]);
  //     cortemp = matern(distemp, range(i), 0.5);
  //     covmat.array() *= cortemp.array();
  //   }

  // }else if(family=="matern_3_2"){

  //   covmat.setOnes();
  //   for(int i=0; i<Dim; i++){
  //     distemp = Rcpp::as<Eigen::MatrixXd>(d[i]);
  //     cortemp = matern(distemp, range(i), 1.5);
  //     covmat.array() *= cortemp.array();
  //   }

  // }else if(family=="matern_5_2"){

  //   covmat.setOnes();
  //   for(int i=0; i<Dim; i++){
  //     distemp = Rcpp::as<Eigen::MatrixXd>(d[i]);
  //     cortemp = matern(distemp, range(i), 2.5);
  //     covmat.array() *= cortemp.array();
  //   }
    
  }else if(family=="gauss"){
    covmat.setOnes();
    for(int i=0; i<Dim; i++){
      distemp = Rcpp::as<Eigen::MatrixXd>(d[i]);
      cortemp = powexp(distemp, range(i), 2.0);
      covmat.array() *= cortemp.array();
    }

  }else if(family=="powexp"){
    covmat.setOnes();
    for(int i=0; i<Dim; i++){
      distemp = Rcpp::as<Eigen::MatrixXd>(d[i]);
      cortemp = powexp(distemp, range(i), nu(i));
      covmat.array() *= cortemp.array();
    }
     
  }else if(family=="cauchy"){
    covmat.setOnes();
    for(int i=0; i<Dim; i++){
      distemp = Rcpp::as<Eigen::MatrixXd>(d[i]);
      cortemp = cauchy(distemp, range(i), tail(i), nu(i));
      covmat.array() *= cortemp.array();
    }
     
  }else{
    Rcpp::stop("The family of covariance functions is not yet supported!\n");
  }      

  return covmat;
}


// anisotropic covariance function
Eigen::MatrixXd UQ::ARD_kernel(const Rcpp::List& d, const Eigen::VectorXd& range,  
  const double& tail, const double& nu, const std::string& family){

  int n1, n2, Dim;

  Eigen::MatrixXd dtemp, covmat, covtemp;

  Dim = d.size();
  dtemp = Rcpp::as<Eigen::MatrixXd>(d[0]);
  n1 = dtemp.rows();
  n2 = dtemp.cols();
  covmat = Eigen::MatrixXd::Ones(n1, n2);
  
  Eigen::MatrixXd distemp(n1,n2);
  if(family=="CH"){

    distemp.setZero();
    for(int i=0; i<Dim; i++){
      distemp.array() +=  (Rcpp::as<Eigen::MatrixXd>(d[i])/range(i)).array() * (Rcpp::as<Eigen::MatrixXd>(d[i])/range(i)).array();
    }
    distemp = distemp.array().sqrt();
    covmat = CH(distemp, 1.0, tail, nu);

  }else if(family=="matern"){

    distemp.setZero();
    for(int i=0; i<Dim; i++){
      distemp.array() +=  (Rcpp::as<Eigen::MatrixXd>(d[i])/range(i)).array() * (Rcpp::as<Eigen::MatrixXd>(d[i])/range(i)).array();
    }
    distemp = distemp.array().sqrt();
    covmat = matern(distemp, 1.0, nu);


  // }else if(family=="exp"){

  //   distemp.setZero();
  //   for(int i=0; i<Dim; i++){
  //     distemp.array() +=  (Rcpp::as<Eigen::MatrixXd>(d[i])/range(i)).array() * (Rcpp::as<Eigen::MatrixXd>(d[i])/range(i)).array();
  //   }
  //   distemp = distemp.array().sqrt();
  //   covmat = matern(distemp, 1.0, 0.5);

  // }else if(family=="matern_3_2"){

  //   distemp.setZero();
  //   for(int i=0; i<Dim; i++){
  //     distemp.array() +=  (Rcpp::as<Eigen::MatrixXd>(d[i])/range(i)).array() * (Rcpp::as<Eigen::MatrixXd>(d[i])/range(i)).array();
  //   }
  //   distemp = distemp.array().sqrt();
  //   covmat = matern(distemp, 1.0, 1.5);

  // }else if(family=="matern_5_2"){

  //   distemp.setZero();
  //   for(int i=0; i<Dim; i++){
  //     distemp.array() +=  (Rcpp::as<Eigen::MatrixXd>(d[i])/range(i)).array() * (Rcpp::as<Eigen::MatrixXd>(d[i])/range(i)).array();
  //   }
  //   distemp = distemp.array().sqrt();
  //   covmat = matern(distemp, 1.0, 2.5);
    
  }else if(family=="gauss"){

    distemp.setZero();
    for(int i=0; i<Dim; i++){
      distemp.array() +=  (Rcpp::as<Eigen::MatrixXd>(d[i])/range(i)).array() * (Rcpp::as<Eigen::MatrixXd>(d[i])/range(i)).array();
    }
    distemp = distemp.array().sqrt();
    covmat = powexp(distemp, 1.0, 2.0);

  }else if(family=="powexp"){

    distemp.setZero();
    for(int i=0; i<Dim; i++){
      distemp.array() +=  (Rcpp::as<Eigen::MatrixXd>(d[i])/range(i)).array() * (Rcpp::as<Eigen::MatrixXd>(d[i])/range(i)).array();
    }
    distemp = distemp.array().sqrt();
    covmat = powexp(distemp, 1.0, nu);
     
  }else if(family=="cauchy"){

    distemp.setZero();
    for(int i=0; i<Dim; i++){
      distemp.array() +=  (Rcpp::as<Eigen::MatrixXd>(d[i])/range(i)).array() * (Rcpp::as<Eigen::MatrixXd>(d[i])/range(i)).array();
    }
    distemp = distemp.array().sqrt();
    covmat = cauchy(distemp, 1.0, tail, nu);
     
  }else{
    Rcpp::stop("The family of covariance functions is not yet supported!\n");
  }      

  return covmat;
}

/****************************************************************************************/
/****************************************************************************************/

Rcpp::List UQ::deriv_tensor_kernel(const Rcpp::List& d, const Eigen::VectorXd& range,  
  const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, const std::string& family){

  int n1, n2, Dim;

  Eigen::MatrixXd dtemp;


  Dim = d.size();
  dtemp = Rcpp::as<Eigen::MatrixXd>(d[0]);
  n1 = dtemp.rows();
  n2 = dtemp.cols();

  
  Eigen::MatrixXd distemp(n1,n2), temp1(n1,n2), temp2(n1,n2), temp3(n1,n2);
  int count;
  Rcpp::List dR(3*Dim);

  if(family=="CH"){

    for(int k=0; k<Dim; k++){
      dR[k] = Eigen::MatrixXd::Ones(n1,n2);
      dR[Dim+k] = Eigen::MatrixXd::Ones(n1,n2);
      dR[2*Dim+k] = Eigen::MatrixXd::Ones(n1,n2);
      for(int i=0; i<Dim; i++){
        distemp = Rcpp::as<Eigen::MatrixXd>(d[i]);
        if(i==k){
          temp1 = CH_deriv_range(distemp, range(i), tail(i), nu(i));
          temp2 = CH_deriv_tail(distemp, range(i), tail(i), nu(i));
          temp3 = CH_deriv_nu(distemp, range(i), tail(i), nu(i));

          dR[k] = (Rcpp::as<Eigen::MatrixXd>(dR[k])).array() * temp1.array();
          dR[Dim+k] = (Rcpp::as<Eigen::MatrixXd>(dR[Dim+k])).array() * temp2.array();
          dR[2*Dim+k] = (Rcpp::as<Eigen::MatrixXd>(dR[2*Dim+k])).array() * temp3.array();

        }else{
          temp1 = CH(distemp, range(i), tail(i), nu(i)); 
          dR[k] = (Rcpp::as<Eigen::MatrixXd>(dR[k])).array() * temp1.array();  
          dR[Dim+k] = (Rcpp::as<Eigen::MatrixXd>(dR[Dim+k])).array() * temp1.array();  
          dR[2*Dim+k] = (Rcpp::as<Eigen::MatrixXd>(dR[2*Dim+k])).array() * temp1.array();      
        }
      }
    }




    count = 3*Dim;

  }else if(family=="matern"){

    // covmat.setOnes();
    for(int k=0; k<Dim; k++){
      dR[k] = Eigen::MatrixXd::Ones(n1,n2);
      for(int i=0; i<Dim; i++){
        distemp = Rcpp::as<Eigen::MatrixXd>(d[i]);
        if(i==k){
          temp1 = matern_deriv_range(distemp, range(i), nu(i));
          dR[k] = (Rcpp::as<Eigen::MatrixXd>(dR[k])).array() * temp1.array();

        }else{
          temp1 = matern(distemp, range(i), nu(i));    
          dR[k] = (Rcpp::as<Eigen::MatrixXd>(dR[k])).array() * temp1.array();  
        }
      }
    }

    dR[Dim] = R_NilValue; 
    count = Dim+1;

  // }else if(family=="exp"){

  //   for(int k=0; k<Dim; k++){
  //     dR[k] = Eigen::MatrixXd::Ones(n1,n2);
  //     for(int i=0; i<Dim; i++){
  //       distemp = Rcpp::as<Eigen::MatrixXd>(d[i]);
  //       if(i==k){
  //         temp1 = matern_deriv_range(distemp, range(i), 0.5);
  //         dR[k] = (Rcpp::as<Eigen::MatrixXd>(dR[k])).array() * temp1.array();
  //       }else{
  //         temp1 = matern(distemp, range(i), 0.5);    
  //         dR[k] = (Rcpp::as<Eigen::MatrixXd>(dR[k])).array() * temp1.array();    
  //       }
  //     }
  //   }

  //   dR[Dim] = R_NilValue; 
  //   count = Dim+1;

  // }else if(family=="matern_3_2"){

  //   for(int k=0; k<Dim; k++){
  //     dR[k] = Eigen::MatrixXd::Ones(n1,n2);
  //     for(int i=0; i<Dim; i++){
  //       distemp = Rcpp::as<Eigen::MatrixXd>(d[i]);
  //       if(i==k){
  //         temp1 = matern_deriv_range(distemp, range(i), 1.5);
  //         dR[k] = (Rcpp::as<Eigen::MatrixXd>(dR[k])).array() * temp1.array();
  //       }else{
  //         temp1 = matern(distemp, range(i), 1.5);    
  //         dR[k] = (Rcpp::as<Eigen::MatrixXd>(dR[k])).array() * temp1.array();    
  //       }
  //     }
  //   }

  //   dR[Dim] = R_NilValue; 
  //   count = Dim+1;

  // }else if(family=="matern_5_2"){

  //   for(int k=0; k<Dim; k++){
  //     dR[k] = Eigen::MatrixXd::Ones(n1,n2);
  //     for(int i=0; i<Dim; i++){
  //       distemp = Rcpp::as<Eigen::MatrixXd>(d[i]);
  //       if(i==k){
  //         temp1 = matern_deriv_range(distemp, range(i), 2.5);
  //         dR[k] = (Rcpp::as<Eigen::MatrixXd>(dR[k])).array() * temp1.array();
  //       }else{
  //         temp1 = matern(distemp, range(i), 2.5);    
  //         dR[k] = (Rcpp::as<Eigen::MatrixXd>(dR[k])).array() * temp1.array();    
  //       }
  //     }
  //   }

  //   dR[Dim] = R_NilValue; 
  //   count = Dim+1;
    
  }else if(family=="gauss"){

    for(int k=0; k<Dim; k++){
      dR[k] = Eigen::MatrixXd::Ones(n1,n2);
      for(int i=0; i<Dim; i++){
        distemp = Rcpp::as<Eigen::MatrixXd>(d[i]);
        if(i==k){
          temp1 = powexp_deriv_range(distemp, range(i), 2.0);
          dR[k] = (Rcpp::as<Eigen::MatrixXd>(dR[k])).array() * temp1.array();
        }else{
          temp1 = powexp(distemp, range(i), 2.0);
          dR[k] = (Rcpp::as<Eigen::MatrixXd>(dR[k])).array() * temp1.array();
        }
      }
    }

    dR[Dim] = R_NilValue; 
    count = Dim+1;

  }else if(family=="powexp"){

    for(int k=0; k<Dim; k++){
      dR[k] = Eigen::MatrixXd::Ones(n1,n2);
      for(int i=0; i<Dim; i++){
        distemp = Rcpp::as<Eigen::MatrixXd>(d[i]);
        if(i==k){
          temp1 = powexp_deriv_range(distemp, range(i), nu(i));
          dR[k] = (Rcpp::as<Eigen::MatrixXd>(dR[k])).array() * temp1.array();
        }else{
          temp1 = powexp(distemp, range(i), nu(i));
          dR[k] = (Rcpp::as<Eigen::MatrixXd>(dR[k])).array() * temp1.array();
        }
      }
    }

    dR[Dim] = R_NilValue; 
    count = Dim+1;
     
  }else if(family=="cauchy"){

    for(int k=0; k<Dim; k++){
      dR[k] = Eigen::MatrixXd::Ones(n1,n2);
      dR[Dim+k] = Eigen::MatrixXd::Ones(n1,n2);
      for(int i=0; i<Dim; i++){
        distemp = Rcpp::as<Eigen::MatrixXd>(d[i]);
        if(i==k){
          temp1 = cauchy_deriv_range(distemp, range(i), tail(i), nu(i));
          temp2 = cauchy_deriv_tail(distemp, range(i), tail(i), nu(i));
          dR[k] = (Rcpp::as<Eigen::MatrixXd>(dR[k])).array() * temp1.array();
          dR[Dim+k] = (Rcpp::as<Eigen::MatrixXd>(dR[Dim+k])).array() * temp2.array();
        }else{
          temp1 = cauchy(distemp, range(i), tail(i), nu(i)); 
          dR[k] = (Rcpp::as<Eigen::MatrixXd>(dR[k])).array() * temp1.array();  
          dR[Dim+k] = (Rcpp::as<Eigen::MatrixXd>(dR[Dim+k])).array() * temp1.array();       
        }
      }
    }

    dR[2*Dim] = R_NilValue;
    count = 2*Dim+1;
     
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

/****************************************************************************************/
/****************************************************************************************/

Rcpp::List UQ::deriv_ARD_kernel(const Rcpp::List& d, const Eigen::VectorXd& range, const double& tail, 
      const double& nu, const std::string& family){

  // int Dim = d.size();
  
  Rcpp::List dR;

  if(family=="CH"){
    dR = deriv_ARD_CH(d, range, tail, nu);

  }else if(family=="matern"){
    dR = deriv_ARD_matern(d, range, nu);    

  // }else if(family=="exp"){
  //   dR = deriv_ARD_matern(d, range, 0.5); 

  // }else if(family=="matern_3_2"){
  //   dR = deriv_ARD_matern(d, range, 1.5);

  // }else if(family=="matern_5_2"){
  //   dR = deriv_ARD_matern(d, range, 2.5);
    
  }else if(family=="gauss"){
    dR = deriv_ARD_powexp(d, range, 2.0);

  }else if(family=="powexp"){
    dR = deriv_ARD_powexp(d, range, nu);

  }else if(family=="cauchy"){
    dR = deriv_ARD_cauchy(d, range, tail, nu);
     
  }else{
    Rcpp::stop("The family of covariance functions is not yet supported!\n");
  }      

  return dR;
}

/****************************************************************************************/
/****************************************************************************************/


/****************************************************************************************/
/****************************************************************************************/
// a wrapper for covariance kernels with various forms and families
// Eigen::MatrixXd UQ::kernel(Rcpp::List& d, Rcpp::List& param, Rcpp::List& covmodel){

//   std::string family = Rcpp::as<std::string>(covmodel["family"]);
//   std::string form = Rcpp::as<std::string>(covmodel["form"]);

//   Eigen::MatrixXd mat;

//   if(form=="tensor"){
//     mat = UQ::tensor_kernel(d, param, family);
//   }else if(form=="ARD"){
//     mat = UQ::ARD_kernel(d, param, family);
//   }else{
//     Rcpp::stop("The covariance kernel is not implemented.\n");
//   }

//   return mat;
// }

/****************************************************************************************/
/****************************************************************************************/
/****************************************************************************************/
/****************************************************************************************/


/****************************************************************************************/
/* Compute likelihood function and its gradient */
/****************************************************************************************/

/*********** Marginal/integrated log-likelihood function ************/
// @title Compute the marginal log-likelihood 
// @description This function computes the profile loglikelihood function.
//
// @param par a list of parameters. 
// @param y a numerical vector of output
// @param d a list of distances.
// @param covmodel a string indicating the type of covariance model
// @param bound Default value is \code{NULL}. Otherwise, it should be a list
// containing the following elements depending on the covariance class:
// \itemize{
// \item{For nugget parameter \strong{nugget}, it lies in the interval \eqn{(0, 1)}.
// It is a list containing lower bound \strong{lb} and 
// upper bound \strong{ub} with default value 
// \code{{nugget}=Rcpp::List{lb=0, ub=1}}.}
// \item{For the Confluent Hypergeometric covariance class, correlation parameters consis of
// \strong{range} and \strong{tail}. \strong{range} is a Rcpp::List containing
// lower bound \strong{lb} and upper bound \strong{ub} with default value
// \code{{range}=Rcpp::List{lb=1e-20, ub=1e10}}. \strong{tail} is a Rcpp::List
// containing lower bound \strong{lb} and upper bound \strong{ub} with 
// default value \code{{tail}=Rcpp::List{lb=1e-5, ub=6}}.}
// \item{For the Matérn covariance, exponential covariance, Gaussian 
// covariance, and powered-exponential covariance, the range parameter 
//  has suppport \eqn{(0, \infty)}. The log inverse range parameterization
//  is used: \eqn{\xi:=-\log(\phi)}. There is no need to specify \strong{bound}.}
//  \item{For Cauchy covariance, \strong{bound} is specified for the 
//  tail decay parameter \strong{tail}. \strong{tail} is a Rcpp::List containing
//  lower bound \strong{lb} and upper bound \strong{ub} with default value
//  \code{{tail}=Rcpp::List{lb=1e-5, ub=2}}.}
// }
// @param nugget_est a logical value. If it is \code{TRUE}, the nugget parameter 
// will be estimated; otherwise the nugget is not included in the covariance
// model.


// Integrated/Marginal loglikelihood L(theta; y)
// theta contains the correlation parameter on the 
double UQ::MLoglik(const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
  const Eigen::VectorXd& nu, const double& nugget, const Eigen::MatrixXd& y, 
  const Eigen::MatrixXd& H, const Rcpp::List& d, const Rcpp::List& covmodel){

  int n = y.rows(); // number of model runs
  int q = y.cols(); // number of output variables
  int p = H.cols(); // number of covariates 
  Eigen::MatrixXd R(n,n), RInv(n,n), Q(n,n), RH(n,p), HRH(p,p);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH, ldltq;


  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  if(form=="tensor"){
    R = UQ::tensor_kernel(d, range, tail, nu, family);
  }else if(form=="ARD"){
    R = UQ::ARD_kernel(d, range, tail(0), nu(0), family);
  }else{
    Rcpp::stop("The covariance kernel is not supported yet.\n");
  }
  

  R.diagonal().array() += nugget;

  ldltR.compute(R);
  double lndetR = ldltR.vectorD().array().log().sum();
  RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
  
  RH = RInv * H;
  HRH = H.transpose() * RH;
  ldltH.compute(HRH);

  Q = RInv - RH*ldltH.solve(RH.transpose());
  Eigen::MatrixXd S2 = y.transpose()*Q*y; 
  double lndetHRH = ldltH.vectorD().array().log().sum();
  ldltq.compute(S2);
  double lndetS2 = ldltq.vectorD().array().log().sum();

  double loglik = -0.5*q*lndetR -0.5*q*lndetHRH - 0.5*(n-p)*lndetS2;

  return loglik;

}


// Derivative of the marginal likelihood w.r.t. each parameter 
Eigen::VectorXd UQ::gradient_MLoglik(const Eigen::VectorXd& range, 
  const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, const double& nugget, 
  const Eigen::MatrixXd& y, const Eigen::MatrixXd& H, const Rcpp::List& d, const Rcpp::List& covmodel,
  const bool& smoothness_est){

  int n = y.rows();
  int q = y.cols();
  int p = H.cols();
  //int dim = d.size();
  Eigen::MatrixXd R(n,n), RInv(n,n), Q(n,n), RH(n,p), HRH(p,p);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH;



  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  Rcpp::List dR;
  if(form=="tensor"){
    R = UQ::tensor_kernel(d, range, tail, nu, family);
    dR = UQ::deriv_tensor_kernel(d, range, tail, nu, family);
  }else if(form=="ARD"){
    R = UQ::ARD_kernel(d, range, tail(0), nu(0), family);
    dR = UQ::deriv_ARD_kernel(d, range, tail(0), nu(0), family);
  }else{
    Rcpp::stop("The covariance kernel is not supported yet.\n");
  } 

  R.diagonal().array() += nugget;

  ldltR.compute(R);
  RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
  
  RH = RInv * H;
  HRH = H.transpose() * RH;
  ldltH.compute(HRH);

  Q = RInv - RH*ldltH.solve(RH.transpose());
  Eigen::MatrixXd S2 = y.transpose()*Q*y; 


  Eigen::MatrixXd O(n,n), Qy(n,q);
  Qy = Q*y;
  Eigen::MatrixXd SyQ = S2.llt().solve(Qy.transpose());
  O = Q - Qy * SyQ;

  // Rcpp::List dR = deriv_kernel(d, par, covmodel);

  int len = dR.size() -1; // record the number of parameters after excluding smoothness parameter
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


// Profile log-likelihood, which is proportion to the integrated log-likelihood
double UQ::PLoglik(const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
  const Eigen::VectorXd& nu, const double& nugget, const Eigen::MatrixXd& y, 
  const Eigen::MatrixXd& H, const Rcpp::List& d, const Rcpp::List& covmodel){

 double loglik;
 loglik = UQ::MLoglik(range, tail, nu, nugget, y,  H, d, covmodel);

  return loglik;
}


/*****************************************************************************************/
/*****************************************************************************************/

/*****************************************************************************************/
/*****************************************************************************************/


/*****************************************************************************************/
/*****************************************************************************************/
// log of marginal posterior = log of integrated likelihood + log of reference prior
double UQ::mposterior_ref_range(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
  Rcpp::List& d, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
      const Eigen::VectorXd& nu, const double& nugget, const Rcpp::List& covmodel,
  const bool& nugget_est, const bool& smoothness_est){

  int n = output.rows();
  int q = H.cols();
  int dim = d.size();
  Eigen::MatrixXd R(n,n), RInv(n,n), Q(n,n), RH(n,q), HRH(q,q);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH;

  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  Rcpp::List dR;
  if(form=="tensor"){
    R = UQ::tensor_kernel(d, range, tail, nu, family);
    dR = UQ::deriv_tensor_kernel(d, range, tail, nu, family);
  }else if(form=="ARD"){
    R = UQ::ARD_kernel(d, range, tail(0), nu(0), family);
    dR = UQ::deriv_ARD_kernel(d, range, tail(0), nu(0), family);
  }else{
    Rcpp::stop("The covariance kernel is not supported yet.\n");
  }

  R.diagonal().array() += nugget;

  ldltR.compute(R);
  RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
  
  RH = RInv * H;
  HRH = H.transpose() * RH;
  ldltH.compute(HRH);
  Q = RInv - RH*ldltH.solve(RH.transpose());

  Eigen::MatrixXd S2 = output.transpose()*Q*output;

  // compute log-likelihood
  double loglik = -0.5*ldltR.vectorD().array().log().sum() - 0.5*ldltH.vectorD().array().log().sum() 
                  -0.5*(n-q)*S2.ldlt().vectorD().array().log().sum();

  // compute log-of reference prior
  // Rcpp::List dR = deriv_kernel(d, par, covmodel);

  Rcpp::List W(dim+1);
  Eigen::MatrixXd W_l(n, n), W_k(n, n);

  for(int k=0; k<dim; k++){
    W[k] = Rcpp::as<Eigen::MatrixXd>(dR[k])*Q;
  }

  int count = dim+1; // record the actual dimension of FisherIR

  Eigen::MatrixXd FisherIR(dim+2, dim+2);


  if(nugget_est){ //include nugget

    W[dim] = Q;  // corresponding to nugget

    FisherIR(0,0) = n - q;
    for(int l=0; l<(dim+1); l++){
      W_l = Rcpp::as<Eigen::MatrixXd>(W[l]);

      FisherIR(0, l+1) = W_l.trace();
      FisherIR(l+1, 0) = W_l.trace();

      for(int k=0; k<(dim+1); k++){
        W_k = Rcpp::as<Eigen::MatrixXd>(W[k]);
        FisherIR(l+1, k+1) = (W_l*W_k).trace();
        FisherIR(k+1, l+1) = (W_l*W_k).trace();
      }
    }

    count = dim + 2;    

  }else{


    FisherIR(0,0) = n - q;
    for(int l=0; l<dim; l++){
      W_l = Rcpp::as<Eigen::MatrixXd>(W[l]);
      FisherIR(0, l+1) = W_l.trace();
      FisherIR(l+1, 0) = W_l.trace();

      for(int k=0; k<dim; k++){
        W_k = Rcpp::as<Eigen::MatrixXd>(W[k]);
        FisherIR(l+1, k+1) = (W_l*W_k).trace();
        FisherIR(k+1, l+1) = (W_l*W_k).trace();
      }

    }  

    count = dim + 1;

  }

  ldltR.compute(FisherIR.block(0,0,count,count));
  double lndetI = 0.5*ldltR.vectorD().array().log().sum();

  if(family=="CH" || family =="cauchy"){
    lndetI += -(1.0 + tail.array() * tail.array()).log().sum();
  }

  if(nugget_est){
    lndetI += -log(1.0 + nugget*nugget);
  }  

  // compute the marginal posterior
  double posterior = loglik + lndetI;

  return posterior;

}

/*****************************************************************************************/
/*****************************************************************************************/

/*****************************************************************************************/
/*****************************************************************************************/
// log of marginal posterior = log of integrated likelihood + log of cauchy prior
double UQ::mposterior_cauchy_range(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
                                   const Rcpp::List& d, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
      const Eigen::VectorXd& nu, const double& nugget, const Rcpp::List& covmodel,
  const bool& nugget_est, const bool& smoothness_est){

  int n = output.rows();
  int q = H.cols();
  //int dim = d.size();
  Eigen::MatrixXd R(n,n), RInv(n,n), Q(n,n), RH(n,q), HRH(q,q);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH;


  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  if(form=="tensor"){
    R = UQ::tensor_kernel(d, range, tail, nu, family);
  }else if(form=="ARD"){
    R = UQ::ARD_kernel(d, range, tail(0), nu(0), family);
  }else{
    Rcpp::stop("The covariance kernel is not supported yet.\n");
  }

  R.diagonal().array() += nugget;

  ldltR.compute(R);
  RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
  
  RH = RInv * H;
  HRH = H.transpose() * RH;
  ldltH.compute(HRH);
  Q = RInv - RH*ldltH.solve(RH.transpose());

  Eigen::MatrixXd S2 = output.transpose()*Q*output;

  // compute log-likelihood
  double loglik = -0.5*ldltR.vectorD().array().log().sum() - 0.5*ldltH.vectorD().array().log().sum() 
                -0.5*(n-q)*S2.ldlt().vectorD().array().log().sum();

  // compute log-of cauchy prior
  double logprior=0.0;
  logprior = -(1.0 + range.array() * range.array()).log().sum();

  if(family=="CH" || family =="cauchy"){
    logprior += -(1.0 + tail.array() * tail.array()).log().sum();
  }

  if(nugget_est){
    logprior += -log(1.0 + nugget*nugget);
  }

  // compute the marginal posterior
  double posterior = loglik + logprior;

  return posterior;

}

/*****************************************************************************************/
/*****************************************************************************************/


/*****************************************************************************************/
/*****************************************************************************************/
// gradient of log of marginal posterior = gradient of log of integrated likelihood + gradient log of cauchy prior
Eigen::VectorXd UQ::grad_mposterior_cauchy_range(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
  Rcpp::List& d, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
  const Eigen::VectorXd& nu, const double& nugget, const Rcpp::List& covmodel,
  const bool& smoothness_est){

  //int n = y.size();
  //int q = H.cols();
  int dim = d.size();

  Eigen::VectorXd grad_mloglik = UQ::gradient_MLoglik(range, tail, nu, nugget, output, H, d, 
                                covmodel, smoothness_est);

  int npar = grad_mloglik.size();

  Eigen::VectorXd gradf = grad_mloglik;


  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);



  for(int k=0; k<dim; k++){
    gradf(k) += - 2.0*range(k) / (1.0+range(k)*range(k));
  }

  if(family=="CH" || family=="cauchy"){
    if(form=="tensor"){
      for(int k=0; k<dim; k++){
        gradf(dim+k) += -2.0*tail(k) / (1.0+tail(k)*tail(k));
      } 
    }else if(form=="ARD"){
      gradf(dim) += -2.0*tail(0) / (1.0+tail(0)*tail(0));
    }else{
      Rcpp::stop("The covariance kernel is not supported yet.\n");
    }

  }

  
  // w.r.t. nugget
    gradf(npar) += -2.0*nugget / (1.0+nugget*nugget);
  
  // w.r.t. smoothness parameter


  return gradf;

}

/*****************************************************************************************/
/*****************************************************************************************/


/*****************************************************************************************/
/*****************************************************************************************/
Eigen::MatrixXd UQ::simulate(const Eigen::MatrixXd& input, const Eigen::MatrixXd& H, 
  const Eigen::MatrixXd& coeff, const double& sig2, const Eigen::VectorXd& range, 
  const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, const double& nugget,  
  const Rcpp::List& covmodel, const int& nsample){

  int n = input.rows();
  Rcpp::List d = UQ::adist(input, input);

  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);
  Eigen::MatrixXd R(n,n);

  if(form=="tensor"){
    R = UQ::tensor_kernel(d, range, tail, nu, family);
  }else if(form=="ARD"){
    R = UQ::ARD_kernel(d, range, tail(0), nu(0), family);
  }else{
    Rcpp::stop("The form of the covariance kernel is not implemented.\n");
  }


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
Rcpp::List UQ::predict(const Eigen::MatrixXd& y, const Eigen::MatrixXd& H, 
  const Eigen::MatrixXd& input, const Eigen::MatrixXd& input_new, 
  const Eigen::MatrixXd& Hnew, const Eigen::VectorXd& range, 
  const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, const double& nugget,
  const Rcpp::List& covmodel){

  int n = y.rows();
  int q = y.cols();
  int m = input_new.rows();
  int p = H.cols();
  int dim = input.cols();
  Eigen::MatrixXd R(n,n), RInv(n,n), RH(n,p), HRH(p,p), Rnew(n,m), HRHInv(p,p);
  Eigen::MatrixXd res(n,q), Ry(n,q), bhat(p,q);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH, ldlt;


  Rcpp::List d(dim), d0(dim);

  Eigen::MatrixXd sig2hat(q,q), predmean(m, q);

  d = UQ::adist(input, input);
  d0 = UQ::adist(input, input_new);

  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  if(form=="tensor"){
    R = UQ::tensor_kernel(d, range, tail, nu, family);
    Rnew = UQ::tensor_kernel(d0, range, tail, nu, family);
  }else if(form=="ARD"){
    R = UQ::ARD_kernel(d, range, tail(0), nu(0), family);
    Rnew = UQ::ARD_kernel(d0, range, tail(0), nu(0), family);
  }else{
    Rcpp::stop("The covariance kernel is not supported yet.\n");
  }
  R.diagonal().array() += nugget;

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

  // return Rcpp::List::create(Rcpp::_["mean"] = predmean, 
  //                           Rcpp::_["corr"] = pred_corr,
  //                           Rcpp::_["sig2"] = sig2hat,
  //                           Rcpp::_["df"] = n-p
  // ); 

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
                              Rcpp::_["df"] = n-p,
                              Rcpp::_["input_corr"] = pred_corr,
                              Rcpp::_["output_cov"] = sig2hat
    );

}


/*****************************************************************************************/
/*****************************************************************************************/
// Simulate from the Predictive distribution 
Rcpp::List UQ::tensor_simulate_predictive_dist(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
  const Eigen::MatrixXd& input, const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew, 
  const Eigen::MatrixXd& range, const Eigen::MatrixXd& tail, const Eigen::MatrixXd& nu, 
  const Eigen::VectorXd& nugget, const Rcpp::List& covmodel){

  int n = output.rows();
  int q = output.cols();
  int m = input_new.rows();
  int p = H.cols();
  int nsample = range.rows();

  int dim = input.cols();
  Eigen::MatrixXd R(n,n), RInv(n,n), RH(n,p), HRH(p,p), Rnew(n,m), HRHInv(p,p);
  Eigen::MatrixXd res(n,q), Ry(n,q), bhat(p,q);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH, ldlt;

  double df = n-p;
  Rcpp::List d(dim), d0(dim);

  Eigen::MatrixXd sig2hat(q,q), predmean(m, q), L(q,q);

  Eigen::MatrixXd ysim(m,q); 

  d = UQ::adist(input, input);
  d0 = UQ::adist(input, input_new);

  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  Eigen::VectorXd Rtmp(n), tmp(p), pred_corr(m);

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

    R = UQ::tensor_kernel(d, range.row(it), tail.row(it), nu.row(it), family);
    Rnew = UQ::tensor_kernel(d0, range.row(it), tail.row(it), nu.row(it), family);

    R.diagonal().array() += nugget(it);

    ldltR.compute(R);
    RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
    RH = RInv * H;
    HRH = H.transpose() * RH;
    ldltH.compute(HRH);
    Ry = RInv*output;

    bhat = ldltH.solve(H.transpose()*Ry);
    res = output - H*bhat;
    // Rnew = UQ::kernel(d0, par, covmodel);
    predmean = Hnew*bhat;
    predmean += Rnew.transpose()*(RInv*res);
    sig2hat = res.transpose() * RInv*res / df;

    HRHInv = ldltH.solve(Eigen::MatrixXd::Identity(p,p));

    
    for(int k=0; k<m; k++){
      Rtmp = Rnew.col(k);
      tmp = Hnew.row(k) - RH.transpose()*Rtmp;
      pred_corr(k) = R(0,0) - Rtmp.dot(RInv*Rtmp) + tmp.dot(HRHInv*tmp);
    }

    // simulate from posterior predictive distribution
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



Rcpp::List UQ::ARD_simulate_predictive_dist(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
  const Eigen::MatrixXd& input, const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew, 
  const Eigen::MatrixXd& range, const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, 
  const Eigen::VectorXd& nugget, const Rcpp::List& covmodel){

  int n = output.rows();
  int q = output.cols();
  int m = input_new.rows();
  int p = H.cols();
  int nsample = range.rows();

  int dim = input.cols();
  Eigen::MatrixXd R(n,n), RInv(n,n), RH(n,p), HRH(p,p), Rnew(n,m), HRHInv(p,p);
  Eigen::MatrixXd res(n,q), Ry(n,q), bhat(p,q);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH, ldlt;

  double df = n-p;
  Rcpp::List d(dim), d0(dim);

  Eigen::MatrixXd sig2hat(q,q), predmean(m, q), L(q,q);

  Eigen::MatrixXd ysim(m,q); 

  d = UQ::adist(input, input);
  d0 = UQ::adist(input, input_new);

  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  Eigen::VectorXd Rtmp(n), tmp(p), pred_corr(m);

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

    R = UQ::ARD_kernel(d, range.row(it), tail(it), nu(it), family);
    Rnew = UQ::ARD_kernel(d0, range.row(it), tail(it), nu(it), family);

    R.diagonal().array() += nugget(it);

    ldltR.compute(R);
    RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
    RH = RInv * H;
    HRH = H.transpose() * RH;
    ldltH.compute(HRH);
    Ry = RInv*output;

    bhat = ldltH.solve(H.transpose()*Ry);
    res = output - H*bhat;
    // Rnew = UQ::kernel(d0, par, covmodel);
    predmean = Hnew*bhat;
    predmean += Rnew.transpose()*(RInv*res);
    sig2hat = res.transpose() * RInv*res / df;

    HRHInv = ldltH.solve(Eigen::MatrixXd::Identity(p,p));

    
    for(int k=0; k<m; k++){
      Rtmp = Rnew.col(k);
      tmp = Hnew.row(k) - RH.transpose()*Rtmp;
      pred_corr(k) = R(0,0) - Rtmp.dot(RInv*Rtmp) + tmp.dot(HRHInv*tmp);
    }

    // simulate from posterior predictive distribution
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
// Conditional simulation
Rcpp::List UQ::tensor_condsim(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
  const Eigen::MatrixXd& input, const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew, 
  const Eigen::VectorXd& range, const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, 
  const double& nugget, const Rcpp::List& covmodel, int nsample){

  int n = output.rows();
  int q = output.cols();
  int m = input_new.rows();
  int p = H.cols();

  int dim = input.cols();
  Eigen::MatrixXd R(n,n), RInv(n,n), RH(n,p), HRH(p,p), Rnew(n,m), HRHInv(p,p);
  Eigen::MatrixXd res(n,q), Ry(n,q), bhat(p,q);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH, ldlt;

  double df = n-p;
  Rcpp::List d(dim), d0(dim);

  Eigen::MatrixXd sig2hat(q,q), predmean(m, q), L(q,q);

  Eigen::MatrixXd ysim(m,q); 

  d = UQ::adist(input, input);
  d0 = UQ::adist(input, input_new);

  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  Eigen::VectorXd Rtmp(n), tmp(p), pred_corr(m);

  Rcpp::List ysim_sample(nsample);

#ifdef USE_R
    GetRNGstate();
#endif



    R = UQ::tensor_kernel(d, range, tail, nu, family);
    Rnew = UQ::tensor_kernel(d0, range, tail, nu, family);

    R.diagonal().array() += nugget;
    for(int i=0; i<n; i++){
      for(int j=0; j<m; j++){
        if(Rnew(i,j)==1.0){
          Rnew(i,j) += nugget;
        }
      }
    }

    ldltR.compute(R);
    RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
    RH = RInv * H;
    HRH = H.transpose() * RH;
    ldltH.compute(HRH);
    Ry = RInv*output;

    bhat = ldltH.solve(H.transpose()*Ry);
    res = output - H*bhat;
    // Rnew = UQ::kernel(d0, par, covmodel);
    predmean = Hnew*bhat;
    predmean += Rnew.transpose()*(RInv*res);
    sig2hat = res.transpose() * RInv*res / df;

    HRHInv = ldltH.solve(Eigen::MatrixXd::Identity(p,p));

    
    for(int k=0; k<m; k++){
      Rtmp = Rnew.col(k);
      tmp = Hnew.row(k) - RH.transpose()*Rtmp;
      pred_corr(k) = R(0,0) - Rtmp.dot(RInv*Rtmp) + tmp.dot(HRHInv*tmp);
    }

    // simulate from posterior predictive distribution
    L = sig2hat.llt().matrixL();

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


Rcpp::List UQ::ARD_condsim(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
  const Eigen::MatrixXd& input, const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew, 
  const Eigen::VectorXd& range, const double& tail, const double& nu, 
  const double& nugget, const Rcpp::List& covmodel, int nsample){

  int n = output.rows();
  int q = output.cols();
  int m = input_new.rows();
  int p = H.cols();

  int dim = input.cols();
  Eigen::MatrixXd R(n,n), RInv(n,n), RH(n,p), HRH(p,p), Rnew(n,m), HRHInv(p,p);
  Eigen::MatrixXd res(n,q), Ry(n,q), bhat(p,q);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH, ldlt;

  double df = n-p;
  Rcpp::List d(dim), d0(dim);

  Eigen::MatrixXd sig2hat(q,q), predmean(m, q), L(q,q);

  Eigen::MatrixXd ysim(m,q); 

  d = UQ::adist(input, input);
  d0 = UQ::adist(input, input_new);

  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  Eigen::VectorXd Rtmp(n), tmp(p), pred_corr(m);

  Rcpp::List ysim_sample(nsample);

#ifdef USE_R
    GetRNGstate();
#endif



    R = UQ::ARD_kernel(d, range, tail, nu, family);
    Rnew = UQ::ARD_kernel(d0, range, tail, nu, family);

    R.diagonal().array() += nugget;

    for(int i=0; i<n; i++){
      for(int j=0; j<m; j++){
        if(Rnew(i,j)==1.0){
          Rnew(i,j) += nugget;
        }
      }
    }

    ldltR.compute(R);
    RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
    RH = RInv * H;
    HRH = H.transpose() * RH;
    ldltH.compute(HRH);
    Ry = RInv*output;

    bhat = ldltH.solve(H.transpose()*Ry);
    res = output - H*bhat;
    // Rnew = UQ::kernel(d0, par, covmodel);
    predmean = Hnew*bhat;
    predmean += Rnew.transpose()*(RInv*res);
    sig2hat = res.transpose() * RInv*res / df;

    HRHInv = ldltH.solve(Eigen::MatrixXd::Identity(p,p));

    
    for(int k=0; k<m; k++){
      Rtmp = Rnew.col(k);
      tmp = Hnew.row(k) - RH.transpose()*Rtmp;
      pred_corr(k) = R(0,0) - Rtmp.dot(RInv*Rtmp) + tmp.dot(HRHInv*tmp);
    }

    // simulate from posterior predictive distribution
    L = sig2hat.llt().matrixL();

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
// unconstrained parameter space 
/*****************************************************************************************/
// MCMC for ARD covariance kernels with RW on unconstrained parameter space
Rcpp::List UQ::ARD_MCMCOBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
  const Eigen::MatrixXd& input, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
  const Eigen::VectorXd& nu, const double& nugget, const Rcpp::List& covmodel, 
  const bool& smoothness_est, const Rcpp::List& proposal, const int& nsample, const bool& verbose){

  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  Rcpp::List d = UQ::adist(input, input);
  int Dim = d.size();
  
  Eigen::VectorXd Delta_range;
  if(proposal.containsElementNamed("range")){
    Delta_range = Rcpp::as<Eigen::VectorXd>(proposal["range"]);
  }else{
    Delta_range = 0.1*Eigen::VectorXd::Ones(Dim);
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

  double loglik_curr = UQ::MLoglik(range, tail, nu, nugget, output, H, d, covmodel);
  double loglik_prop=0.0;
  
  // initial values for parameters  
  Eigen::VectorXd range_curr(Dim), tail_curr(1), nu_curr(1);
  double nugget_curr;
  range_curr = range;
  nu_curr(0) = nu(0);
  tail_curr(0) = tail(0);
  nugget_curr = nugget;


  Eigen::MatrixXd range_sample(nsample, Dim);
  Eigen::VectorXd tail_sample(nsample), nu_sample(nsample), nugget_sample(nsample);

  Eigen::VectorXd range_prop(Dim), tail_prop(1), nu_prop(1);
  double nugget_prop;
  range_prop = range_curr;
  tail_prop = tail_curr;
  nu_prop = nu_curr;
  nugget_prop = nugget_curr;

  // MCMC stuff 
  double log_prior_curr, log_prior_prop, log_pr_tran, log_pr_rtran, MH_ratio;
  double unif_rnd;
  Rcpp::LogicalMatrix accept_rate_range(nsample, Dim);
  Rcpp::LogicalVector accept_rate_nugget(nsample), accept_rate_tail(nsample), accept_rate_nu(nsample);
  double Jacobian_curr=0, Jacobian_prop=0;

  // uniform prior on nu 
  double nu_lb = 0.0, nu_ub = 6;
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
  if(family=="CH" || family=="cauchy"){ 

    // begin MCMC sampling 
    for(int it=0; it<nsample; it++){
      if(Progress::check_abort()){
        return R_NilValue;
      }
      prog.increment();

      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
      // update range parameter 
      for(int k=0; k<Dim; k++){
        // generate proposal
        range_prop(k) = exp(Rcpp::rnorm(1, log(range_curr(k)), Delta_range(k))[0]);
        loglik_prop = UQ::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          range_sample(it,k) = range_curr(k);
        }else{
          // log prior density of cauchy dist
          log_prior_curr = -log(1.0+range_curr(k)*range_curr(k));
          log_prior_prop = -log(1.0+range_prop(k)*range_prop(k));

          // log proposal density: RW on log(range)
          log_pr_tran =  0;
          log_pr_rtran = 0;

          // Jacobian 
          Jacobian_curr = log(range_curr(k));
          Jacobian_prop = log(range_prop(k));

          MH_ratio = (loglik_prop - loglik_curr) + (log_prior_prop-log_prior_curr) 
                          + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_range(it,k) = TRUE;
            range_curr(k) = range_prop(k);
            loglik_curr = loglik_prop;
          }

          range_sample(it,k) = range_curr(k);
        }
      }

      // update tail decay parameter
      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
      // generate proposal
      tail_prop(0) = exp(Rcpp::rnorm(1, log(tail_curr(0)), Delta_tail)[0]);
      loglik_prop = UQ::MLoglik(range_curr, tail_prop, nu_curr, nugget_curr, output, H, d, covmodel);

      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        tail_sample(it) = tail_curr(0);
      }else{
        // log prior density of cauchy dist
        log_prior_curr = -log(1.0+tail_curr(0)*tail_curr(0));
        log_prior_prop = -log(1.0+tail_prop(0)*tail_prop(0));

        // log proposal density: Random Walk
        log_pr_tran =  0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(tail_curr(0));
        Jacobian_prop = log(tail_prop(0));
        MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                      + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_tail[it] = TRUE;
          tail_curr(0) = tail_prop(0);
          loglik_curr = loglik_prop;
        }

        tail_sample(it) = tail_curr(0);
      }

      // if(nugget_est){

        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        // generate proposal
        nugget_prop = exp(Rcpp::rnorm(1, log(nugget_curr), Delta_nugget)[0]);
        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nugget_sample(it) = nugget_curr;
        }else{
          // log prior density of cauchy dist
          log_prior_curr = -log(1.0+nugget_curr*nugget_curr);
          log_prior_prop = -log(1.0+nugget_prop*nugget_prop);

          // log proposal density 
          log_pr_tran =  0;
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

      // update smoothness parameter
      if(smoothness_est){

        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        // generate proposal
        // nu_prop(0) = exp(Rcpp::rnorm(1, log(nu_curr(0)), Delta_nu)[0]);
        nu_prop(0) = ilogit(Rcpp::rnorm(1, logit(nu_curr(0), nu_lb, nu_ub), Delta_nu)[0], nu_lb, nu_ub);   

        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nu_sample(it) = nu_curr(0);
        }else{
          // log prior density of uniform dist
          log_prior_curr = 0;
          log_prior_prop = 0;

          // log proposal density 
          log_pr_tran =  0;
          log_pr_rtran = 0;

          // Jacobian 
          // Jacobian_curr = log(nu_curr(0));
          // Jacobian_prop = log(nu_prop(0));
          Jacobian_curr = log(nu_curr(0)-nu_lb) + log(nu_ub-nu_curr(0));
          Jacobian_prop = log(nu_prop(0)-nu_lb) + log(nu_ub-nu_prop(0));

          MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                          + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_nu[it] = TRUE;
            nu_curr(0) = nu_prop(0);
            loglik_curr = loglik_prop;
          }

          nu_sample(it) = nu_curr(0);
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

      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // update range parameter 
      for(int k=0; k<Dim; k++){
        range_prop = range_curr;
        // generate proposal
        range_prop(k) = exp(Rcpp::rnorm(1, log(range_curr(k)), Delta_range(k))[0]);
        loglik_prop = UQ::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // log prior density of cauchy dist
        log_prior_curr = -log(1.0+range_curr(k)*range_curr(k));
        log_prior_prop = -log(1.0+range_prop(k)*range_prop(k));

        // log proposal density 
        log_pr_tran = 0;
        log_pr_rtran = 0;

        // Jacobian
        Jacobian_curr = log(range_curr(k));
        Jacobian_prop = log(range_prop(k));
        MH_ratio = (loglik_prop - loglik_curr) + (log_prior_prop-log_prior_curr) 
                       + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_range(it,k) = TRUE;
          range_curr(k) = range_prop(k);
          loglik_curr = loglik_prop;
        }

        range_sample(it,k) = range_curr(k);
      }

      // if(nugget_est){

        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // generate proposal
        nugget_prop = exp(Rcpp::rnorm(1, log(nugget_curr), Delta_nugget)[0]);
        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

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
          accept_rate_nugget[it]=TRUE;
          nugget_curr = nugget_prop;
          loglik_curr = loglik_prop;
        }

        nugget_sample(it) = nugget_curr;
      // }

      // update smoothness parameter
      if(smoothness_est){

        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // generate proposal
        // nu_prop(0) = exp(Rcpp::rnorm(1, log(nu_curr(0)), Delta_nu)[0]);
        nu_prop(0) = ilogit(Rcpp::rnorm(1, logit(nu_curr(0), nu_lb, nu_ub), Delta_nu)[0], nu_lb, nu_ub);   

        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

        // log prior density 
        // log_prior_curr = -log(1.0+nu_curr(0)*nu_curr(0));
        // log_prior_prop = -log(1.0+nu_prop(0)*nu_prop(0));
        log_prior_curr = 0;
        log_prior_prop = 0;

        // log proposal density 
        log_pr_tran =  0;
        log_pr_rtran = 0;

        // Jacobian 
        // Jacobian_curr = log(nu_curr(0));
        // Jacobian_prop = log(nu_prop(0));
        Jacobian_curr = log(nu_curr(0)-nu_lb) + log(nu_ub-nu_curr(0));
        Jacobian_prop = log(nu_prop(0)-nu_lb) + log(nu_ub-nu_prop(0));

        MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_nu[it] = TRUE;
          nu_curr(0) = nu_prop(0);
          loglik_curr = loglik_prop;
        }

        nu_sample(it) = nu_curr(0);
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
                                Rcpp::_["accept_rate_nu"]=accept_rate_nu);      
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
                                Rcpp::_["accept_rate_nu"]=accept_rate_nu);      
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


// MCMC for ARD covariance kernels with RW on unconstrained parameter space
Rcpp::List UQ::ARD_MCMCSBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
  const Eigen::MatrixXd& input, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
  const Eigen::VectorXd& nu, const double& nugget, const Rcpp::List& covmodel, 
  const bool& smoothness_est, const Rcpp::List& prior, const Rcpp::List& proposal, const int& nsample, const bool & verbose){


  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  Rcpp::List d = UQ::adist(input, input);
  int Dim = d.size();
  
  Eigen::VectorXd Delta_range;
  if(proposal.containsElementNamed("range")){
    Delta_range = Rcpp::as<Eigen::VectorXd>(proposal["range"]);
  }else{
    Delta_range = 0.1*Eigen::VectorXd::Ones(Dim);
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
  Eigen::VectorXd range_a=Eigen::VectorXd::Ones(Dim), range_b=Eigen::VectorXd::Ones(Dim);
  Eigen::VectorXd range_lb=Eigen::VectorXd::Zero(Dim);
  Eigen::VectorXd range_ub(Dim);
  for(int k=0; k<Dim; k++){
    range_ub(k) = 3.0*(Rcpp::as<Eigen::MatrixXd>(d[k])).maxCoeff();
  } 
  if(prior.containsElementNamed("range")){
    range_prior = Rcpp::as<Rcpp::List>(prior["range"]);
    if(range_prior.containsElementNamed("a")){
      int n_a = (Rcpp::as<Eigen::VectorXd>(range_prior["a"])).size();
      if(n_a==Dim){
        range_a = Rcpp::as<Eigen::VectorXd>(range_prior["a"]);
      }
      if(n_a==1){
        range_a = Rcpp::as<double>(range_prior["a"])*Eigen::VectorXd::Ones(Dim);
      }       
    }

    if(range_prior.containsElementNamed("b")){
      int n_b = (Rcpp::as<Eigen::VectorXd>(range_prior["b"])).size();
      if(n_b==Dim){
        range_b = Rcpp::as<Eigen::VectorXd>(range_prior["b"]);
      }
      if(n_b==1){
        range_b = Rcpp::as<double>(range_prior["b"])*Eigen::VectorXd::Ones(Dim);
      } 
    }

    if(range_prior.containsElementNamed("lb")){
      int n_lb = (Rcpp::as<Eigen::VectorXd>(range_prior["lb"])).size();
      if(n_lb==Dim){
        range_lb = Rcpp::as<Eigen::VectorXd>(range_prior["lb"]);
      }
      if(n_lb==1){
        range_lb = Rcpp::as<double>(range_prior["lb"])*Eigen::VectorXd::Ones(Dim);
      }
    }

    if(range_prior.containsElementNamed("ub")){
      int n_ub = (Rcpp::as<Eigen::VectorXd>(range_prior["ub"])).size();
      if(n_ub==Dim){
        range_ub = Rcpp::as<Eigen::VectorXd>(range_prior["ub"]);
      }
      if(n_ub==1){
        range_ub = Rcpp::as<double>(range_prior["ub"])*Eigen::VectorXd::Ones(Dim);
      }
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
  double nu_lb=0.1, nu_ub=6.0;
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

  double loglik_curr = UQ::MLoglik(range, tail, nu, nugget, output, H, d, covmodel);
  double loglik_prop=0.0;
  
  // initial values for parameters  
  Eigen::VectorXd range_curr(Dim), tail_curr(1), nu_curr(1);
  double nugget_curr;
  range_curr = range;
  nu_curr(0) = nu(0);
  tail_curr(0) = tail(0);
  nugget_curr = nugget;


  Eigen::MatrixXd range_sample(nsample, Dim);
  Eigen::VectorXd tail_sample(nsample), nu_sample(nsample), nugget_sample(nsample);

  Eigen::VectorXd range_prop(Dim), tail_prop(1), nu_prop(1);
  double nugget_prop;
  range_prop = range_curr;
  tail_prop = tail_curr;
  nu_prop = nu_curr;
  nugget_prop = nugget_curr;

  // MCMC stuff 
  double log_prior_curr, log_prior_prop, log_pr_tran, log_pr_rtran, MH_ratio;
  double unif_rnd;
  Rcpp::LogicalMatrix accept_rate_range(nsample, Dim);
  Rcpp::LogicalVector accept_rate_nugget(nsample), accept_rate_tail(nsample), accept_rate_nu(nsample);
  double Jacobian_curr=0, Jacobian_prop=0;

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

      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
      // update range parameter 
      for(int k=0; k<Dim; k++){
        // generate proposal
        range_prop(k) = ilogit(Rcpp::rnorm(1, logit(range_curr(k), range_lb(k), range_ub(k)), Delta_range(k))[0], range_lb(k), range_ub(k));
        loglik_prop = UQ::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          range_sample(it,k) = range_curr(k);
        }else{
          // log prior density of cauchy dist
          log_prior_curr = (range_a(k)-1.0)*log(range_curr(k)-range_lb(k)) + (range_b(k)-1.0)*log(range_ub(k)-range_curr(k));
          log_prior_prop = (range_a(k)-1.0)*log(range_prop(k)-range_lb(k)) + (range_b(k)-1.0)*log(range_ub(k)-range_prop(k));

          // log proposal density: random work on log(range)
          // log_pr_tran = -log(range_prop(k)) - (log(range_prop(k)) - log(range_curr(k)))*(log(range_prop(k)) - log(range_curr(k))) / (2.0*Delta_range(k)*Delta_range(k));
          // log_pr_rtran = -log(range_curr(k)) - (log(range_prop(k)) - log(range_curr(k)))*(log(range_prop(k)) - log(range_curr(k))) / (2.0*Delta_range(k)*Delta_range(k));
          log_pr_tran =  0;
          log_pr_rtran = 0;

          // Jacobian 
          Jacobian_curr = log(range_curr(k)-range_lb(k)) + log(range_ub(k)-range_curr(k));
          Jacobian_prop = log(range_prop(k)-range_lb(k)) + log(range_ub(k)-range_prop(k));

          MH_ratio = (loglik_prop - loglik_curr) + (log_prior_prop-log_prior_curr) 
                          + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_range(it, k) = TRUE;
            range_curr(k) = range_prop(k);
            loglik_curr = loglik_prop;
          }

          range_sample(it,k) = range_curr(k);
        }
      }

      // update tail decay parameter
      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
      // generate proposal
      tail_prop(0) = ilogit(Rcpp::rnorm(1, logit(tail_curr(0), tail_lb, tail_ub), Delta_tail)[0], tail_lb, tail_ub);
      loglik_prop = UQ::MLoglik(range_curr, tail_prop, nu_curr, nugget_curr, output, H, d, covmodel);

      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        tail_sample(it) = tail_curr(0);
      }else{
        // log prior density of cauchy dist
        log_prior_curr = (tail_a-1.0)*log(tail_curr(0)-tail_lb) + (tail_b-1.0)*log(tail_ub-tail_curr(0));
        log_prior_prop = (tail_a-1.0)*log(tail_prop(0)-tail_lb) + (tail_b-1.0)*log(tail_ub-tail_prop(0));


        // log proposal density 
        log_pr_tran =  0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(tail_curr(0)-tail_lb) + log(tail_ub-tail_curr(0));
        Jacobian_prop = log(tail_prop(0)-tail_lb) + log(tail_ub-tail_prop(0));

        MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                      + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_tail[it]= TRUE;
          tail_curr(0) = tail_prop(0);
          loglik_curr = loglik_prop;
        }

        tail_sample(it) = tail_curr(0);
      }

      // if(nugget_est){

        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        // generate proposal
        nugget_prop = ilogit(Rcpp::rnorm(1, logit(nugget_curr, nugget_lb, nugget_ub), Delta_nugget)[0], nugget_lb, nugget_ub);
        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nugget_sample(it) = nugget_curr;
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
            accept_rate_nugget[it]=TRUE;
            nugget_curr = nugget_prop;
            loglik_curr = loglik_prop;
          }

          nugget_sample(it) = nugget_curr;
        }
      // } 

      // update smoothness parameter
      if(smoothness_est){

        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        // generate proposal
        nu_prop(0) = ilogit(Rcpp::rnorm(1, logit(nu_curr(0), nu_lb, nu_ub), Delta_nu)[0], nu_lb, nu_ub);
        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nu_sample(it) = nu_curr(0);
        }else{
          // log prior density of cauchy dist
          log_prior_curr = (nu_a-1.0)*log(nu_curr(0)-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_curr(0));
          log_prior_prop = (nu_a-1.0)*log(nu_prop(0)-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_prop(0));

          // log proposal density 
          log_pr_tran =  0;
          log_pr_rtran = 0;

          // Jacobian 
          Jacobian_curr = log(nu_curr(0)-nu_lb) + log(nu_ub-nu_curr(0));
          Jacobian_prop = log(nu_prop(0)-nu_lb) + log(nu_ub-nu_prop(0));

          MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                          + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_nu[it]=TRUE;
            nu_curr(0) = nu_prop(0);
            loglik_curr = loglik_prop;
          }

          nu_sample(it) = nu_curr(0);
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

      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // update range parameter 
      for(int k=0; k<Dim; k++){

        // generate proposal
        range_prop(k) = ilogit(Rcpp::rnorm(1, logit(range_curr(k), range_lb(k), range_ub(k)), Delta_range(k))[0], range_lb(k), range_ub(k));
        loglik_prop = UQ::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // log prior density of cauchy dist
        log_prior_curr = (range_a(k)-1.0)*log(range_curr(k)-range_lb(k)) + (range_b(k)-1.0)*log(range_ub(k)-range_curr(k));
        log_prior_prop = (range_a(k)-1.0)*log(range_prop(k)-range_lb(k)) + (range_b(k)-1.0)*log(range_ub(k)-range_prop(k));

        // log proposal density 
        log_pr_tran = 0;
        log_pr_rtran = 0;

        // Jacobian
        Jacobian_curr = log(range_curr(k)-range_lb(k)) + log(range_ub(k)-range_curr(k));
        Jacobian_prop = log(range_prop(k)-range_lb(k)) + log(range_ub(k)-range_prop(k));

        MH_ratio = (loglik_prop - loglik_curr) + (log_prior_prop-log_prior_curr) 
                       + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_range(it,k) = TRUE;
          range_curr(k) = range_prop(k);
          loglik_curr = loglik_prop;
        }

        range_sample(it,k) = range_curr(k);
      }

      // if(nugget_est){

        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // generate proposal
        nugget_prop = ilogit(Rcpp::rnorm(1, logit(nugget_curr, nugget_lb, nugget_ub), Delta_nugget)[0], nugget_lb, nugget_ub);
        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

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
          accept_rate_nugget[it]=TRUE;
          nugget_curr = nugget_prop;
          loglik_curr = loglik_prop;
        }

        nugget_sample(it) = nugget_curr;
      // }

      // update smoothness parameter
      if(smoothness_est){

        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // generate proposal
        nu_prop(0) = ilogit(Rcpp::rnorm(1, logit(nu_curr(0), nu_lb, nu_ub), Delta_nu)[0], nu_lb, nu_ub);

        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

        // log prior density of cauchy dist
        log_prior_curr = (nu_a-1.0)*log(nu_curr(0)-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_curr(0));
        log_prior_prop = (nu_a-1.0)*log(nu_prop(0)-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_prop(0));

        // log proposal density 
        log_pr_tran =  0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(nu_curr(0)-nu_lb) + log(nu_ub-nu_curr(0));
        Jacobian_prop = log(nu_prop(0)-nu_lb) + log(nu_ub-nu_prop(0));

        MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_nu[it]=TRUE;
          nu_curr(0) = nu_prop(0);
          loglik_curr = loglik_prop;
        }

        nu_sample(it) = nu_curr(0);
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
                                Rcpp::_["accept_rate_nu"]=accept_rate_nu);      
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
                                Rcpp::_["accept_rate_nu"]=accept_rate_nu);      
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


// MCMC for tensor covariance kernels with RW on unconstrained parameter space
Rcpp::List UQ::tensor_MCMCOBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
  const Eigen::MatrixXd& input, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
  const Eigen::VectorXd& nu, const double& nugget, const Rcpp::List& covmodel, 
  const bool& smoothness_est, const Rcpp::List& proposal, const int& nsample, const bool& verbose){


  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  Rcpp::List d = UQ::adist(input, input);
  int Dim = d.size();


  Eigen::VectorXd Delta_range;
  if(proposal.containsElementNamed("range")){
    Delta_range = Rcpp::as<Eigen::VectorXd>(proposal["range"]);
  }else{
    Delta_range = 0.1*Eigen::VectorXd::Ones(Dim);
  }

  Eigen::VectorXd Delta_tail; // sd in the LN proposal distribution.
  if(proposal.containsElementNamed("tail")){
    Delta_tail = Rcpp::as<Eigen::VectorXd>(proposal["tail"]);
  }else{
    Delta_tail = 0.1*Eigen::VectorXd::Ones(Dim);
  }

  double Delta_nugget;
  if(proposal.containsElementNamed("nugget")){
    Delta_nugget = Rcpp::as<double>(proposal["nugget"]);
  }else{
    Delta_nugget = 0.1;
  }
  
  Eigen::VectorXd Delta_nu;
  if(proposal.containsElementNamed("nu")){
    Delta_nu = Rcpp::as<Eigen::VectorXd>(proposal["nu"]);
  }else{
    Delta_nu = 0.1*Eigen::VectorXd::Ones(Dim);
  }

  double loglik_curr = UQ::MLoglik(range, tail, nu, nugget, output, H, d, covmodel);
  double loglik_prop=0.0;
  
  // initial values for parameters 
  Eigen::VectorXd range_curr = range;
  Eigen::VectorXd tail_curr = tail;
  Eigen::VectorXd nu_curr = nu;
  double nugget_curr = nugget;


  Eigen::MatrixXd range_sample(nsample, Dim);
  Eigen::MatrixXd tail_sample(nsample, Dim);
  Eigen::MatrixXd nu_sample(nsample, Dim);
  Eigen::VectorXd nugget_sample(nsample);

  Eigen::VectorXd range_prop(Dim), tail_prop(Dim), nu_prop(Dim);
  double nugget_prop;
  range_prop = range_curr;
  tail_prop = tail_curr;
  nu_prop = nu_curr;
  nugget_prop = nugget_curr;

  // MCMC stuff
  double log_prior_curr, log_prior_prop, log_pr_tran, log_pr_rtran, MH_ratio;
  double unif_rnd;
  Rcpp::LogicalMatrix accept_rate_range(nsample, Dim), accept_rate_tail(nsample, Dim);
  Rcpp::LogicalVector accept_rate_nu(nsample), accept_rate_nugget(nsample);
  double Jacobian_curr, Jacobian_prop;

  // uniform prior U(nu_lb, nu_ub) on smoothness parameter nu
  double nu_lb = 0.0, nu_ub = 6.0;
  if(family=="cauchy" || family=="powexp"){
    nu_ub = 2.0;
  }
  // create an empty list for abortion 
  // Rcpp::List abort_value = Rcpp::List::create(Rcpp::_["abort"]=R_NilValue);
  int k=0;

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

      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // update range parameter 
      for(int k=0; k<Dim; k++){

        // generate proposal
        range_prop(k) = exp(Rcpp::rnorm(1, log(range_curr(k)), Delta_range(k))[0]);
        loglik_prop = UQ::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          range_sample(it,k) = range_curr(k);
        }else{
          // log prior density of cauchy dist
          log_prior_curr = -log(1.0+range_curr(k)*range_curr(k));
          log_prior_prop = -log(1.0+range_prop(k)*range_prop(k));

          // log proposal density 
          log_pr_tran =  0;
          log_pr_rtran = 0;

          // Jacobian
          Jacobian_curr = log(range_curr(k));
          Jacobian_prop = log(range_prop(k));

          MH_ratio = (loglik_prop - loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_range(it,k) = TRUE;
            range_curr(k) = range_prop(k);
            loglik_curr = loglik_prop;
          }
          range_sample(it,k) = range_curr(k);
        }

          
      }

      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // update tail decay parameter
      for(int k=0; k<Dim; k++){

        // generate proposal
        tail_prop(k) = exp(Rcpp::rnorm(1, log(tail_curr(k)), Delta_tail(k))[0]);
        loglik_prop = UQ::MLoglik(range_curr, tail_prop, nu_curr, nugget_curr, output, H, d, covmodel);

        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          tail_sample(it,k) = tail_curr(k);
        }else{
          // log prior density of cauchy dist
          log_prior_curr = -log(1.0+tail_curr(k)*tail_curr(k));
          log_prior_prop = -log(1.0+tail_prop(k)*tail_prop(k));

          // log proposal density 
          log_pr_tran = 0;
          log_pr_rtran = 0;

          // Jacobian 
          Jacobian_curr = log(tail_curr(k));
          Jacobian_prop = log(tail_prop(k));

          MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_tail(it,k) = TRUE;
            tail_curr(k) = tail_prop(k);
            loglik_curr = loglik_prop;
          }

          tail_sample(it,k) = tail_curr(k);
        }
      }

      //if(nugget_est){
        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // generate proposal
        nugget_prop = exp(Rcpp::rnorm(1, log(nugget_curr), Delta_nugget)[0]);
        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nugget_sample(it) = nugget_curr;
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
            accept_rate_nugget[it]=TRUE;
            nugget_curr = nugget_prop;
            loglik_curr = loglik_prop;
          }
          nugget_sample(it) = nugget_curr;
        }
      //}    

      // update smoothness parameter
      if(smoothness_est){
        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        k = 0;
        

          // generate proposal
          // nu_prop(k) = exp(Rcpp::rnorm(1, log(nu_curr(k)), Delta_nu(k))[0]);
          nu_prop(k) = ilogit(Rcpp::rnorm(1, logit(nu_curr(k), nu_lb, nu_ub), Delta_nu(k))[0], nu_lb, nu_ub);   

          loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);
          if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
            nu_sample(it,k) = nu_curr(k);
          }else{
            // log prior density of cauchy dist
            // log_prior_curr = -log(1.0+nu_curr(k)*nu_curr(k));
            // log_prior_prop = -log(1.0+nu_prop(k)*nu_prop(k));
            log_prior_curr = 0;
            log_prior_prop = 0;

            // log proposal density 
            log_pr_tran = 0;
            log_pr_rtran = 0;

            // Jacobian 
            // Jacobian_curr = log(nu_curr(k));
            // Jacobian_prop = log(nu_prop(k));
            Jacobian_curr = log(nu_curr(k)-nu_lb) + log(nu_ub-nu_curr(k));
            Jacobian_prop = log(nu_prop(k)-nu_lb) + log(nu_ub-nu_prop(k));

            MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                          + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

            unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
            if(log(unif_rnd)<MH_ratio){ // accept 
              accept_rate_nu[it] = TRUE;
              nu_curr(k) = nu_prop(k);
              loglik_curr = loglik_prop;
            }

            nu_sample(it,k) = nu_curr(k);
          }

          for(int j=1; j<Dim; j++){
            nu_curr(j) = nu_curr(k);
          }

        
      } // end smooth

      
    } 


  }else if(family=="matern" || family=="powexp" || family=="gauss") {

    // begin MCMC sampling 
    for(int it=0; it<nsample; it++){
      if(Progress::check_abort()){
        return R_NilValue;
      }
      prog.increment();

      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // update range parameter 
      for(int k=0; k<Dim; k++){

        // generate proposal
        range_prop(k) = exp(Rcpp::rnorm(1, log(range_curr(k)), Delta_range(k))[0]);
        loglik_prop = UQ::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // log prior density of cauchy dist
        log_prior_curr = -log(1.0+range_curr(k)*range_curr(k));
        log_prior_prop = -log(1.0+range_prop(k)*range_prop(k));

        // log proposal density 
        log_pr_tran =  0;
        log_pr_rtran = 0;

        // Jacobian
        Jacobian_curr = log(range_curr(k));
        Jacobian_prop = log(range_prop(k));

        MH_ratio = (loglik_prop - loglik_curr) + (log_prior_prop-log_prior_curr) 
                      + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_range(it,k) = TRUE;
          range_curr(k) = range_prop(k);
          loglik_curr = loglik_prop;
        }

        range_sample(it,k) = range_curr(k);
      }

      // if(nugget_est){
        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        // generate proposal
        nugget_prop = exp(Rcpp::rnorm(1, log(nugget_curr), Delta_nugget)[0]);
        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

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
          accept_rate_nugget[it]=TRUE;
          nugget_curr = nugget_prop;
          loglik_curr = loglik_prop;
        }
        nugget_sample(it) = nugget_curr;
      // }

      // update smoothness parameter
      if(smoothness_est){

        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        // update smoothness parameter
        k = 0; // common smoothness parameter across each dimension
        // for(int k=0; k<Dim; k++){

          // generate proposal
          // nu_prop(k) = exp(Rcpp::rnorm(1, log(nu_curr(k)), Delta_nu(k))[0]);
          nu_prop(k) = ilogit(Rcpp::rnorm(1, logit(nu_curr(k), nu_lb, nu_ub), Delta_nu(k))[0], nu_lb, nu_ub);   

          loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

          // log prior density of cauchy dist
          // log_prior_curr = -log(1.0+nu_curr(k)*nu_curr(k));
          // log_prior_prop = -log(1.0+nu_prop(k)*nu_prop(k));
          log_prior_curr = 0;
          log_prior_prop = 0;

          // log proposal density 
          log_pr_tran = 0;
          log_pr_rtran = 0;

          // Jacobian 
          // Jacobian_curr = log(nu_curr(k));
          // Jacobian_prop = log(nu_prop(k));
          Jacobian_curr = log(nu_curr(k)-nu_lb) + log(nu_ub-nu_curr(k));
          Jacobian_prop = log(nu_prop(k)-nu_lb) + log(nu_ub-nu_prop(k));

          MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_nu(it) = TRUE;
            nu_curr(k) = nu_prop(k);
            loglik_curr = loglik_prop;
          }

          nu_sample(it,k) = nu_curr(k);
          for(int j=1; j<Dim; j++){
            nu_curr(j) = nu_curr(k);
          }
        // }
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
                                Rcpp::_["accept_rate_nu"]=accept_rate_nu);      
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
                                Rcpp::_["accept_rate_nu"]=accept_rate_nu);      
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


// MCMC for tensor covariance kernels with RW on unconstrained parameter space
Rcpp::List UQ::tensor_MCMCSBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
  const Eigen::MatrixXd& input, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
  const Eigen::VectorXd& nu, const double& nugget, const Rcpp::List& covmodel, 
  const bool& smoothness_est, const Rcpp::List& prior, const Rcpp::List& proposal, const int& nsample, const bool& verbose){


  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  Rcpp::List d = UQ::adist(input, input);
  int Dim = d.size();


  Eigen::VectorXd Delta_range;
  if(proposal.containsElementNamed("range")){
    Delta_range = Rcpp::as<Eigen::VectorXd>(proposal["range"]);
  }else{
    Delta_range = 0.1*Eigen::VectorXd::Ones(Dim);
  }

  Eigen::VectorXd Delta_tail; // sd in the LN proposal distribution.
  if(proposal.containsElementNamed("tail")){
    Delta_tail = Rcpp::as<Eigen::VectorXd>(proposal["tail"]);
  }else{
    Delta_tail = 0.1*Eigen::VectorXd::Ones(Dim);
  }

  double Delta_nugget;
  if(proposal.containsElementNamed("nugget")){
    Delta_nugget = Rcpp::as<double>(proposal["nugget"]);
  }else{
    Delta_nugget = 0.1;
  }
  
  Eigen::VectorXd Delta_nu;
  if(proposal.containsElementNamed("nu")){
    Delta_nu = Rcpp::as<Eigen::VectorXd>(proposal["nu"]);
  }else{
    Delta_nu = 0.1*Eigen::VectorXd::Ones(Dim);
  }

  // tuning parameters in priors
  int n_a, n_b, n_lb, n_ub;
  Rcpp::List range_prior; //beta(a, b, lb, ub) 
  Eigen::VectorXd range_a=Eigen::VectorXd::Ones(Dim), range_b=Eigen::VectorXd::Ones(Dim);
  Eigen::VectorXd range_lb=Eigen::VectorXd::Zero(Dim);
  Eigen::VectorXd range_ub(Dim);
  for(int k=0; k<Dim; k++){
    range_ub(k) = 3.0*(Rcpp::as<Eigen::MatrixXd>(d[k])).maxCoeff();
  } 
  if(prior.containsElementNamed("range")){
    range_prior = Rcpp::as<Rcpp::List>(prior["range"]);
    if(range_prior.containsElementNamed("a")){
      n_a = (Rcpp::as<Eigen::VectorXd>(range_prior["a"])).size();
      if(n_a==Dim){
        range_a = Rcpp::as<Eigen::VectorXd>(range_prior["a"]);
      }
      if(n_a==1){
        range_a = Rcpp::as<double>(range_prior["a"])*Eigen::VectorXd::Ones(Dim);
      }       
    }

    if(range_prior.containsElementNamed("b")){
      n_b = (Rcpp::as<Eigen::VectorXd>(range_prior["b"])).size();
      if(n_b==Dim){
        range_b = Rcpp::as<Eigen::VectorXd>(range_prior["b"]);
      }
      if(n_b==1){
        range_b = Rcpp::as<double>(range_prior["b"])*Eigen::VectorXd::Ones(Dim);
      } 
    }

    if(range_prior.containsElementNamed("lb")){
      n_lb = (Rcpp::as<Eigen::VectorXd>(range_prior["lb"])).size();
      if(n_lb==Dim){
        range_lb = Rcpp::as<Eigen::VectorXd>(range_prior["lb"]);
      }
      if(n_lb==1){
        range_lb = Rcpp::as<double>(range_prior["lb"])*Eigen::VectorXd::Ones(Dim);
      }
    }

    if(range_prior.containsElementNamed("ub")){
      n_ub = (Rcpp::as<Eigen::VectorXd>(range_prior["ub"])).size();
      if(n_ub==Dim){
        range_ub = Rcpp::as<Eigen::VectorXd>(range_prior["ub"]);
      }
      if(n_ub==1){
        range_ub = Rcpp::as<double>(range_prior["ub"])*Eigen::VectorXd::Ones(Dim);
      }
    }    
  }

  Rcpp::List tail_prior; //beta(a, b, lb, ub) 
  Eigen::VectorXd tail_a=Eigen::VectorXd::Ones(Dim), tail_b=Eigen::VectorXd::Ones(Dim);
  Eigen::VectorXd tail_lb= Eigen::VectorXd::Zero(Dim);
  Eigen::VectorXd tail_ub = 5.0*Eigen::VectorXd::Ones(Dim); 
  if(prior.containsElementNamed("tail")){
    tail_prior = Rcpp::as<Rcpp::List>(prior["tail"]);
    if(tail_prior.containsElementNamed("a")){
      n_a = (Rcpp::as<Eigen::VectorXd>(tail_prior["a"])).size();
      if(n_a==Dim){
        tail_a = Rcpp::as<Eigen::VectorXd>(tail_prior["a"]);
      }
      if(n_a==1){
        tail_a = Rcpp::as<double>(tail_prior["a"])*Eigen::VectorXd::Ones(Dim);
      }
    }
    if(tail_prior.containsElementNamed("b")){
      n_b = (Rcpp::as<Eigen::VectorXd>(tail_prior["b"])).size();
      if(n_b==Dim){
        tail_a = Rcpp::as<Eigen::VectorXd>(tail_prior["b"]);
      }
      if(n_b==1){
        tail_a = Rcpp::as<double>(tail_prior["b"])*Eigen::VectorXd::Ones(Dim);
      }
    }
    if(tail_prior.containsElementNamed("lb")){
      n_lb = (Rcpp::as<Eigen::VectorXd>(tail_prior["lb"])).size();
      if(n_lb==Dim){
        tail_a = Rcpp::as<Eigen::VectorXd>(tail_prior["lb"]);
      }
      if(n_lb==1){
        tail_a = Rcpp::as<double>(tail_prior["lb"])*Eigen::VectorXd::Ones(Dim);
      }
    }
    if(tail_prior.containsElementNamed("ub")){
      n_ub = (Rcpp::as<Eigen::VectorXd>(tail_prior["ub"])).size();
      if(n_ub==Dim){
        tail_ub = Rcpp::as<Eigen::VectorXd>(tail_prior["ub"]);
      }
      if(n_ub==1){
        tail_ub = Rcpp::as<double>(tail_prior["ub"])*Eigen::VectorXd::Ones(Dim);
      }
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
  double nu_lb=0.1, nu_ub=6.0;
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

  double loglik_curr = UQ::MLoglik(range, tail, nu, nugget, output, H, d, covmodel);
  double loglik_prop=0.0;
  
  // initial values for parameters 
  Eigen::VectorXd range_curr = range;
  Eigen::VectorXd tail_curr = tail;
  Eigen::VectorXd nu_curr = nu;
  double nugget_curr = nugget;


  Eigen::MatrixXd range_sample(nsample, Dim);
  Eigen::MatrixXd tail_sample(nsample, Dim);
  Eigen::MatrixXd nu_sample(nsample, Dim);
  Eigen::VectorXd nugget_sample(nsample);

  Eigen::VectorXd range_prop(Dim), tail_prop(Dim), nu_prop(Dim);
  double nugget_prop;
  range_prop = range_curr;
  tail_prop = tail_curr;
  nu_prop = nu_curr;
  nugget_prop = nugget_curr;

  // MCMC stuff
  double log_prior_curr, log_prior_prop, log_pr_tran, log_pr_rtran, MH_ratio;
  double unif_rnd;
  Rcpp::LogicalMatrix accept_rate_range(nsample,Dim), accept_rate_tail(nsample,Dim);
  Rcpp::LogicalVector accept_rate_nu(nsample), accept_rate_nugget(nsample);
  double Jacobian_curr, Jacobian_prop;


  // create an empty list for abortion 
  // Rcpp::List abort_value = Rcpp::List::create(Rcpp::_["abort"]=R_NilValue);
  int k=0;

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

      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // update range parameter 
      for(int k=0; k<Dim; k++){

        // generate proposal
        range_prop(k) = ilogit(Rcpp::rnorm(1, logit(range_curr(k), range_lb(k), range_ub(k)), Delta_range(k))[0], range_lb(k), range_ub(k));
        loglik_prop = UQ::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          range_sample(it,k) = range_curr(k);
        }else{
          // log prior density of beta dist
          log_prior_curr = (range_a(k)-1.0)*log(range_curr(k)-range_lb(k)) + (range_b(k)-1.0)*log(range_ub(k)-range_curr(k));
          log_prior_prop = (range_a(k)-1.0)*log(range_prop(k)-range_lb(k)) + (range_b(k)-1.0)*log(range_ub(k)-range_prop(k));

          // log proposal density 
          log_pr_tran =  0;
          log_pr_rtran = 0;

          // Jacobian 
          Jacobian_curr = log(range_curr(k)-range_lb(k)) + log(range_ub(k)-range_curr(k));
          Jacobian_prop = log(range_prop(k)-range_lb(k)) + log(range_ub(k)-range_prop(k));

          MH_ratio = (loglik_prop - loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_range(it,k) = TRUE;
            range_curr(k) = range_prop(k);
            loglik_curr = loglik_prop;
          }

          range_sample(it,k) = range_curr(k);
        }
      }

      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // update tail decay parameter
      for(int k=0; k<Dim; k++){

        // generate proposal
        tail_prop(k) = ilogit(Rcpp::rnorm(1, logit(tail_curr(k), tail_lb(k), tail_ub(k)), Delta_tail(k))[0], tail_lb(k), tail_ub(k));
        loglik_prop = UQ::MLoglik(range_curr, tail_prop, nu_curr, nugget_curr, output, H, d, covmodel);
        
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          tail_sample(it,k) = tail_curr(k);
        }else{
          // log prior density
          log_prior_curr = (tail_a(k)-1.0)*log(tail_curr(k)-tail_lb(k)) + (tail_b(k)-1.0)*log(tail_ub(k)-tail_curr(k));
          log_prior_prop = (tail_a(k)-1.0)*log(tail_prop(k)-tail_lb(k)) + (tail_b(k)-1.0)*log(tail_ub(k)-tail_prop(k));

          // log proposal density 
          log_pr_tran = 0;
          log_pr_rtran = 0;

          // Jacobian 
          Jacobian_curr = log(tail_curr(k)-tail_lb(k)) + log(tail_ub(k)-tail_curr(k));
          Jacobian_prop = log(tail_prop(k)-tail_lb(k)) + log(tail_ub(k)-tail_prop(k));

          MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_tail(it,k) = TRUE;
            tail_curr(k) = tail_prop(k);
            loglik_curr = loglik_prop;
          }

          tail_sample(it,k) = tail_curr(k);
        }
      }

      //if(nugget_est){
        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // generate proposal
        nugget_prop = ilogit(Rcpp::rnorm(1, logit(nugget_curr, nugget_lb, nugget_ub), Delta_nugget)[0], nugget_lb, nugget_ub);
        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nugget_sample(it) = nugget_curr;
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
            accept_rate_nugget[it]=TRUE;
            nugget_curr = nugget_prop;
            loglik_curr = loglik_prop;
          }
          nugget_sample(it) = nugget_curr;
        }
      //}    

      // update smoothness parameter
      if(smoothness_est){
        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        k = 0;
        // for(int k=0; k<Dim; k++){

          // generate proposal
          nu_prop(k) = ilogit(Rcpp::rnorm(1, logit(nu_curr(k), nu_lb, nu_ub), Delta_nu(k))[0], nu_lb, nu_ub);
          loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

          if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
            nu_sample(it,k) = nu_curr(k);
          }else{
            // log prior density of cauchy dist
            log_prior_curr = (nu_a-1.0)*log(nu_curr(k)-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_curr(k));
            log_prior_prop = (nu_a-1.0)*log(nu_prop(k)-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_prop(k));

            // log proposal density 
            log_pr_tran = 0;
            log_pr_rtran = 0;

            // Jacobian 
            Jacobian_curr = log(nu_curr(k)-nu_lb) + log(nu_ub-nu_curr(k));
            Jacobian_prop = log(nu_prop(k)-nu_lb) + log(nu_ub-nu_prop(k));

            MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                          + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

            unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
            if(log(unif_rnd)<MH_ratio){ // accept 
              accept_rate_nu(it) = TRUE;
              nu_curr(k) = nu_prop(k);
              loglik_curr = loglik_prop;
            }
            nu_sample(it,k) = nu_curr(k);
          }

          for(int j=1; j<Dim; j++){
            nu_curr(j) = nu_curr(k);
          }
        // }
      }


    } 


  }else if(family=="matern" || family=="powexp" || family=="gauss") {

    // begin MCMC sampling 
    for(int it=0; it<nsample; it++){
      if(Progress::check_abort()){
        return R_NilValue;
      }
      prog.increment();

      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // update range parameter 
      for(int k=0; k<Dim; k++){

        // generate proposal
        range_prop(k) = ilogit(Rcpp::rnorm(1, logit(range_curr(k), range_lb(k), range_ub(k)), Delta_range(k))[0], range_lb(k), range_ub(k));
        loglik_prop = UQ::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // log prior density of cauchy dist
        log_prior_curr = (range_a(k)-1.0)*log(range_curr(k)-range_lb(k)) + (range_b(k)-1.0)*log(range_ub(k)-range_curr(k));
        log_prior_prop = (range_a(k)-1.0)*log(range_prop(k)-range_lb(k)) + (range_b(k)-1.0)*log(range_ub(k)-range_prop(k));

        // log proposal density 
        log_pr_tran =  0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(range_curr(k)-range_lb(k)) + log(range_ub(k)-range_curr(k));
        Jacobian_prop = log(range_prop(k)-range_lb(k)) + log(range_ub(k)-range_prop(k));

        MH_ratio = (loglik_prop - loglik_curr) + (log_prior_prop-log_prior_curr) 
                      + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_range(it,k) = TRUE;
          range_curr(k) = range_prop(k);
          loglik_curr = loglik_prop;
        }

        range_sample(it,k) = range_curr(k);
      }

      // if(nugget_est){
        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        // generate proposal
        nugget_prop = ilogit(Rcpp::rnorm(1, logit(nugget_curr, nugget_lb, nugget_ub), Delta_nugget)[0], nugget_lb, nugget_ub);
        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

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
          accept_rate_nugget[it]=TRUE;
          nugget_curr = nugget_prop;
          loglik_curr = loglik_prop;
        }
        nugget_sample(it) = nugget_curr;
      // }

      // update smoothness parameter
      if(smoothness_est){

        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        // update smoothness parameter
        k = 0; // common smoothness parameter across each dimension
        // for(int k=0; k<Dim; k++){

          // generate proposal
          nu_prop(k) = ilogit(Rcpp::rnorm(1, logit(nu_curr(k), nu_lb, nu_ub), Delta_nu(k))[0], nu_lb, nu_ub);
          loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

          // log prior density of cauchy dist
          log_prior_curr = (nu_a-1.0)*log(nu_curr(k)-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_curr(k));
          log_prior_prop = (nu_a-1.0)*log(nu_prop(k)-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_prop(k));

          // log proposal density 
          log_pr_tran = 0;
          log_pr_rtran = 0;

          // Jacobian 
          Jacobian_curr = log(nu_curr(k)-nu_lb) + log(nu_ub-nu_curr(k));
          Jacobian_prop = log(nu_prop(k)-nu_lb) + log(nu_ub-nu_prop(k));

          MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_nu[it] = TRUE;
            nu_curr(k) = nu_prop(k);
            loglik_curr = loglik_prop;
          }

          nu_sample(it,k) = nu_curr(k);
          for(int j=1; j<Dim; j++){
            nu_curr(j) = nu_curr(k);
          }
        // }
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
                                Rcpp::_["accept_rate_nu"]=accept_rate_nu);      
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
                                Rcpp::_["accept_rate_nu"]=accept_rate_nu);      
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

/**********************************************************************************/
/**********************************************************************************/

/*****************************************************************************************/
// MCMC for ARD covariance kernels with RW on unconstrained parameter space
Rcpp::List UQ::ARD_MCMCOBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
  const Eigen::MatrixXd& input, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
  const Eigen::VectorXd& nu, const double& nugget, const Rcpp::List& covmodel, 
  const bool& smoothness_est, const Rcpp::List& proposal, const int& nsample, const bool& verbose,
  const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew){


  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  Rcpp::List d = UQ::adist(input, input);
  int Dim = d.size();
  
  Eigen::VectorXd Delta_range;
  if(proposal.containsElementNamed("range")){
    Delta_range = Rcpp::as<Eigen::VectorXd>(proposal["range"]);
  }else{
    Delta_range = 0.1*Eigen::VectorXd::Ones(Dim);
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

  double loglik_curr = UQ::MLoglik(range, tail, nu, nugget, output, H, d, covmodel);
  double loglik_prop=0.0;
  
  // initial values for parameters  
  Eigen::VectorXd range_curr(Dim), tail_curr(1), nu_curr(1);
  double nugget_curr;
  range_curr = range;
  nu_curr(0) = nu(0);
  tail_curr(0) = tail(0);
  nugget_curr = nugget;


  Eigen::MatrixXd range_sample(nsample, Dim);
  Eigen::VectorXd tail_sample(nsample), nu_sample(nsample), nugget_sample(nsample);
  nu_sample = nu_curr(0)*Eigen::VectorXd::Ones(nsample);

  Eigen::VectorXd range_prop(Dim), tail_prop(1), nu_prop(1);
  double nugget_prop;
  range_prop = range_curr;
  tail_prop = tail_curr;
  nu_prop = nu_curr;
  nugget_prop = nugget_curr;

  // MCMC stuff 
  double log_prior_curr, log_prior_prop, log_pr_tran, log_pr_rtran, MH_ratio;
  double unif_rnd;
  Rcpp::LogicalMatrix accept_rate_range(nsample, Dim);
  Rcpp::LogicalVector accept_rate_nugget(nsample), accept_rate_tail(nsample), accept_rate_nu(nsample);
  double Jacobian_curr=0, Jacobian_prop=0;

  // uniform prior on nu
  double nu_lb = 0.0, nu_ub = 6.0;
  if(family=="cauchy" || family=="powexp"){
    nu_ub = 2.0;
  }
  /************************* Prediction ****************************************/
  int n = output.rows();
  int q = output.cols();
  int m = input_new.rows();
  int p = H.cols();
  int dim = input.cols();
  Eigen::MatrixXd R(n,n), RInv(n,n), RH(n,p), HRH(p,p), Rnew(n,m), HRHInv(p,p);
  Eigen::MatrixXd res(n,q), Ry(n,q), bhat(p,q);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH, ldlt;

  double df = n-p;
  Rcpp::List d0(dim);

  Eigen::MatrixXd sig2hat(q,q), predmean(m, q), L(q,q);

  Eigen::VectorXd Rtmp(n), tmp(p), pred_corr(m);

  Eigen::MatrixXd ysim(m,q);
  Rcpp::List ysim_sample(nsample); 

  //d = UQ::adist(input, input);
  d0 = UQ::adist(input, input_new);

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

      /************************************* Parameter Estimation ************************************/
      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
      // update range parameter 
      for(int k=0; k<Dim; k++){
        // generate proposal
        range_prop(k) = exp(Rcpp::rnorm(1, log(range_curr(k)), Delta_range(k))[0]);
        loglik_prop = UQ::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          range_sample(it,k) = range_curr(k);
        }else{
          // log prior density of cauchy dist
          log_prior_curr = -log(1.0+range_curr(k)*range_curr(k));
          log_prior_prop = -log(1.0+range_prop(k)*range_prop(k));

          // log proposal density: RW on log(range)
          log_pr_tran =  0;
          log_pr_rtran = 0;

          // Jacobian 
          Jacobian_curr = log(range_curr(k));
          Jacobian_prop = log(range_prop(k));

          MH_ratio = (loglik_prop - loglik_curr) + (log_prior_prop-log_prior_curr) 
                          + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_range(it,k) = TRUE;
            range_curr(k) = range_prop(k);
            loglik_curr = loglik_prop;
          }

          range_sample(it,k) = range_curr(k);
        }
      }

      // update tail decay parameter
      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
      // generate proposal
      tail_prop(0) = exp(Rcpp::rnorm(1, log(tail_curr(0)), Delta_tail)[0]);
      loglik_prop = UQ::MLoglik(range_curr, tail_prop, nu_curr, nugget_curr, output, H, d, covmodel);

      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        tail_sample(it) = tail_curr(0);
      }else{
        // log prior density of cauchy dist
        log_prior_curr = -log(1.0+tail_curr(0)*tail_curr(0));
        log_prior_prop = -log(1.0+tail_prop(0)*tail_prop(0));

        // log proposal density: Random Walk
        log_pr_tran =  0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(tail_curr(0));
        Jacobian_prop = log(tail_prop(0));
        MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                      + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_tail[it] = TRUE;
          tail_curr(0) = tail_prop(0);
          loglik_curr = loglik_prop;
        }

        tail_sample(it) = tail_curr(0);
      }

      // if(nugget_est){

        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        // generate proposal
        nugget_prop = exp(Rcpp::rnorm(1, log(nugget_curr), Delta_nugget)[0]);
        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nugget_sample(it) = nugget_curr;
        }else{
          // log prior density of cauchy dist
          log_prior_curr = -log(1.0+nugget_curr*nugget_curr);
          log_prior_prop = -log(1.0+nugget_prop*nugget_prop);

          // log proposal density 
          log_pr_tran =  0;
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

      // update smoothness parameter
      if(smoothness_est){

        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        // generate proposal
        // nu_prop(0) = exp(Rcpp::rnorm(1, log(nu_curr(0)), Delta_nu)[0]);
        nu_prop(0) = ilogit(Rcpp::rnorm(1, logit(nu_curr(0), nu_lb, nu_ub), Delta_nu)[0], nu_lb, nu_ub);   

        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nu_sample(it) = nu_curr(0);
        }else{
          // log prior density of cauchy dist
          // log_prior_curr = -log(1.0+nu_curr(0)*nu_curr(0));
          // log_prior_prop = -log(1.0+nu_prop(0)*nu_prop(0));
          log_prior_curr = 0;
          log_prior_prop = 0;

          // log proposal density 
          log_pr_tran =  0;
          log_pr_rtran = 0;

          // Jacobian 
          // Jacobian_curr = log(nu_curr(0));
          // Jacobian_prop = log(nu_prop(0));
          Jacobian_curr = log(nu_curr(0)-nu_lb) + log(nu_ub-nu_curr(0));
          Jacobian_prop = log(nu_prop(0)-nu_lb) + log(nu_ub-nu_prop(0));

          MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                          + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_nu[it] = TRUE;
            nu_curr(0) = nu_prop(0);
            loglik_curr = loglik_prop;
          }

          nu_sample(it) = nu_curr(0);
        }
      }       

      /*************************************************************************************/
      /*****************************Prediction**********************************/

      R = UQ::ARD_kernel(d, range_sample.row(it), tail_sample(it), nu_sample(it), family);
      Rnew = UQ::ARD_kernel(d0, range_sample.row(it), tail_sample(it), nu_sample(it), family);

      R.diagonal().array() += nugget_sample(it);

      ldltR.compute(R);
      RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
      RH = RInv * H;
      HRH = H.transpose() * RH;
      ldltH.compute(HRH);
      Ry = RInv*output;

      bhat = ldltH.solve(H.transpose()*Ry);
      res = output - H*bhat;
      // Rnew = UQ::kernel(d0, par, covmodel);
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
      /*****************************************************************************************/

    } 


  }else if(family=="matern" || family=="powexp" || family=="gauss") {

    // begin MCMC sampling 
    for(int it=0; it<nsample; it++){
      if(Progress::check_abort()){
        return R_NilValue;
      }
      prog.increment();

      /*************************************************************************************/
      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // update range parameter 
      for(int k=0; k<Dim; k++){

        // generate proposal
        range_prop(k) = exp(Rcpp::rnorm(1, log(range_curr(k)), Delta_range(k))[0]);
        loglik_prop = UQ::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // log prior density of cauchy dist
        log_prior_curr = -log(1.0+range_curr(k)*range_curr(k));
        log_prior_prop = -log(1.0+range_prop(k)*range_prop(k));

        // log proposal density 
        log_pr_tran = 0;
        log_pr_rtran = 0;

        // Jacobian
        Jacobian_curr = log(range_curr(k));
        Jacobian_prop = log(range_prop(k));
        MH_ratio = (loglik_prop - loglik_curr) + (log_prior_prop-log_prior_curr) 
                       + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_range(it,k) = TRUE;
          range_curr(k) = range_prop(k);
          loglik_curr = loglik_prop;
        }

        range_sample(it,k) = range_curr(k);
      }

      // if(nugget_est){

        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // generate proposal
        nugget_prop = exp(Rcpp::rnorm(1, log(nugget_curr), Delta_nugget)[0]);
        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

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
          accept_rate_nugget[it]=TRUE;
          nugget_curr = nugget_prop;
          loglik_curr = loglik_prop;
        }

        nugget_sample(it) = nugget_curr;
      // }

      // update smoothness parameter
      if(smoothness_est){

        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // generate proposal
        // nu_prop(0) = exp(Rcpp::rnorm(1, log(nu_curr(0)), Delta_nu)[0]);
        nu_prop(0) = ilogit(Rcpp::rnorm(1, logit(nu_curr(0), nu_lb, nu_ub), Delta_nu)[0], nu_lb, nu_ub);   

        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

        // log prior density of cauchy dist
        // log_prior_curr = -log(1.0+nu_curr(0)*nu_curr(0));
        // log_prior_prop = -log(1.0+nu_prop(0)*nu_prop(0));
        log_prior_curr = 0;
        log_prior_prop = 0;

        // log proposal density 
        log_pr_tran =  0;
        log_pr_rtran = 0;

        // Jacobian 
        // Jacobian_curr = log(nu_curr(0));
        // Jacobian_prop = log(nu_prop(0));
        Jacobian_curr = log(nu_curr(0)-nu_lb) + log(nu_ub-nu_curr(0));
        Jacobian_prop = log(nu_prop(0)-nu_lb) + log(nu_ub-nu_prop(0));

        MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_nu[it] = TRUE;
          nu_curr(0) = nu_prop(0);
          loglik_curr = loglik_prop;
        }

        nu_sample(it) = nu_curr(0);
      }

      /*************************************************************************************/
      /*****************************Prediction**********************************/

      R = UQ::ARD_kernel(d, range_sample.row(it), tail_sample(it), nu_sample(it), family);
      Rnew = UQ::ARD_kernel(d0, range_sample.row(it), tail_sample(it), nu_sample(it), family);

      R.diagonal().array() += nugget_sample(it);

      ldltR.compute(R);
      RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
      RH = RInv * H;
      HRH = H.transpose() * RH;
      ldltH.compute(HRH);
      Ry = RInv*output;

      bhat = ldltH.solve(H.transpose()*Ry);
      res = output - H*bhat;
      // Rnew = UQ::kernel(d0, par, covmodel);
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
      /*****************************************************************************************/

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
                                Rcpp::_["pred"] = ysim_sample);      
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
                                Rcpp::_["pred"] = ysim_sample);      
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

// MCMC for ARD covariance kernels with RW on unconstrained parameter space
Rcpp::List UQ::ARD_MCMCSBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
  const Eigen::MatrixXd& input, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
  const Eigen::VectorXd& nu, const double& nugget, const Rcpp::List& covmodel, 
  const bool& smoothness_est, const Rcpp::List& prior, const Rcpp::List& proposal, const int & nsample, const bool& verbose,
  const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew){


  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  Rcpp::List d = UQ::adist(input, input);
  int Dim = d.size();
  
  Eigen::VectorXd Delta_range;
  if(proposal.containsElementNamed("range")){
    Delta_range = Rcpp::as<Eigen::VectorXd>(proposal["range"]);
  }else{
    Delta_range = 0.1*Eigen::VectorXd::Ones(Dim);
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
  Eigen::VectorXd range_a=Eigen::VectorXd::Ones(Dim), range_b=Eigen::VectorXd::Ones(Dim);
  Eigen::VectorXd range_lb=Eigen::VectorXd::Zero(Dim);
  Eigen::VectorXd range_ub(Dim);
  for(int k=0; k<Dim; k++){
    range_ub(k) = 3.0*(Rcpp::as<Eigen::MatrixXd>(d[k])).maxCoeff();
  } 
  if(prior.containsElementNamed("range")){
    range_prior = Rcpp::as<Rcpp::List>(prior["range"]);
    if(range_prior.containsElementNamed("a")){
      int n_a = (Rcpp::as<Eigen::VectorXd>(range_prior["a"])).size();
      if(n_a==Dim){
        range_a = Rcpp::as<Eigen::VectorXd>(range_prior["a"]);
      }
      if(n_a==1){
        range_a = Rcpp::as<double>(range_prior["a"])*Eigen::VectorXd::Ones(Dim);
      }       
    }

    if(range_prior.containsElementNamed("b")){
      int n_b = (Rcpp::as<Eigen::VectorXd>(range_prior["b"])).size();
      if(n_b==Dim){
        range_b = Rcpp::as<Eigen::VectorXd>(range_prior["b"]);
      }
      if(n_b==1){
        range_b = Rcpp::as<double>(range_prior["b"])*Eigen::VectorXd::Ones(Dim);
      } 
    }

    if(range_prior.containsElementNamed("lb")){
      int n_lb = (Rcpp::as<Eigen::VectorXd>(range_prior["lb"])).size();
      if(n_lb==Dim){
        range_lb = Rcpp::as<Eigen::VectorXd>(range_prior["lb"]);
      }
      if(n_lb==1){
        range_lb = Rcpp::as<double>(range_prior["lb"])*Eigen::VectorXd::Ones(Dim);
      }
    }

    if(range_prior.containsElementNamed("ub")){
      int n_ub = (Rcpp::as<Eigen::VectorXd>(range_prior["ub"])).size();
      if(n_ub==Dim){
        range_ub = Rcpp::as<Eigen::VectorXd>(range_prior["ub"]);
      }
      if(n_ub==1){
        range_ub = Rcpp::as<double>(range_prior["ub"])*Eigen::VectorXd::Ones(Dim);
      }
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
  double nu_lb=0.1, nu_ub=6.0;
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

  double loglik_curr = UQ::MLoglik(range, tail, nu, nugget, output, H, d, covmodel);
  double loglik_prop=0.0;
  
  // initial values for parameters  
  Eigen::VectorXd range_curr(Dim), tail_curr(1), nu_curr(1);
  double nugget_curr;
  range_curr = range;
  nu_curr(0) = nu(0);
  tail_curr(0) = tail(0);
  nugget_curr = nugget;


  Eigen::MatrixXd range_sample(nsample, Dim);
  Eigen::VectorXd tail_sample(nsample), nu_sample(nsample), nugget_sample(nsample);
  nu_sample = nu_curr(0)*Eigen::VectorXd::Ones(nsample);

  Eigen::VectorXd range_prop(Dim), tail_prop(1), nu_prop(1);
  double nugget_prop;
  range_prop = range_curr;
  tail_prop = tail_curr;
  nu_prop = nu_curr;
  nugget_prop = nugget_curr;

  // MCMC stuff 
  double log_prior_curr, log_prior_prop, log_pr_tran, log_pr_rtran, MH_ratio;
  double unif_rnd;
  Rcpp::LogicalMatrix accept_rate_range(nsample, Dim);
  Rcpp::LogicalVector accept_rate_nugget(nsample), accept_rate_tail(nsample), accept_rate_nu(nsample);
  double Jacobian_curr=0, Jacobian_prop=0;

  /************************* Prediction ****************************************/
  int n = output.rows();
  int q = output.cols();
  int m = input_new.rows();
  int p = H.cols();
  int dim = input.cols();
  Eigen::MatrixXd R(n,n), RInv(n,n), RH(n,p), HRH(p,p), Rnew(n,m), HRHInv(p,p);
  Eigen::MatrixXd res(n,q), Ry(n,q), bhat(p,q);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH, ldlt;

  double df = n-p;
  Rcpp::List d0(dim);

  Eigen::MatrixXd sig2hat(q,q), predmean(m, q), L(q,q);

  Eigen::VectorXd Rtmp(n), tmp(p), pred_corr(m);

  Eigen::MatrixXd ysim(m,q);
  Rcpp::List ysim_sample(nsample); 

  //d = UQ::adist(input, input);
  d0 = UQ::adist(input, input_new);

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

      /************************************* Parameter Estimation ************************************/
      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
      // update range parameter 
      for(int k=0; k<Dim; k++){
        // generate proposal
        range_prop(k) = ilogit(Rcpp::rnorm(1, logit(range_curr(k), range_lb(k), range_ub(k)), Delta_range(k))[0], range_lb(k), range_ub(k));
        loglik_prop = UQ::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          range_sample(it,k) = range_curr(k);
        }else{
          // log prior density of cauchy dist
          log_prior_curr = (range_a(k)-1.0)*log(range_curr(k)-range_lb(k)) + (range_b(k)-1.0)*log(range_ub(k)-range_curr(k));
          log_prior_prop = (range_a(k)-1.0)*log(range_prop(k)-range_lb(k)) + (range_b(k)-1.0)*log(range_ub(k)-range_prop(k));

          // log proposal density: random work on log(range)
          // log_pr_tran = -log(range_prop(k)) - (log(range_prop(k)) - log(range_curr(k)))*(log(range_prop(k)) - log(range_curr(k))) / (2.0*Delta_range(k)*Delta_range(k));
          // log_pr_rtran = -log(range_curr(k)) - (log(range_prop(k)) - log(range_curr(k)))*(log(range_prop(k)) - log(range_curr(k))) / (2.0*Delta_range(k)*Delta_range(k));
          log_pr_tran =  0;
          log_pr_rtran = 0;

          // Jacobian 
          Jacobian_curr = log(range_curr(k)-range_lb(k)) + log(range_ub(k)-range_curr(k));
          Jacobian_prop = log(range_prop(k)-range_lb(k)) + log(range_ub(k)-range_prop(k));

          MH_ratio = (loglik_prop - loglik_curr) + (log_prior_prop-log_prior_curr) 
                          + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_range(it, k) = TRUE;
            range_curr(k) = range_prop(k);
            loglik_curr = loglik_prop;
          }

          range_sample(it,k) = range_curr(k);
        }
      }

      // update tail decay parameter
      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
      // generate proposal
      tail_prop(0) = ilogit(Rcpp::rnorm(1, logit(tail_curr(0), tail_lb, tail_ub), Delta_tail)[0], tail_lb, tail_ub);
      loglik_prop = UQ::MLoglik(range_curr, tail_prop, nu_curr, nugget_curr, output, H, d, covmodel);

      if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
        tail_sample(it) = tail_curr(0);
      }else{
        // log prior density of cauchy dist
        log_prior_curr = (tail_a-1.0)*log(tail_curr(0)-tail_lb) + (tail_b-1.0)*log(tail_ub-tail_curr(0));
        log_prior_prop = (tail_a-1.0)*log(tail_prop(0)-tail_lb) + (tail_b-1.0)*log(tail_ub-tail_prop(0));


        // log proposal density 
        log_pr_tran =  0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(tail_curr(0)-tail_lb) + log(tail_ub-tail_curr(0));
        Jacobian_prop = log(tail_prop(0)-tail_lb) + log(tail_ub-tail_prop(0));

        MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                      + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_tail[it]= TRUE;
          tail_curr(0) = tail_prop(0);
          loglik_curr = loglik_prop;
        }

        tail_sample(it) = tail_curr(0);
      }

      // if(nugget_est){

        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        // generate proposal
        nugget_prop = ilogit(Rcpp::rnorm(1, logit(nugget_curr, nugget_lb, nugget_ub), Delta_nugget)[0], nugget_lb, nugget_ub);
        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nugget_sample(it) = nugget_curr;
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
            accept_rate_nugget[it]=TRUE;
            nugget_curr = nugget_prop;
            loglik_curr = loglik_prop;
          }

          nugget_sample(it) = nugget_curr;
        }
      // } 

      // update smoothness parameter
      if(smoothness_est){

        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        // generate proposal
        nu_prop(0) = ilogit(Rcpp::rnorm(1, logit(nu_curr(0), nu_lb, nu_ub), Delta_nu)[0], nu_lb, nu_ub);
        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nu_sample(it) = nu_curr(0);
        }else{
          // log prior density of cauchy dist
          log_prior_curr = (nu_a-1.0)*log(nu_curr(0)-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_curr(0));
          log_prior_prop = (nu_a-1.0)*log(nu_prop(0)-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_prop(0));

          // log proposal density 
          log_pr_tran =  0;
          log_pr_rtran = 0;

          // Jacobian 
          Jacobian_curr = log(nu_curr(0)-nu_lb) + log(nu_ub-nu_curr(0));
          Jacobian_prop = log(nu_prop(0)-nu_lb) + log(nu_ub-nu_prop(0));

          MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                          + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_nu[it]=TRUE;
            nu_curr(0) = nu_prop(0);
            loglik_curr = loglik_prop;
          }

          nu_sample(it) = nu_curr(0);
        }
      }       

      /*************************************************************************************/
      /*****************************Prediction**********************************/

      R = UQ::ARD_kernel(d, range_sample.row(it), tail_sample(it), nu_sample(it), family);
      Rnew = UQ::ARD_kernel(d0, range_sample.row(it), tail_sample(it), nu_sample(it), family);

      R.diagonal().array() += nugget_sample(it);

      ldltR.compute(R);
      RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
      RH = RInv * H;
      HRH = H.transpose() * RH;
      ldltH.compute(HRH);
      Ry = RInv*output;

      bhat = ldltH.solve(H.transpose()*Ry);
      res = output - H*bhat;
      // Rnew = UQ::kernel(d0, par, covmodel);
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
      /*****************************************************************************************/

    } 


  }else if(family=="matern" || family=="powexp" || family=="gauss") {

    // begin MCMC sampling 
    for(int it=0; it<nsample; it++){
      if(Progress::check_abort()){
        return R_NilValue;
      }
      prog.increment();

      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // update range parameter 
      for(int k=0; k<Dim; k++){

        /*************************************************************************************/

        // generate proposal
        range_prop(k) = ilogit(Rcpp::rnorm(1, logit(range_curr(k), range_lb(k), range_ub(k)), Delta_range(k))[0], range_lb(k), range_ub(k));
        loglik_prop = UQ::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // log prior density of cauchy dist
        log_prior_curr = (range_a(k)-1.0)*log(range_curr(k)-range_lb(k)) + (range_b(k)-1.0)*log(range_ub(k)-range_curr(k));
        log_prior_prop = (range_a(k)-1.0)*log(range_prop(k)-range_lb(k)) + (range_b(k)-1.0)*log(range_ub(k)-range_prop(k));

        // log proposal density 
        log_pr_tran = 0;
        log_pr_rtran = 0;

        // Jacobian
        Jacobian_curr = log(range_curr(k)-range_lb(k)) + log(range_ub(k)-range_curr(k));
        Jacobian_prop = log(range_prop(k)-range_lb(k)) + log(range_ub(k)-range_prop(k));

        MH_ratio = (loglik_prop - loglik_curr) + (log_prior_prop-log_prior_curr) 
                       + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_range(it,k) = TRUE;
          range_curr(k) = range_prop(k);
          loglik_curr = loglik_prop;
        }

        range_sample(it,k) = range_curr(k);
      }

      // if(nugget_est){

        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // generate proposal
        nugget_prop = ilogit(Rcpp::rnorm(1, logit(nugget_curr, nugget_lb, nugget_ub), Delta_nugget)[0], nugget_lb, nugget_ub);
        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

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
          accept_rate_nugget[it]=TRUE;
          nugget_curr = nugget_prop;
          loglik_curr = loglik_prop;
        }

        nugget_sample(it) = nugget_curr;
      // }

      // update smoothness parameter
      if(smoothness_est){

        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // generate proposal
        nu_prop(0) = ilogit(Rcpp::rnorm(1, logit(nu_curr(0), nu_lb, nu_ub), Delta_nu)[0], nu_lb, nu_ub);

        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

        // log prior density of cauchy dist
        log_prior_curr = (nu_a-1.0)*log(nu_curr(0)-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_curr(0));
        log_prior_prop = (nu_a-1.0)*log(nu_prop(0)-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_prop(0));

        // log proposal density 
        log_pr_tran =  0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(nu_curr(0)-nu_lb) + log(nu_ub-nu_curr(0));
        Jacobian_prop = log(nu_prop(0)-nu_lb) + log(nu_ub-nu_prop(0));

        MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_nu[it]=TRUE;
          nu_curr(0) = nu_prop(0);
          loglik_curr = loglik_prop;
        }

        nu_sample(it) = nu_curr(0);
      }

      /*************************************************************************************/
      /*****************************Prediction**********************************/

      R = UQ::ARD_kernel(d, range_sample.row(it), tail_sample(it), nu_sample(it), family);
      Rnew = UQ::ARD_kernel(d0, range_sample.row(it), tail_sample(it), nu_sample(it), family);

      R.diagonal().array() += nugget_sample(it);

      ldltR.compute(R);
      RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
      RH = RInv * H;
      HRH = H.transpose() * RH;
      ldltH.compute(HRH);
      Ry = RInv*output;

      bhat = ldltH.solve(H.transpose()*Ry);
      res = output - H*bhat;
      // Rnew = UQ::kernel(d0, par, covmodel);
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
      /*****************************************************************************************/


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
                                Rcpp::_["pred"] = ysim_sample);      
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
                                Rcpp::_["pred"] = ysim_sample);      
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


// MCMC for tensor covariance kernels with RW on unconstrained parameter space
Rcpp::List UQ::tensor_MCMCOBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
  const Eigen::MatrixXd& input, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
  const Eigen::VectorXd& nu, const double& nugget, const Rcpp::List& covmodel, 
  const bool& smoothness_est, const Rcpp::List& proposal, const int& nsample, const bool& verbose,
  const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew){


  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  Rcpp::List d = UQ::adist(input, input);
  int Dim = d.size();


  Eigen::VectorXd Delta_range;
  if(proposal.containsElementNamed("range")){
    Delta_range = Rcpp::as<Eigen::VectorXd>(proposal["range"]);
  }else{
    Delta_range = 0.1*Eigen::VectorXd::Ones(Dim);
  }

  Eigen::VectorXd Delta_tail; // sd in the LN proposal distribution.
  if(proposal.containsElementNamed("tail")){
    Delta_tail = Rcpp::as<Eigen::VectorXd>(proposal["tail"]);
  }else{
    Delta_tail = 0.1*Eigen::VectorXd::Ones(Dim);
  }

  double Delta_nugget;
  if(proposal.containsElementNamed("nugget")){
    Delta_nugget = Rcpp::as<double>(proposal["nugget"]);
  }else{
    Delta_nugget = 0.1;
  }
  
  Eigen::VectorXd Delta_nu;
  if(proposal.containsElementNamed("nu")){
    Delta_nu = Rcpp::as<Eigen::VectorXd>(proposal["nu"]);
  }else{
    Delta_nu = 0.1*Eigen::VectorXd::Ones(Dim);
  }

  double loglik_curr = UQ::MLoglik(range, tail, nu, nugget, output, H, d, covmodel);
  double loglik_prop=0.0;
  
  // initial values for parameters 
  Eigen::VectorXd range_curr = range;
  Eigen::VectorXd tail_curr = tail;
  Eigen::VectorXd nu_curr = nu;
  double nugget_curr = nugget;


  Eigen::MatrixXd range_sample(nsample, Dim);
  Eigen::MatrixXd tail_sample(nsample, Dim);
  Eigen::MatrixXd nu_sample(nsample, Dim);
  Eigen::VectorXd nugget_sample(nsample);

  Eigen::VectorXd range_prop(Dim), tail_prop(Dim), nu_prop(Dim);
  double nugget_prop;
  range_prop = range_curr;
  tail_prop = tail_curr;
  nu_prop = nu_curr;
  nugget_prop = nugget_curr;

  // MCMC stuff
  double log_prior_curr, log_prior_prop, log_pr_tran, log_pr_rtran, MH_ratio;
  double unif_rnd;
  Rcpp::LogicalMatrix accept_rate_range(nsample, Dim), accept_rate_tail(nsample, Dim);
  Rcpp::LogicalVector accept_rate_nu(nsample), accept_rate_nugget(nsample);
  double Jacobian_curr, Jacobian_prop;

  /************************* Prediction ****************************************/
  int n = output.rows();
  int q = output.cols();
  int m = input_new.rows();
  int p = H.cols();
  int dim = input.cols();
  Eigen::MatrixXd R(n,n), RInv(n,n), RH(n,p), HRH(p,p), Rnew(n,m), HRHInv(p,p);
  Eigen::MatrixXd res(n,q), Ry(n,q), bhat(p,q);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH, ldlt;

  double df = n-p;
  Rcpp::List d0(dim);

  Eigen::MatrixXd sig2hat(q,q), predmean(m, q), L(q,q);

  Eigen::VectorXd Rtmp(n), tmp(p), pred_corr(m);

  Eigen::MatrixXd ysim(m,q);
  Rcpp::List ysim_sample(nsample); 

  //d = UQ::adist(input, input);
  d0 = UQ::adist(input, input_new);

  // U(nu_lb, nu_ub) on nu
  double nu_lb = 0, nu_ub = 6;
  if(family=="cauchy" || family=="powexp"){
    nu_ub = 2.0;
  }
  // create an empty list for abortion 
  // Rcpp::List abort_value = Rcpp::List::create(Rcpp::_["abort"]=R_NilValue);
  int k=0;

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

      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // update range parameter 
      for(int k=0; k<Dim; k++){

        // generate proposal
        range_prop(k) = exp(Rcpp::rnorm(1, log(range_curr(k)), Delta_range(k))[0]);
        loglik_prop = UQ::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          range_sample(it,k) = range_curr(k);
        }else{
          // log prior density of cauchy dist
          log_prior_curr = -log(1.0+range_curr(k)*range_curr(k));
          log_prior_prop = -log(1.0+range_prop(k)*range_prop(k));

          // log proposal density 
          log_pr_tran =  0;
          log_pr_rtran = 0;

          // Jacobian
          Jacobian_curr = log(range_curr(k));
          Jacobian_prop = log(range_prop(k));

          MH_ratio = (loglik_prop - loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_range(it,k) = TRUE;
            range_curr(k) = range_prop(k);
            loglik_curr = loglik_prop;
          }
          range_sample(it,k) = range_curr(k);
        }

          
      }

      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // update tail decay parameter
      for(int k=0; k<Dim; k++){

        // generate proposal
        tail_prop(k) = exp(Rcpp::rnorm(1, log(tail_curr(k)), Delta_tail(k))[0]);
        loglik_prop = UQ::MLoglik(range_curr, tail_prop, nu_curr, nugget_curr, output, H, d, covmodel);

        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          tail_sample(it,k) = tail_curr(k);
        }else{
          // log prior density of cauchy dist
          log_prior_curr = -log(1.0+tail_curr(k)*tail_curr(k));
          log_prior_prop = -log(1.0+tail_prop(k)*tail_prop(k));

          // log proposal density 
          log_pr_tran = 0;
          log_pr_rtran = 0;

          // Jacobian 
          Jacobian_curr = log(tail_curr(k));
          Jacobian_prop = log(tail_prop(k));

          MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_tail(it,k) = TRUE;
            tail_curr(k) = tail_prop(k);
            loglik_curr = loglik_prop;
          }

          tail_sample(it,k) = tail_curr(k);
        }
      }

      //if(nugget_est){
        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // generate proposal
        nugget_prop = exp(Rcpp::rnorm(1, log(nugget_curr), Delta_nugget)[0]);
        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nugget_sample(it) = nugget_curr;
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
            accept_rate_nugget[it]=TRUE;
            nugget_curr = nugget_prop;
            loglik_curr = loglik_prop;
          }
          nugget_sample(it) = nugget_curr;
        }
      //}    

      // update smoothness parameter
      if(smoothness_est){
        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        k = 0;
        

          // generate proposal
          // nu_prop(k) = exp(Rcpp::rnorm(1, log(nu_curr(k)), Delta_nu(k))[0]);
          nu_prop(k) = ilogit(Rcpp::rnorm(1, logit(nu_curr(k), nu_lb, nu_ub), Delta_nu(k))[0], nu_lb, nu_ub);   

          loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);
          if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
            nu_sample(it,k) = nu_curr(k);
          }else{
            // log prior density of cauchy dist
            // log_prior_curr = -log(1.0+nu_curr(k)*nu_curr(k));
            // log_prior_prop = -log(1.0+nu_prop(k)*nu_prop(k));
            log_prior_curr = 0;
            log_prior_prop = 0;

            // log proposal density 
            log_pr_tran = 0;
            log_pr_rtran = 0;

            // Jacobian 
            // Jacobian_curr = log(nu_curr(k));
            // Jacobian_prop = log(nu_prop(k));
            Jacobian_curr = log(nu_curr(k)-nu_lb) + log(nu_ub-nu_curr(k));
            Jacobian_prop = log(nu_prop(k)-nu_lb) + log(nu_ub-nu_prop(k));

            MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                          + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

            unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
            if(log(unif_rnd)<MH_ratio){ // accept 
              accept_rate_nu[it] = TRUE;
              nu_curr(k) = nu_prop(k);
              loglik_curr = loglik_prop;
            }

            nu_sample(it,k) = nu_curr(k);
          }

          for(int j=1; j<Dim; j++){
            nu_curr(j) = nu_curr(k);
          }

        
      } // end smooth

      /*************************************************************************************/
      /*****************************Prediction**********************************/

      R = UQ::tensor_kernel(d, range_sample.row(it), tail_sample.row(it), nu_curr, family);
      Rnew = UQ::tensor_kernel(d0, range_sample.row(it), tail_sample.row(it), nu_curr, family);

      R.diagonal().array() += nugget_sample(it);

      ldltR.compute(R);
      RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
      RH = RInv * H;
      HRH = H.transpose() * RH;
      ldltH.compute(HRH);
      Ry = RInv*output;

      bhat = ldltH.solve(H.transpose()*Ry);
      res = output - H*bhat;
      // Rnew = UQ::kernel(d0, par, covmodel);
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
      /*****************************************************************************************/
            
    } 


  }else if(family=="matern" || family=="powexp" || family=="gauss") {

    // begin MCMC sampling 
    for(int it=0; it<nsample; it++){
      if(Progress::check_abort()){
        return R_NilValue;
      }
      prog.increment();

      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // update range parameter 
      for(int k=0; k<Dim; k++){

        // generate proposal
        range_prop(k) = exp(Rcpp::rnorm(1, log(range_curr(k)), Delta_range(k))[0]);
        loglik_prop = UQ::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // log prior density of cauchy dist
        log_prior_curr = -log(1.0+range_curr(k)*range_curr(k));
        log_prior_prop = -log(1.0+range_prop(k)*range_prop(k));

        // log proposal density 
        log_pr_tran =  0;
        log_pr_rtran = 0;

        // Jacobian
        Jacobian_curr = log(range_curr(k));
        Jacobian_prop = log(range_prop(k));

        MH_ratio = (loglik_prop - loglik_curr) + (log_prior_prop-log_prior_curr) 
                      + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_range(it,k) = TRUE;
          range_curr(k) = range_prop(k);
          loglik_curr = loglik_prop;
        }

        range_sample(it,k) = range_curr(k);
      }

      // if(nugget_est){
        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        // generate proposal
        nugget_prop = exp(Rcpp::rnorm(1, log(nugget_curr), Delta_nugget)[0]);
        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

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
          accept_rate_nugget[it]=TRUE;
          nugget_curr = nugget_prop;
          loglik_curr = loglik_prop;
        }
        nugget_sample(it) = nugget_curr;
      // }

      // update smoothness parameter
      if(smoothness_est){

        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        // update smoothness parameter
        k = 0; // common smoothness parameter across each dimension
        // for(int k=0; k<Dim; k++){

          // generate proposal
          // nu_prop(k) = exp(Rcpp::rnorm(1, log(nu_curr(k)), Delta_nu(k))[0]);
          nu_prop(k) = ilogit(Rcpp::rnorm(1, logit(nu_curr(k), nu_lb, nu_ub), Delta_nu(k))[0], nu_lb, nu_ub);   

          loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

          // log prior density of cauchy dist
          // log_prior_curr = -log(1.0+nu_curr(k)*nu_curr(k));
          // log_prior_prop = -log(1.0+nu_prop(k)*nu_prop(k));
          log_prior_curr = 0;
          log_prior_prop = 0;

          // log proposal density 
          log_pr_tran = 0;
          log_pr_rtran = 0;

          // Jacobian 
          // Jacobian_curr = log(nu_curr(k));
          // Jacobian_prop = log(nu_prop(k));
          Jacobian_curr = log(nu_curr(k)-nu_lb) + log(nu_ub-nu_curr(k));
          Jacobian_prop = log(nu_prop(k)-nu_lb) + log(nu_ub-nu_prop(k));

          MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_nu(it) = TRUE;
            nu_curr(k) = nu_prop(k);
            loglik_curr = loglik_prop;
          }

          nu_sample(it,k) = nu_curr(k);
          for(int j=1; j<Dim; j++){
            nu_curr(j) = nu_curr(k);
          }
        // }
      }

      /*************************************************************************************/
      /*****************************Prediction**********************************/

      R = UQ::tensor_kernel(d, range_sample.row(it), tail_sample.row(it), nu_curr, family);
      Rnew = UQ::tensor_kernel(d0, range_sample.row(it), tail_sample.row(it), nu_curr, family);

      R.diagonal().array() += nugget_sample(it);

      ldltR.compute(R);
      RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
      RH = RInv * H;
      HRH = H.transpose() * RH;
      ldltH.compute(HRH);
      Ry = RInv*output;

      bhat = ldltH.solve(H.transpose()*Ry);
      res = output - H*bhat;
      // Rnew = UQ::kernel(d0, par, covmodel);
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
      /*****************************************************************************************/
      

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
                                Rcpp::_["nu"]=nu_sample.col(0),
                                Rcpp::_["accept_rate_range"]=accept_rate_range,
                                Rcpp::_["accept_rate_tail"]=accept_rate_tail,
                                Rcpp::_["accept_rate_nugget"]=accept_rate_nugget,
                                Rcpp::_["accept_rate_nu"]=accept_rate_nu,
                                Rcpp::_["pred"]=ysim_sample);      
    }else{
      return Rcpp::List::create(Rcpp::_["range"]=range_sample, 
                                Rcpp::_["tail"]=tail_sample,
                                Rcpp::_["nugget"]=nugget_sample,
                                Rcpp::_["accept_rate_range"]=accept_rate_range,
                                Rcpp::_["accept_rate_tail"]=accept_rate_tail,
                                Rcpp::_["accept_rate_nugget"]=accept_rate_nugget,
                                Rcpp::_["pred"]=ysim_sample);
    }

  }else if(family=="matern" || family=="powexp" || family=="gauss"){

    if(smoothness_est){
      return Rcpp::List::create(Rcpp::_["range"]=range_sample, 
                                Rcpp::_["nugget"]=nugget_sample,
                                Rcpp::_["nu"]=nu_sample.col(0),
                                Rcpp::_["accept_rate_range"]=accept_rate_range,
                                Rcpp::_["accept_rate_nugget"]=accept_rate_nugget,
                                Rcpp::_["accept_rate_nu"]=accept_rate_nu,
                                Rcpp::_["pred"]=ysim_sample);      
    }else{
      return Rcpp::List::create(Rcpp::_["range"]=range_sample, 
                                Rcpp::_["nugget"]=nugget_sample,
                                Rcpp::_["accept_rate_range"]=accept_rate_range,
                                Rcpp::_["accept_rate_nugget"]=accept_rate_nugget,
                                Rcpp::_["pred"]=ysim_sample);
    }
  }else{
    Rcpp::stop("The family of covariance functions is not implemented yet.\n");
  }

}


// MCMC for tensor covariance kernels with RW on unconstrained parameter space
Rcpp::List UQ::tensor_MCMCSBayes(const Eigen::MatrixXd& output, const Eigen::MatrixXd& H, 
  const Eigen::MatrixXd& input, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, 
  const Eigen::VectorXd& nu, const double& nugget, const Rcpp::List& covmodel, 
  const bool& smoothness_est, const Rcpp::List& prior, const Rcpp::List& proposal, const int& nsample, const bool& verbose,
  const Eigen::MatrixXd& input_new, const Eigen::MatrixXd& Hnew){


  std::string family = Rcpp::as<std::string>(covmodel["family"]);
  std::string form = Rcpp::as<std::string>(covmodel["form"]);

  Rcpp::List d = UQ::adist(input, input);
  int Dim = d.size();


  Eigen::VectorXd Delta_range;
  if(proposal.containsElementNamed("range")){
    Delta_range = Rcpp::as<Eigen::VectorXd>(proposal["range"]);
  }else{
    Delta_range = 0.1*Eigen::VectorXd::Ones(Dim);
  }

  Eigen::VectorXd Delta_tail; // sd in the LN proposal distribution.
  if(proposal.containsElementNamed("tail")){
    Delta_tail = Rcpp::as<Eigen::VectorXd>(proposal["tail"]);
  }else{
    Delta_tail = 0.1*Eigen::VectorXd::Ones(Dim);
  }

  double Delta_nugget;
  if(proposal.containsElementNamed("nugget")){
    Delta_nugget = Rcpp::as<double>(proposal["nugget"]);
  }else{
    Delta_nugget = 0.1;
  }
  
  Eigen::VectorXd Delta_nu;
  if(proposal.containsElementNamed("nu")){
    Delta_nu = Rcpp::as<Eigen::VectorXd>(proposal["nu"]);
  }else{
    Delta_nu = 0.1*Eigen::VectorXd::Ones(Dim);
  }

  // tuning parameters in priors
  int n_a, n_b, n_lb, n_ub;
  Rcpp::List range_prior; //beta(a, b, lb, ub) 
  Eigen::VectorXd range_a=Eigen::VectorXd::Ones(Dim), range_b=Eigen::VectorXd::Ones(Dim);
  Eigen::VectorXd range_lb=Eigen::VectorXd::Zero(Dim);
  Eigen::VectorXd range_ub(Dim);
  for(int k=0; k<Dim; k++){
    range_ub(k) = 3.0*(Rcpp::as<Eigen::MatrixXd>(d[k])).maxCoeff();
  } 
  if(prior.containsElementNamed("range")){
    range_prior = Rcpp::as<Rcpp::List>(prior["range"]);
    if(range_prior.containsElementNamed("a")){
      n_a = (Rcpp::as<Eigen::VectorXd>(range_prior["a"])).size();
      if(n_a==Dim){
        range_a = Rcpp::as<Eigen::VectorXd>(range_prior["a"]);
      }
      if(n_a==1){
        range_a = Rcpp::as<double>(range_prior["a"])*Eigen::VectorXd::Ones(Dim);
      }       
    }

    if(range_prior.containsElementNamed("b")){
      n_b = (Rcpp::as<Eigen::VectorXd>(range_prior["b"])).size();
      if(n_b==Dim){
        range_b = Rcpp::as<Eigen::VectorXd>(range_prior["b"]);
      }
      if(n_b==1){
        range_b = Rcpp::as<double>(range_prior["b"])*Eigen::VectorXd::Ones(Dim);
      } 
    }

    if(range_prior.containsElementNamed("lb")){
      n_lb = (Rcpp::as<Eigen::VectorXd>(range_prior["lb"])).size();
      if(n_lb==Dim){
        range_lb = Rcpp::as<Eigen::VectorXd>(range_prior["lb"]);
      }
      if(n_lb==1){
        range_lb = Rcpp::as<double>(range_prior["lb"])*Eigen::VectorXd::Ones(Dim);
      }
    }

    if(range_prior.containsElementNamed("ub")){
      n_ub = (Rcpp::as<Eigen::VectorXd>(range_prior["ub"])).size();
      if(n_ub==Dim){
        range_ub = Rcpp::as<Eigen::VectorXd>(range_prior["ub"]);
      }
      if(n_ub==1){
        range_ub = Rcpp::as<double>(range_prior["ub"])*Eigen::VectorXd::Ones(Dim);
      }
    }    
  }

  Rcpp::List tail_prior; //beta(a, b, lb, ub) 
  Eigen::VectorXd tail_a=Eigen::VectorXd::Ones(Dim), tail_b=Eigen::VectorXd::Ones(Dim);
  Eigen::VectorXd tail_lb= Eigen::VectorXd::Zero(Dim);
  Eigen::VectorXd tail_ub = 5.0*Eigen::VectorXd::Ones(Dim); 
  if(prior.containsElementNamed("tail")){
    tail_prior = Rcpp::as<Rcpp::List>(prior["tail"]);
    if(tail_prior.containsElementNamed("a")){
      n_a = (Rcpp::as<Eigen::VectorXd>(tail_prior["a"])).size();
      if(n_a==Dim){
        tail_a = Rcpp::as<Eigen::VectorXd>(tail_prior["a"]);
      }
      if(n_a==1){
        tail_a = Rcpp::as<double>(tail_prior["a"])*Eigen::VectorXd::Ones(Dim);
      }
    }
    if(tail_prior.containsElementNamed("b")){
      n_b = (Rcpp::as<Eigen::VectorXd>(tail_prior["b"])).size();
      if(n_b==Dim){
        tail_a = Rcpp::as<Eigen::VectorXd>(tail_prior["b"]);
      }
      if(n_b==1){
        tail_a = Rcpp::as<double>(tail_prior["b"])*Eigen::VectorXd::Ones(Dim);
      }
    }
    if(tail_prior.containsElementNamed("lb")){
      n_lb = (Rcpp::as<Eigen::VectorXd>(tail_prior["lb"])).size();
      if(n_lb==Dim){
        tail_a = Rcpp::as<Eigen::VectorXd>(tail_prior["lb"]);
      }
      if(n_lb==1){
        tail_a = Rcpp::as<double>(tail_prior["lb"])*Eigen::VectorXd::Ones(Dim);
      }
    }
    if(tail_prior.containsElementNamed("ub")){
      n_ub = (Rcpp::as<Eigen::VectorXd>(tail_prior["ub"])).size();
      if(n_ub==Dim){
        tail_ub = Rcpp::as<Eigen::VectorXd>(tail_prior["ub"]);
      }
      if(n_ub==1){
        tail_ub = Rcpp::as<double>(tail_prior["ub"])*Eigen::VectorXd::Ones(Dim);
      }
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
  double nu_lb=0.1, nu_ub=6.0;
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

  double loglik_curr = UQ::MLoglik(range, tail, nu, nugget, output, H, d, covmodel);
  double loglik_prop=0.0;
  
  // initial values for parameters 
  Eigen::VectorXd range_curr = range;
  Eigen::VectorXd tail_curr = tail;
  Eigen::VectorXd nu_curr = nu;
  double nugget_curr = nugget;


  Eigen::MatrixXd range_sample(nsample, Dim);
  Eigen::MatrixXd tail_sample(nsample, Dim);
  Eigen::MatrixXd nu_sample(nsample, Dim);
  Eigen::VectorXd nugget_sample(nsample);

  Eigen::VectorXd range_prop(Dim), tail_prop(Dim), nu_prop(Dim);
  double nugget_prop;
  range_prop = range_curr;
  tail_prop = tail_curr;
  nu_prop = nu_curr;
  nugget_prop = nugget_curr;

  // MCMC stuff
  double log_prior_curr, log_prior_prop, log_pr_tran, log_pr_rtran, MH_ratio;
  double unif_rnd;
  Rcpp::LogicalMatrix accept_rate_range(nsample,Dim), accept_rate_tail(nsample,Dim);
  Rcpp::LogicalVector accept_rate_nu(nsample), accept_rate_nugget(nsample);
  double Jacobian_curr, Jacobian_prop;

  /************************* Prediction ****************************************/
  int n = output.rows();
  int q = output.cols();
  int m = input_new.rows();
  int p = H.cols();
  int dim = input.cols();
  Eigen::MatrixXd R(n,n), RInv(n,n), RH(n,p), HRH(p,p), Rnew(n,m), HRHInv(p,p);
  Eigen::MatrixXd res(n,q), Ry(n,q), bhat(p,q);
  Eigen::LDLT<Eigen::MatrixXd> ldltR, ldltH, ldlt;

  double df = n-p;
  Rcpp::List d0(dim);

  Eigen::MatrixXd sig2hat(q,q), predmean(m, q), L(q,q);

  Eigen::VectorXd Rtmp(n), tmp(p), pred_corr(m);

  Eigen::MatrixXd ysim(m,q);
  Rcpp::List ysim_sample(nsample); 

  //d = UQ::adist(input, input);
  d0 = UQ::adist(input, input_new);

  // create an empty list for abortion 
  // Rcpp::List abort_value = Rcpp::List::create(Rcpp::_["abort"]=R_NilValue);
  int k=0;

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

      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // update range parameter 
      for(int k=0; k<Dim; k++){

        // generate proposal
        range_prop(k) = ilogit(Rcpp::rnorm(1, logit(range_curr(k), range_lb(k), range_ub(k)), Delta_range(k))[0], range_lb(k), range_ub(k));
        loglik_prop = UQ::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          range_sample(it,k) = range_curr(k);
        }else{
          // log prior density of cauchy dist
          log_prior_curr = (range_a(k)-1.0)*log(range_curr(k)-range_lb(k)) + (range_b(k)-1.0)*log(range_ub(k)-range_curr(k));
          log_prior_prop = (range_a(k)-1.0)*log(range_prop(k)-range_lb(k)) + (range_b(k)-1.0)*log(range_ub(k)-range_prop(k));

          // log proposal density 
          log_pr_tran =  0;
          log_pr_rtran = 0;

          // Jacobian 
          Jacobian_curr = log(range_curr(k)-range_lb(k)) + log(range_ub(k)-range_curr(k));
          Jacobian_prop = log(range_prop(k)-range_lb(k)) + log(range_ub(k)-range_prop(k));

          MH_ratio = (loglik_prop - loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_range(it,k) = TRUE;
            range_curr(k) = range_prop(k);
            loglik_curr = loglik_prop;
          }

          range_sample(it,k) = range_curr(k);
        }
      }

      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // update tail decay parameter
      for(int k=0; k<Dim; k++){

        // generate proposal
        tail_prop(k) = ilogit(Rcpp::rnorm(1, logit(tail_curr(k), tail_lb(k), tail_ub(k)), Delta_tail(k))[0], tail_lb(k), tail_ub(k));
        loglik_prop = UQ::MLoglik(range_curr, tail_prop, nu_curr, nugget_curr, output, H, d, covmodel);
        
        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          tail_sample(it,k) = tail_curr(k);
        }else{
          // log prior density
          log_prior_curr = (tail_a(k)-1.0)*log(tail_curr(k)-tail_lb(k)) + (tail_b(k)-1.0)*log(tail_ub(k)-tail_curr(k));
          log_prior_prop = (tail_a(k)-1.0)*log(tail_prop(k)-tail_lb(k)) + (tail_b(k)-1.0)*log(tail_ub(k)-tail_prop(k));

          // log proposal density 
          log_pr_tran = 0;
          log_pr_rtran = 0;

          // Jacobian 
          Jacobian_curr = log(tail_curr(k)-tail_lb(k)) + log(tail_ub(k)-tail_curr(k));
          Jacobian_prop = log(tail_prop(k)-tail_lb(k)) + log(tail_ub(k)-tail_prop(k));

          MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_tail(it,k) = TRUE;
            tail_curr(k) = tail_prop(k);
            loglik_curr = loglik_prop;
          }

          tail_sample(it,k) = tail_curr(k);
        }
      }

      //if(nugget_est){
        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // generate proposal
        nugget_prop = ilogit(Rcpp::rnorm(1, logit(nugget_curr, nugget_lb, nugget_ub), Delta_nugget)[0], nugget_lb, nugget_ub);
        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

        if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
          nugget_sample(it) = nugget_curr;
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
            accept_rate_nugget[it]=TRUE;
            nugget_curr = nugget_prop;
            loglik_curr = loglik_prop;
          }
          nugget_sample(it) = nugget_curr;
        }
      //}    

      // update smoothness parameter
      if(smoothness_est){
        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        k = 0;
        // for(int k=0; k<Dim; k++){

          // generate proposal
          nu_prop(k) = ilogit(Rcpp::rnorm(1, logit(nu_curr(k), nu_lb, nu_ub), Delta_nu(k))[0], nu_lb, nu_ub);
          loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

          if(Rcpp::traits::is_nan<REALSXP>(loglik_prop) || Rcpp::traits::is_infinite<REALSXP>(loglik_prop)){
            nu_sample(it,k) = nu_curr(k);
          }else{
            // log prior density of cauchy dist
            log_prior_curr = (nu_a-1.0)*log(nu_curr(k)-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_curr(k));
            log_prior_prop = (nu_a-1.0)*log(nu_prop(k)-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_prop(k));

            // log proposal density 
            log_pr_tran = 0;
            log_pr_rtran = 0;

            // Jacobian 
            Jacobian_curr = log(nu_curr(k)-nu_lb) + log(nu_ub-nu_curr(k));
            Jacobian_prop = log(nu_prop(k)-nu_lb) + log(nu_ub-nu_prop(k));

            MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                          + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

            unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
            if(log(unif_rnd)<MH_ratio){ // accept 
              accept_rate_nu(it) = TRUE;
              nu_curr(k) = nu_prop(k);
              loglik_curr = loglik_prop;
            }
            nu_sample(it,k) = nu_curr(k);
          }

          for(int j=1; j<Dim; j++){
            nu_curr(j) = nu_curr(k);
          }
        // }
      }

      /*************************************************************************************/
      /*****************************Prediction**********************************/

      R = UQ::tensor_kernel(d, range_sample.row(it), tail_sample.row(it), nu_curr, family);
      Rnew = UQ::tensor_kernel(d0, range_sample.row(it), tail_sample.row(it), nu_curr, family);

      R.diagonal().array() += nugget_sample(it);

      ldltR.compute(R);
      RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
      RH = RInv * H;
      HRH = H.transpose() * RH;
      ldltH.compute(HRH);
      Ry = RInv*output;

      bhat = ldltH.solve(H.transpose()*Ry);
      res = output - H*bhat;
      // Rnew = UQ::kernel(d0, par, covmodel);
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
      /*****************************************************************************************/
    } 


  }else if(family=="matern" || family=="powexp" || family=="gauss") {

    // begin MCMC sampling 
    for(int it=0; it<nsample; it++){
      if(Progress::check_abort()){
        return R_NilValue;
      }
      prog.increment();

      // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

      // update range parameter 
      for(int k=0; k<Dim; k++){

        // generate proposal
        range_prop(k) = ilogit(Rcpp::rnorm(1, logit(range_curr(k), range_lb(k), range_ub(k)), Delta_range(k))[0], range_lb(k), range_ub(k));
        loglik_prop = UQ::MLoglik(range_prop, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);

        // log prior density of cauchy dist
        log_prior_curr = (range_a(k)-1.0)*log(range_curr(k)-range_lb(k)) + (range_b(k)-1.0)*log(range_ub(k)-range_curr(k));
        log_prior_prop = (range_a(k)-1.0)*log(range_prop(k)-range_lb(k)) + (range_b(k)-1.0)*log(range_ub(k)-range_prop(k));

        // log proposal density 
        log_pr_tran =  0;
        log_pr_rtran = 0;

        // Jacobian 
        Jacobian_curr = log(range_curr(k)-range_lb(k)) + log(range_ub(k)-range_curr(k));
        Jacobian_prop = log(range_prop(k)-range_lb(k)) + log(range_ub(k)-range_prop(k));

        MH_ratio = (loglik_prop - loglik_curr) + (log_prior_prop-log_prior_curr) 
                      + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

        unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
        if(log(unif_rnd)<MH_ratio){ // accept 
          accept_rate_range(it,k) = TRUE;
          range_curr(k) = range_prop(k);
          loglik_curr = loglik_prop;
        }

        range_sample(it,k) = range_curr(k);
      }

      // if(nugget_est){
        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        // generate proposal
        nugget_prop = ilogit(Rcpp::rnorm(1, logit(nugget_curr, nugget_lb, nugget_ub), Delta_nugget)[0], nugget_lb, nugget_ub);
        loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_prop, output, H, d, covmodel);

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
          accept_rate_nugget[it]=TRUE;
          nugget_curr = nugget_prop;
          loglik_curr = loglik_prop;
        }
        nugget_sample(it) = nugget_curr;
      // }

      // update smoothness parameter
      if(smoothness_est){

        // loglik_curr = UQ::MLoglik(range_curr, tail_curr, nu_curr, nugget_curr, output, H, d, covmodel);
        // update smoothness parameter
        k = 0; // common smoothness parameter across each dimension
        // for(int k=0; k<Dim; k++){

          // generate proposal
          nu_prop(k) = ilogit(Rcpp::rnorm(1, logit(nu_curr(k), nu_lb, nu_ub), Delta_nu(k))[0], nu_lb, nu_ub);
          loglik_prop = UQ::MLoglik(range_curr, tail_curr, nu_prop, nugget_curr, output, H, d, covmodel);

          // log prior density of cauchy dist
          log_prior_curr = (nu_a-1.0)*log(nu_curr(k)-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_curr(k));
          log_prior_prop = (nu_a-1.0)*log(nu_prop(k)-nu_lb) + (nu_b-1.0)*log(nu_ub-nu_prop(k));

          // log proposal density 
          log_pr_tran = 0;
          log_pr_rtran = 0;

          // Jacobian 
          Jacobian_curr = log(nu_curr(k)-nu_lb) + log(nu_ub-nu_curr(k));
          Jacobian_prop = log(nu_prop(k)-nu_lb) + log(nu_ub-nu_prop(k));

          MH_ratio = (loglik_prop-loglik_curr) + (log_prior_prop-log_prior_curr) 
                        + (log_pr_rtran-log_pr_tran) + (Jacobian_prop-Jacobian_curr);

          unif_rnd = Rcpp::runif(1, 0.0, 1.0)[0];
          if(log(unif_rnd)<MH_ratio){ // accept 
            accept_rate_nu[it] = TRUE;
            nu_curr(k) = nu_prop(k);
            loglik_curr = loglik_prop;
          }

          nu_sample(it,k) = nu_curr(k);
          for(int j=1; j<Dim; j++){
            nu_curr(j) = nu_curr(k);
          }
        // }
      }

      /*************************************************************************************/
      /*****************************Prediction**********************************/

      R = UQ::tensor_kernel(d, range_sample.row(it), tail_sample.row(it), nu_curr, family);
      Rnew = UQ::tensor_kernel(d0, range_sample.row(it), tail_sample.row(it), nu_curr, family);

      R.diagonal().array() += nugget_sample(it);

      ldltR.compute(R);
      RInv = ldltR.solve(Eigen::MatrixXd::Identity(n,n));
      RH = RInv * H;
      HRH = H.transpose() * RH;
      ldltH.compute(HRH);
      Ry = RInv*output;

      bhat = ldltH.solve(H.transpose()*Ry);
      res = output - H*bhat;
      // Rnew = UQ::kernel(d0, par, covmodel);
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
      /*****************************************************************************************/

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
                                Rcpp::_["nu"]=nu_sample.col(0),
                                Rcpp::_["accept_rate_range"]=accept_rate_range,
                                Rcpp::_["accept_rate_tail"]=accept_rate_tail,
                                Rcpp::_["accept_rate_nugget"]=accept_rate_nugget,
                                Rcpp::_["accept_rate_nu"]=accept_rate_nu,
                                Rcpp::_["pred"]=ysim_sample);      
    }else{
      return Rcpp::List::create(Rcpp::_["range"]=range_sample, 
                                Rcpp::_["tail"]=tail_sample,
                                Rcpp::_["nugget"]=nugget_sample,
                                Rcpp::_["accept_rate_range"]=accept_rate_range,
                                Rcpp::_["accept_rate_tail"]=accept_rate_tail,
                                Rcpp::_["accept_rate_nugget"]=accept_rate_nugget,
                                Rcpp::_["pred"]=ysim_sample);
    }

  }else if(family=="matern" || family=="powexp" || family=="gauss"){

    if(smoothness_est){
      return Rcpp::List::create(Rcpp::_["range"]=range_sample, 
                                Rcpp::_["nugget"]=nugget_sample,
                                Rcpp::_["nu"]=nu_sample.col(0),
                                Rcpp::_["accept_rate_range"]=accept_rate_range,
                                Rcpp::_["accept_rate_nugget"]=accept_rate_nugget,
                                Rcpp::_["accept_rate_nu"]=accept_rate_nu,
                                Rcpp::_["pred"]=ysim_sample);      
    }else{
      return Rcpp::List::create(Rcpp::_["range"]=range_sample, 
                                Rcpp::_["nugget"]=nugget_sample,
                                Rcpp::_["accept_rate_range"]=accept_rate_range,
                                Rcpp::_["accept_rate_nugget"]=accept_rate_nugget,
                                Rcpp::_["pred"]=ysim_sample);
    }
  }else{
    Rcpp::stop("The family of covariance functions is not implemented yet.\n");
  }

}


/*************** Model Comparison ******************/
Rcpp::List UQ::tensor_model_evaluation(const Eigen::MatrixXd& output, 
  const Eigen::MatrixXd& input, const Eigen::MatrixXd& H, 
  const Eigen::MatrixXd& range, const Eigen::MatrixXd& tail, 
  const Eigen::MatrixXd& nu, const Eigen::VectorXd& nugget,
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

  Rcpp::List d = UQ::adist(input, input);
  Rcpp::List dnew = UQ::adist(input_new, input_new);
  Rcpp::List dcross = UQ::adist(input, input_new);
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

    R = UQ::tensor_kernel(d, range.row(it), tail.row(it), nu.row(it), family);
    R.diagonal().array() += nugget(it);
    lltR.compute(R);
    L = lltR.matrixL();
    Ly = L.triangularView<Eigen::Lower>().solve(output);
    LH = L.triangularView<Eigen::Lower>().solve(H);
    HRy = LH.transpose()*Ly;
    HRH = LH.transpose()*LH;


    Rcross = UQ::tensor_kernel(dcross, range.row(it), tail.row(it), nu.row(it), family);

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
      Rnew = UQ::tensor_kernel(dnew, range.row(it), tail.row(it), nu.row(it), family);
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
  Eigen::VectorXd range_mean = range.colwise().mean();
  Eigen::VectorXd tail_mean = tail.colwise().mean();
  Eigen::VectorXd nu_mean = nu.colwise().mean();
  double nugget_mean = nugget.mean();

  R = UQ::tensor_kernel(d, range_mean, tail_mean, nu_mean, family);
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


  return Rcpp::List::create(Rcpp::_["pD"]=p_D,
                            Rcpp::_["DIC"]=DIC,
                            Rcpp::_["lppd"]=lpd_ptw.mean(),
                            Rcpp::_["ljpd"]=lpd_joint.mean());
}


Rcpp::List UQ::ARD_model_evaluation(const Eigen::MatrixXd& output, 
  const Eigen::MatrixXd& input, const Eigen::MatrixXd& H, 
  const Eigen::MatrixXd& range, const Eigen::VectorXd& tail, 
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

  Rcpp::List d = UQ::adist(input, input);
  Rcpp::List dnew = UQ::adist(input_new, input_new);
  Rcpp::List dcross = UQ::adist(input, input_new);
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

    R = UQ::ARD_kernel(d, range.row(it), tail(it), nu(it), family);
    R.diagonal().array() += nugget(it);
    lltR.compute(R);
    L = lltR.matrixL();
    Ly = L.triangularView<Eigen::Lower>().solve(output);
    LH = L.triangularView<Eigen::Lower>().solve(H);
    HRy = LH.transpose()*Ly;
    HRH = LH.transpose()*LH;


    Rcross = UQ::ARD_kernel(dcross, range.row(it), tail(it), nu(it), family);

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
      Rnew = UQ::ARD_kernel(dnew, range.row(it), tail(it), nu(it), family);
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
  Eigen::VectorXd range_mean = range.colwise().mean();
  double tail_mean = tail.mean();
  double nu_mean = nu.mean();
  double nugget_mean = nugget.mean();

  R = UQ::ARD_kernel(d, range_mean, tail_mean, nu_mean, family);
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


  return Rcpp::List::create(Rcpp::_["pD"]=p_D,
                            Rcpp::_["DIC"]=DIC,
                            Rcpp::_["lppd"]=lpd_ptw.mean(),
                            Rcpp::_["ljpd"]=lpd_joint.mean());
}


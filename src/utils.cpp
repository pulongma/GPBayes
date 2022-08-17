

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


/****************************************************************************************/
/* Distance Matrices  */
/****************************************************************************************/

double gcdistance(const double& lonA, const double& latA, const double& lonB, const double& latB)
{
  
  // Given the latitude and longitude of two places, this function
  // computes great circle distance between them by using haversine
  // formula, which is well-conditioned for small distance
  
  double dist_lat = 0;
  double dist_lon = 0;
  double dTemp = 0;
  double temp_lon = 0;
  double dist = 0;
  double earthRadius = 6371.0087714; //WGS84 mean radius in km 
  double dTemp1 = 0;
  double dTemp2 =0;

  //std::cout<<"M_PI: " <<M_PI<<"\n";
  dist_lat = abs(latA - latB);
  temp_lon = abs(lonA - lonB);
  dist_lon = (temp_lon<180.0)?(temp_lon):(360.0-temp_lon);
  dTemp1 = sin(0.5*dist_lat*M_PI/180.0);
  dTemp2 = sin(0.5*dist_lon*M_PI/180.0);
  dTemp = dTemp1*dTemp1 + cos(latA*M_PI/180.0)*cos(latB*M_PI/180.0)*dTemp2*dTemp2;
  
  dist = earthRadius * 2.0 * atan2(sqrt(dTemp), sqrt(1.0 - dTemp));
  
  return dist;
}

void bounds(const double& lon, const double& lat, double &minLon, double &maxLon, double &minLat, double &maxLat, const double& deltaLon, const double& deltaLat)
{
    // This function changes bounds for selecting points near to the
    // location based on how close the center point is to the poles.
    // It assumes spherical model of the earth.
    double boundPole = M_PI/2;

    maxLat = lat + deltaLat;
    minLat = lat - deltaLat;
    maxLon = lon + deltaLon;
    minLon = lon - deltaLon;
    
    if (maxLat > boundPole)
    {
        maxLat =  boundPole;
        minLon = -M_PI;
        maxLon =  M_PI;
    }
    else if (minLat < -boundPole)
    {
        minLat = -boundPole;
        minLon = -M_PI;
        maxLon = M_PI;
    }
}

double arcsine(double asinTemp)
{
    // Convert radians to degrees
    if (asinTemp > 1.0L)
        asinTemp = 1.0L;
    else if (asinTemp < -1.0L)
        asinTemp = -1.0L;
    return asin(asinTemp);
}

double lonAdjust(const double& lonA, const double& lonB)
{
    
    // This function is used to compute distance between two places
    // that lie on two different sides of 180 degrees. It primarily
    // shifts longitude such that a place with a positive longitude
    // in reference to the central position (with negative longitude) is
    // shifted to the right of the central position whereas if the central
    // position lies has a positive longitude then it shifts the location
    // with a negative longitude to the left of the central position.
    
    double lonAbs = abs(lonA) + abs(lonB);
    double lonTemp = lonB;
    
     if (2*M_PI - lonAbs < M_PI)
      {
        if (lonA < 0 && lonB > 0)
          lonTemp = lonA + ((M_PI - lonB) + (lonA + M_PI));
        else if (lonA > 0 && lonB < 0)
            lonTemp = lonA - ((M_PI - lonA) + (lonB + M_PI));
      }
    
    return lonTemp;
}





/****************************************************************************************/
/* Covariance Matrix */
/****************************************************************************************/
// The function evaluation is wrong when x=0 for b taking 
//' @title Confluent hypergeometric function of the second kind
//' @description This function calls the GSL scientific library to evaluate 
//' the confluent hypergeometric function of the second kind; see Abramowitz and Stegun 1972, p.505.
//' 
//' @param a a real value
//' @param b a real value
//' @param x a real value
//'
//'
//' @author Pulong Ma \email{mpulong@@gmail.com}
//' @seealso \code{\link{CH}}
//' @return a numerical value
// [[Rcpp::export]]
double HypergU(const double& a, const double& b, const double& x){

  gsl_sf_result_e10 result;
  gsl_set_error_handler_off();
  //gsl_sf_hyperg_U_e(a, b, x, &result);
  gsl_sf_hyperg_U_e10_e(a, b, x, &result); 

  if(b<1.0){
    if(x==0.0){
      result.val = exp(gsl_sf_lngamma(1.0-b) - gsl_sf_lngamma(a+1.0-b));
    }else if(log(x)<-0.0 && a>=9.0){ // when a is a large integer and x is close to 0, the function gsl_sf_hyperg_U_e10_e is computed incorrectly in the gsl library 
      gsl_sf_hyperg_U_e10_e(a+1e-6, b, x, &result);
    } 

  }


  

  
  return result.val;
}

//' @title Modified Bessel function of the second kind
//' @description This function calls the GSL scientific library to evaluate 
//' the modified Bessel function of the second kind.
//' 
//' @param nu a real positive value
//' @param z a real positive value
//'
//' @author Pulong Ma \email{mpulong@@gmail.com}
//' @seealso \code{\link{matern}}
//' @return a numerical value 
// [[Rcpp::export]]
double BesselK(const double& nu, const double& z){

  gsl_sf_result result;
  gsl_set_error_handler_off();
  gsl_sf_bessel_Knu_e(nu, z, &result);

  return result.val;

}

double digamma(const double& x){
  gsl_sf_result result;
  gsl_set_error_handler_off();
  gsl_sf_psi_e(x, &result);

  return result.val;
}


//' @title The Confluent Hypergeometric correlation function proposed by Ma and Bhadra (2019)
//'
//' @description This function computes the Confluent Hypergeometric correlation function given
//' a distance matrix. The Confluent Hypergeometric correlation function is given by 
//' \deqn{C(h) = \frac{\Gamma(\nu+\alpha)}{\Gamma(\nu)} 
//' \mathcal{U}\left(\alpha, 1-\nu,  \biggr(\frac{h}{\beta}\biggr)^2 \right),}
//' where \eqn{\alpha} is the tail decay parameter. \eqn{\beta} is the range parameter.
//' \eqn{\nu} is the smoothness parameter. \eqn{\mathcal{U}(\cdot)} is the confluent hypergeometric
//' function of the second kind. For details about this covariance, 
//' see Ma and Bhadra (2019) at \url{https://arxiv.org/abs/1911.05865}.  
//' 
//' @param d a matrix of distances
//' @param range a numerical value containing the range parameter 
//' @param tail a numerical value containing the tail decay parameter
//' @param nu a numerical value containing the smoothness parameter
//'
//' @author Pulong Ma \email{mpulong@@gmail.com}
//' @return a numerical matrix  
//' @seealso \link{GPBayes-package}, \code{\link{GaSP}}, \linkS4class{gp}, \code{\link{matern}}, \code{\link{kernel}}, \code{\link{ikernel}}
// [[Rcpp::export]]
Eigen::MatrixXd CH(const Eigen::MatrixXd & d, const double & range, const double & tail, const double & nu) {
  double con = exp(gsl_sf_lngamma(nu+tail) - gsl_sf_lngamma(nu));
  //double con = lgamma(nu+alpha) - lgamma(nu);

  int n1 = d.rows();
  int n2 = d.cols();
  
  Eigen::MatrixXd covmat(n1, n2);
  covmat.setOnes();
  if(range==0.0){ // limiting case with range=0
    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        if(d(i,j)!=0.0){
          covmat(i,j)=0.0;
        }
      }
    }
  }else{
    double dtemp;
    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        if(d(i,j)!=0.0){
          dtemp = d(i,j) / range;
          covmat(i, j) = con * HypergU(tail, 1.0-nu, dtemp*dtemp);
        }
      }
    }
  }
  
  // covmat = exp(con) * covmat.array();
  
  return covmat;
}


// derivative of CH w.r.t. range parameter
// [[Rcpp::export]]
Eigen::MatrixXd CH_deriv_range(const Eigen::MatrixXd& d, const double & range, const double & tail,
                const double & nu){

  double con = gsl_sf_lngamma(nu+tail) - gsl_sf_lngamma(nu);
  int n1 = d.rows();
  int n2 = d.cols();

  double dtmp;
  double con1 = exp(con) * 2.0 * tail/ range;

  Eigen::MatrixXd mat(n1, n2);
  mat.setZero();
  for(int i=0; i<n1; i++){
    for(int j=0; j<n2; j++){
      if(d(i,j)!=0.0){
        dtmp = d(i,j)/range;
        dtmp *= dtmp;
        mat(i,j) = con1 * dtmp * HypergU(tail+1.0, 2.0-nu, dtmp);
      }
    }
  }

  // mat = exp(con) * 2.0 * nu * mat.array() / range;

  return mat;
}


// derivative of CH w.r.t. tail decay parameter
// [[Rcpp::export]]
Eigen::MatrixXd CH_deriv_tail(const Eigen::MatrixXd& d, const double & range, const double & tail,
                const double & nu){

  double con = gsl_sf_lngamma(nu+tail) - gsl_sf_lngamma(nu);
  int n1 = d.rows();
  int n2 = d.cols();

  double con1 = digamma(nu+tail) - digamma(tail);
  double dtemp;

  Eigen::MatrixXd mat(n1,n2);
  mat.setZero();
  for(int i=0; i<n1; i++){
    for(int j=0; j<n2; j++){
      if(d(i,j)!=0){
        dtemp = d(i,j)/range;
        dtemp *= dtemp;
        mat(i,j) = con1*HypergU(tail, 1.0-nu, dtemp) + 
                HypergU(tail-1.0, -nu, dtemp) - 
                (nu+tail)*HypergU(tail, -nu, dtemp);
      }
    }
  }

  mat = exp(con) * mat.array();

  return mat;
}


// derivative of CH w.r.t. smoothness parameter
// [[Rcpp::export]]
Eigen::MatrixXd CH_deriv_nu(const Eigen::MatrixXd& d, const double & range, const double & tail,
                const double & nu){

  double con = gsl_sf_lngamma(nu+tail) - gsl_sf_lngamma(nu);
  int n1 = d.rows();
  int n2 = d.cols();

  double con1 = digamma(nu+tail) - digamma(tail);
  double dtemp;

  Eigen::MatrixXd mat(n1,n2);
  mat.setZero();
  for(int i=0; i<n1; i++){
    for(int j=0; j<n2; j++){
      if(d(i,j)!=0){
        dtemp = d(i,j)/range;
        dtemp *= dtemp;
        mat(i,j) = con1*HypergU(tail, 1.0-nu, dtemp) + 
                (nu+tail)*HypergU(tail, -nu, dtemp);
      }
    }
  }

  mat = exp(con) * mat.array();

  return mat;
}


// [[Rcpp::export]]
Rcpp::List deriv_ARD_CH(const Rcpp::List& d, const Eigen::VectorXd& range, const double & tail, const double & nu){
  int Dim = d.size();
  int n1, n2;
  Eigen::MatrixXd d0 = Rcpp::as<Eigen::MatrixXd>(d[0]);
  n1 = d0.rows();
  n2 = d0.cols();

  double con;
  Eigen::MatrixXd dtemp;
  Eigen::MatrixXd dscaled = Eigen::MatrixXd::Zero(n1,n2);

  con = exp(gsl_sf_lngamma(nu+tail) - gsl_sf_lngamma(nu)); 

  // derivative w.r.t. range parameters
  Rcpp::List dR(Dim+2);
  // dscaled.setZero();
  for(int k=0; k<Dim; k++){
    dR[k] = Eigen::MatrixXd::Ones(n1,n2);
    dtemp = Rcpp::as<Eigen::MatrixXd>(d[k]);
    dtemp = dtemp.array() / range(k);
    dscaled.array() += dtemp.array() * dtemp.array();    
  }

  Eigen::MatrixXd Utemp(n1,n2);
  for(int i=0; i<n1; i++){
    for(int j=0; j<n2; j++){
      Utemp(i,j) = HypergU(tail+1.0, 2.0-nu, dscaled(i,j));
    }
  }

  for(int k=0; k<Dim; k++){
    dtemp = Rcpp::as<Eigen::MatrixXd>(d[k]);
    dtemp = dtemp.array() / range(k);
    dR[k] = con * 2.0 * (tail/range(k)) * dtemp.array().square() * Utemp.array();
  }

  // derivative w.r.t. tail decay parameter
  dscaled = dscaled.array().sqrt();
  dR[Dim] = CH_deriv_tail(dscaled, 1.0, tail, nu);

  // derivative w.r.t. smoothness parameter
  dR[Dim+1] = CH_deriv_nu(dscaled, 1.0, tail, nu);

  return dR;
}

//' @title The Matérn correlation function proposed by Matérn (1960)
//' 
//' @description This function computes the Matérn correlation function given
//' a distance matrix. The Matérn correlation function is given by
//' \deqn{C(h)=\frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{h}{\phi} \right)^{\nu} 
//' \mathcal{K}_{\nu}\left( \frac{h}{\phi} \right),}
//' where \eqn{\phi} is the range parameter. \eqn{\nu} is the smoothness parameter. 
//' \eqn{\mathcal{K}_{\nu}(\cdot)} is the modified Bessel function of the second kind of order \eqn{\nu}.
//' The form of covariance includes the following special cases by specifying \eqn{\nu} to be 0.5, 1.5, 2.5.
//' \itemize{
//' \item{\eqn{\nu=0.5} corresponds to the  exponential correlation function (\strong{exp}) 
//' of the form
//' \deqn{C(h) = \exp\left\{ - \frac{h}{\phi} \right\} }
//'}
//' \item{\eqn{\nu=1.5} corresponds to the Matérn correlation function with smoothness parameter 1.5 (\strong{matern_3_2}) 
//' of the form
//' \deqn{C(h) = \left( 1 + \frac{h}{\phi} \right) \exp\left\{ - \frac{h}{\phi} \right\} }
//'}
//' \item{\eqn{\nu=2.5} corresponds to the Matérn correlation function with smoothness parameter 2.5 (\strong{matern_5_2}) 
//' of the form
//' \deqn{C(h) = \left\{ 1 + \frac{h}{\phi}  + \frac{1}{3}\left(\frac{h}{\phi}\right)^2 \right\} \exp\left\{ - \frac{h}{\phi} \right\} }
//'}
//'}
//' @param d a matrix of distances
//' @param range a numerical value containing the range parameter 
//' @param nu a numerical value containing the smoothness parameter
//'
//' @author Pulong Ma \email{mpulong@@gmail.com}
//' @return a numerical matrix 
//' @seealso \link{GPBayes-package}, \code{\link{GaSP}}, \linkS4class{gp}, \code{\link{CH}}, \code{\link{kernel}}, \code{\link{ikernel}}
//'
// [[Rcpp::export]]
Eigen::MatrixXd matern(const Eigen::MatrixXd& d, const double & range, const double & nu){
  int n1 = d.rows();
  int n2 = d.cols();
  
  Eigen::MatrixXd covmat(n1, n2);
  double tau=0.0;

  double dtemp, con1; 

  if(range==0.0){ // limiting case with range=0
    covmat.setOnes();
    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        if(d(i,j)!=0.0){
          covmat(i,j)=0.0;
        }
      }
    }    
  }else{
    if(nu==0.5){
      for(int i=0; i<n1; i++){
        for(int j=0; j<n2; j++){
          covmat(i,j) = exp(-d(i,j)/range);
        }
      }
    }else if(nu==1.5){
      for(int i=0; i<n1; i++){
        for(int j=0; j<n2; j++){
          dtemp =  d(i,j)/range;
          covmat(i,j) = (1.0+dtemp) * exp(-dtemp);
        }
      }
    }else if(nu==2.5){
      for(int i=0; i<n1; i++){
        for(int j=0; j<n2; j++){
          dtemp =  d(i,j)/range;
          covmat(i,j) = (1.0 + dtemp + dtemp*dtemp/3.0) * exp(-dtemp);
        }
      }
    }else{
      con1 = pow(2.0, 1.0-nu);
      for(int i=0; i<n1; i++){
        for(int j=0; j<n2; j++){
          if(d(i,j)==0.0){
            covmat(i,j) = 1.0;
          }else{
            tau = d(i,j)/range;
            covmat(i,j) = (con1 / tgamma(nu)) * pow(tau, nu) * BesselK(nu, tau);          
          }
        }
      }
    }
  }

  return covmat;
}

// [[Rcpp::export]]
Eigen::MatrixXd matern_deriv_range(const Eigen::MatrixXd& d, const double & range, const double & nu){

  int n1 = d.rows();
  int n2 = d.cols();
  
  Eigen::MatrixXd covmat(n1, n2);
  double tau=0.0;

  double dtemp, con1; 

  if(nu==0.5){
    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        covmat(i,j) = exp(-d(i,j)/range) * d(i,j)/(range*range);
      }
    }
  }else if(nu==1.5){
    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        dtemp = d(i,j)/range;
        covmat(i,j) = (dtemp*dtemp/range) * exp(-dtemp);
      }
    }
  }else if(nu==2.5){
    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        dtemp = d(i,j)/range;
        covmat(i,j) = (dtemp*dtemp)/(3.0*range) * (1.0 + dtemp) * exp(-dtemp);
      }
    }
  }else{
    con1 = pow(2.0, 1.0-nu);
    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        if(d(i,j)==0.0){
          covmat(i,j) = 0.0;
        }else{
          tau =  d(i,j)/range;
          covmat(i,j) =  (con1 / (range*tgamma(nu))) * pow(tau, nu) * (- 2.0*nu * BesselK(nu, tau) + tau*BesselK(nu+1.0, tau));          
        }
      }
    }
  }

  return covmat;

}

// [[Rcpp::export]]
Rcpp::List deriv_ARD_matern(const Rcpp::List& d, const Eigen::VectorXd& range, const double & nu){

  int Dim = d.size();
  int n1, n2;
  Eigen::MatrixXd d0 = Rcpp::as<Eigen::MatrixXd>(d[0]);
  n1 = d0.rows();
  n2 = d0.cols();

  double con;
  Eigen::MatrixXd dtemp;
  Eigen::MatrixXd dscaled = Eigen::MatrixXd::Zero(n1,n2);


  // derivative w.r.t. range parameters
  Rcpp::List dR(Dim+1);
  // dscaled.setZero();
  for(int k=0; k<Dim; k++){
    dR[k] = Eigen::MatrixXd::Ones(n1,n2);
    dtemp = Rcpp::as<Eigen::MatrixXd>(d[k]);
    dtemp = dtemp.array() / range(k);
    dscaled.array() += dtemp.array() * dtemp.array();    
  }

  dscaled = dscaled.array().sqrt();
  Eigen::MatrixXd d_nu = dscaled.array().pow(nu);
  Eigen::MatrixXd dMatern(n1, n2);
  if(nu==0.5){
    // dMatern = (-1.0*dscaled).array().exp() * (-1);
    for(int k=0; k<Dim; k++){
      dtemp = Rcpp::as<Eigen::MatrixXd>(d[k]);
      dtemp = dtemp.array() / range(k);
      for(int i=0; i<n1; i++){
        for(int j=0; j<n2; j++){
          if(dtemp(i,j)==0.0){
            dMatern(i,j) = 0.0;
          }else{
            dMatern(i,j) = exp(-dscaled(i,j)) * (-1.0);
            dMatern(i,j) *= -1.0/range(k) * dtemp(i,j)*dtemp(i,j) / dscaled(i,j);
          }
        }
      }
      dR[k] = dMatern;
    }
  }
  else if(nu==1.5){
    for(int k=0; k<Dim; k++){
      dtemp = Rcpp::as<Eigen::MatrixXd>(d[k]);
      dtemp = dtemp.array() / range(k);
      for(int i=0; i<n1; i++){
        for(int j=0; j<n2; j++){
          if(dtemp(i,j)==0.0){
            dMatern(i,j) = 0.0;
          }else{
            dMatern(i,j) = exp(-dscaled(i,j)) * (-dscaled(i,j));
            dMatern(i,j) *= -1.0/range(k) * dtemp(i,j)*dtemp(i,j) / dscaled(i,j);
          }
        }
      }
      dR[k] = dMatern;
    }
  }else if(nu==2.5){
    for(int k=0; k<Dim; k++){
      dtemp = Rcpp::as<Eigen::MatrixXd>(d[k]);
      dtemp = dtemp.array() / range(k);
      for(int i=0; i<n1; i++){
        for(int j=0; j<n2; j++){
          if(dtemp(i,j)==0.0){
            dMatern(i,j) = 0.0;
          }else{
            dMatern(i,j) = exp(-dscaled(i,j)) * (-1.0/3.0)*dscaled(i,j)*(1.0 + dscaled(i,j));
            dMatern(i,j) *= -1.0/range(k) * dtemp(i,j)*dtemp(i,j) / dscaled(i,j);
          }
        }
      }
      dR[k] = dMatern;
    }
  }else{ // generic value for nu
    con = pow(2.0, 1.0-nu) / tgamma(nu); 
    for(int k=0; k<Dim; k++){
      dtemp = Rcpp::as<Eigen::MatrixXd>(d[k]);
      dtemp = dtemp.array() / range(k);
      for(int i=0; i<n1; i++){
        for(int j=0; j<n2; j++){
          if(dtemp(i,j)==0.0){
            dMatern(i,j) = 0.0;
          }else{
            dMatern(i,j) = con * d_nu(i,j) * (2.0*nu/dscaled(i,j)*BesselK(nu, dscaled(i,j)) - BesselK(nu+1.0, dscaled(i,j)));
            dMatern(i,j) *= -1.0/range(k) * dtemp(i,j)*dtemp(i,j) / dscaled(i,j);
          }
        }
      }
      dR[k] = dMatern;
    }    
  }

  dR[Dim] = R_NilValue;

return dR;

}

//' @title The powered-exponential correlation function
//'
//' @description This function computes the powered-exponential correlation function given
//' a distance matrix. The powered-exponential correlation function is given by
//'                \deqn{C(h)=\exp\left\{-\left(\frac{h}{\phi}\right)^{\nu}\right\},}
//' where \eqn{\phi} is the range parameter. \eqn{\nu} is the smoothness parameter.
//' The case \eqn{\nu=2} corresponds to the well-known Gaussian correlation.
//' @param d a matrix of distances
//' @param range a numerical value containing the range parameter 
//' @param nu a numerical value containing the smoothness parameter
//' 
//' @author Pulong Ma \email{mpulong@@gmail.com}
//' @return a numerical matrix 
//' @seealso \code{\link{kernel}}
//'
// [[Rcpp::export]]
Eigen::MatrixXd powexp(const Eigen::MatrixXd& d, const double& range, const double& nu){
  int n1 = d.rows();
  int n2 = d.cols();
  
  Eigen::MatrixXd covmat(n1, n2);

  if(nu==2.0){
   for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        covmat(i,j) = exp(-(d(i,j)/range)*(d(i,j)/range));
      }
    }
  }else{

    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        covmat(i,j) = exp(-pow(d(i,j)/range, nu));
      }
    }
  }

  return covmat;
}


Eigen::MatrixXd powexp_deriv_range(const Eigen::MatrixXd& d, const double& range, const double& nu){
  int n1 = d.rows();
  int n2 = d.cols();
  
  Eigen::MatrixXd covmat(n1, n2);

  double dtmp;
  if(nu==2.0){
    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        dtmp = (d(i,j)/range) * (d(i,j)/range);
        covmat(i,j) = exp(-dtmp)*dtmp*2.0/range;
      }
    }
  }else{
    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        dtmp = pow(d(i,j)/range, nu);
        covmat(i,j) = exp(-dtmp)*dtmp*nu/range;
      }
    }    
  }


  return covmat;
}

// [[Rcpp::export]]
Rcpp::List deriv_ARD_powexp(const Rcpp::List& d, const Eigen::VectorXd& range, const double& nu){

  int Dim = d.size();
  int n1, n2;
  Eigen::MatrixXd d0 = Rcpp::as<Eigen::MatrixXd>(d[0]);
  n1 = d0.rows();
  n2 = d0.cols();

  double con;
  Eigen::MatrixXd dtemp;
  Eigen::MatrixXd dscaled = Eigen::MatrixXd::Zero(n1,n2);


  // derivative w.r.t. range parameters
  Rcpp::List dR(Dim+1);
  // dscaled.setZero();
  for(int k=0; k<Dim; k++){
    dR[k] = Eigen::MatrixXd::Ones(n1,n2);
    dtemp = Rcpp::as<Eigen::MatrixXd>(d[k]);
    dtemp = dtemp.array() / range(k);
    dscaled.array() += dtemp.array() * dtemp.array();    
  }

  dscaled = dscaled.array().sqrt();
  Eigen::MatrixXd d_nu = dscaled.array().pow(nu);
  Eigen::MatrixXd dMat(n1, n2);

  for(int k=0; k<Dim; k++){
    dtemp = Rcpp::as<Eigen::MatrixXd>(d[k]);
    dtemp = dtemp.array() / range(k);
    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        if(dtemp(i,j)==0.0){
          dMat(i,j) = 0.0;
        }else{
          con = pow(dscaled(i,j), nu-1.0);
          dMat(i,j) = exp(-dscaled(i,j)*con) * (-nu) * con;
          dMat(i,j) *= -1.0/range(k) * dtemp(i,j)*dtemp(i,j) / dscaled(i,j);
        }
      }
    }
    dR[k] = dMat;
  }
  
  // derivative w.r.t. smoothness parameter
  dR[Dim] = R_NilValue;


return dR;
}

//' @title The generalized Cauchy correlation function
//'
//' @description This function computes the generalized Cauchy correlation function given
//' a distance matrix. The generalized Cauchy covariance is given by
//' \deqn{C(h) = \left\{ 1 + \left( \frac{h}{\phi} \right)^{\nu}  
//'             \right\}^{-\alpha/\nu},}
//' where \eqn{\phi} is the range parameter. \eqn{\alpha} is the tail decay parameter.
//' \eqn{\nu} is the smoothness parameter. 
//' The case where \eqn{\nu=2} corresponds to the Cauchy covariance model, which is infinitely differentiable.
//' 
//' @param d a matrix of distances
//' @param range a numerical value containing the range parameter 
//' @param tail a numerical value containing the tail decay parameter
//' @param nu a numerical value containing the smoothness parameter
//'
//' @author Pulong Ma \email{mpulong@@gmail.com}
//' @return a numerical matrix 
//' @seealso \code{\link{kernel}}
//'
// [[Rcpp::export]]
Eigen::MatrixXd cauchy(const Eigen::MatrixXd& d, const double& range, const double& tail, const double& nu){

  int n1 = d.rows();
  int n2 = d.cols();
  
  Eigen::MatrixXd covmat(n1, n2);
  
  if(nu==2.0){
    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        covmat(i, j) = pow(1.0 + (d(i,j)/range)*(d(i,j)/range), -tail/2.0);
      }
    }    
  }else{
    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        covmat(i, j) = pow(1.0 + pow(d(i,j)/range, nu), -tail/nu);
      }
    }    
  }

  
  
  return covmat;
}


Eigen::MatrixXd cauchy_deriv_range(const Eigen::MatrixXd& d, const double& range, const double& tail, const double& nu){

  int n1 = d.rows();
  int n2 = d.cols();
  
  Eigen::MatrixXd covmat(n1, n2);
  
  double dtmp;
  if(nu==2.0){
    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        dtmp = d(i,j)/range;
        covmat(i, j) = pow(1.0 + dtmp*dtmp, -tail) * dtmp * 2.0 / (range*range);
      }
    }
  }else{
    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        dtmp = pow(d(i,j)/range, nu-1.0);
        covmat(i, j) = pow(1.0 + dtmp*d(i,j)/range, -2.0*tail/nu) * dtmp * nu / (range*range);
      }
    }    
  }
  
  
  return covmat;
}


Eigen::MatrixXd cauchy_deriv_tail(const Eigen::MatrixXd& d, const double& range, const double& tail, const double& nu){

  int n1 = d.rows();
  int n2 = d.cols();
  
  Eigen::MatrixXd covmat(n1, n2);
  
  double dtmp;
  for(int i=0; i<n1; i++){
    for(int j=0; j<n2; j++){
      dtmp = pow(d(i,j)/range, nu);
      covmat(i, j) = - pow(1.0 + dtmp, -tail/nu) * nu * log(1.0+dtmp);
    }
  }
  
  
  return covmat;
}

Rcpp::List deriv_ARD_cauchy(const Rcpp::List& d, const Eigen::VectorXd& range, const double & tail, const double & nu){
  int Dim = d.size();
  int n1, n2;
  Eigen::MatrixXd d0 = Rcpp::as<Eigen::MatrixXd>(d[0]);
  n1 = d0.rows();
  n2 = d0.cols();
  double con;
  Eigen::MatrixXd dtemp;
  Eigen::MatrixXd dscaled = Eigen::MatrixXd::Zero(n1,n2);


  // derivative w.r.t. range parameters
  Rcpp::List dR(Dim+2);
  // dscaled.setZero();
  for(int k=0; k<Dim; k++){
    dR[k] = Eigen::MatrixXd::Ones(n1,n2);
    dtemp = Rcpp::as<Eigen::MatrixXd>(d[k]);
    dtemp = dtemp.array() / range(k);
    dscaled.array() += dtemp.array() * dtemp.array();    
  }

  dscaled = dscaled.array().sqrt();
  Eigen::MatrixXd dMat(n1, n2);

  for(int k=0; k<Dim; k++){
    dtemp = Rcpp::as<Eigen::MatrixXd>(d[k]);
    dtemp = dtemp.array() / range(k);
    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        if(dtemp(i,j)==0.0){
          dMat(i,j) = 0.0;
        }else{
          con = pow(dscaled(i,j), nu-1.0);
          dMat(i,j) = -pow(1.0+con*dscaled(i,j), -tail/nu - 1.0) * tail * con;
          dMat(i,j) *= -1.0/range(k) * dtemp(i,j)*dtemp(i,j) / dscaled(i,j);
        }
      }
    }
    dR[k] = dMat;
  }

  // derivative w.r.t. tail decay parameter
  dR[Dim] = cauchy_deriv_tail(dscaled, 1.0, tail, nu);

  // derivative w.r.t. smoothness parameter
  dR[Dim+1] = R_NilValue;

  return dR;
}


/****************************************************************************************/
/****************************************************************************************/


/****************************************************************************************/
/****************************************************************************************/
Eigen::VectorXd rmvt(const Eigen::VectorXd& mu, const Eigen::MatrixXd& L, 
  const double& sigma, const double& df){
// simulate from X ~ MVT(mu, sigma^2*SIGMA, df)
// mu: q-by-1 vector of means for q-variate random vector 
// L: q-by-q matrix holding the lower cholesky factor of the scale matrix SIGMA
// sigma: a scalar that holds the sqrt root of cross-correlations across q variables 
// df: degrees of freedom

  int q = L.rows();

  Eigen::VectorXd Xmat(q);

#ifdef USE_R
    GetRNGstate();
#endif

  Xmat = mu + sigma * L * Rcpp::as<Eigen::VectorXd>(Rcpp::rnorm(q, 0.0, 1.0))
                / sqrt(Rcpp::rchisq(1, df)[0] / df);


#ifdef USE_R
  PutRNGstate();
#endif  

  return Xmat;
}




/****************************************************************************************/
/****************************************************************************************/
// x in (lb, ub) => y in (-infty, infty)
double logit(double x, double lb, double ub){
  return(log(x-lb)-log(ub-x));
}

double ilogit(double y, double lb, double ub){
  //return(lb + ( exp(y)/(1.0+exp(y))*(ub-lb) ));
  return(ub - (ub-lb)/(1.0+exp(y)));
}

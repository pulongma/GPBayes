// -*- mode: c++; -*-

#ifndef UTILS_H
#define UTILS_H


/****************************************************************************************/
/* Distance Matrices  */
/****************************************************************************************/

double gcdistance(const double& lonA, const double& latA, const double& lonB, const double& latB);
void bounds(const double& lon, const double& lat, double &minLon, double &maxLon, double &minLat, double &maxLat, const double& deltaLon, const double& deltaLat);
double arcsine(double asinTemp);
double lonAdjust(const double& lonA, const double& lonB);
double HypergU(const double& a, const double& b, const double& x);
double BesselK(const double& nu, const double& z);
double digamma(const double& x);

// correlation functions
Eigen::MatrixXd CH(const Eigen::MatrixXd& d, const double & range, const double & tail, const double & nu);
Eigen::MatrixXd matern(const Eigen::MatrixXd& d, const double & range, const double & nu);
Eigen::MatrixXd powexp(const Eigen::MatrixXd& d, const double& range, const double& nu);
Eigen::MatrixXd cauchy(const Eigen::MatrixXd& d, const double& range, const double& tail, const double& nu);

// gradient of correlation functions
//isotropic form
Eigen::MatrixXd CH_deriv_range(const Eigen::MatrixXd& d, const double & range, const double & tail,
                const double & nu);
Eigen::MatrixXd CH_deriv_tail(const Eigen::MatrixXd& d, const double & range, const double & tail,
                const double & nu);
Eigen::MatrixXd CH_deriv_nu(const Eigen::MatrixXd& d, const double & range, const double & tail,
                const double & nu);
Eigen::MatrixXd matern_deriv_range(const Eigen::MatrixXd& d, const double & range, const double & nu);
Eigen::MatrixXd powexp_deriv_range(const Eigen::MatrixXd& d, const double& range, const double& nu);
Eigen::MatrixXd cauchy_deriv_range(const Eigen::MatrixXd& d, const double& range, const double& tail, const double& nu);
Eigen::MatrixXd cauchy_deriv_tail(const Eigen::MatrixXd& d, const double& range, const double& tail, const double& nu);

//ARD form
Rcpp::List deriv_ARD_CH(const Rcpp::List& d, const Eigen::VectorXd& range, const double & tail, const double & nu);
Rcpp::List deriv_ARD_matern(const Rcpp::List& d, const Eigen::VectorXd& range, const double & nu);
Rcpp::List deriv_ARD_powexp(const Rcpp::List& d, const Eigen::VectorXd& range, const double & nu);
Rcpp::List deriv_ARD_cauchy(const Rcpp::List& d, const Eigen::VectorXd& range, const double & tail, const double & nu);

//tensor form

/****************************************************************************************/
/****************************************************************************************/


/****************************************************************************************/
/****************************************************************************************/
double logit(double x, double lb, double ub);
double ilogit(double y, double lb, double ub);
/****************************************************************************************/
/****************************************************************************************/

#endif

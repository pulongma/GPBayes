#ifndef UQ_H
#define UQ_H


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
      const Rcpp::List& d, const Eigen::VectorXd& range, const Eigen::VectorXd& tail, const Eigen::VectorXd& nu, 
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
    
    // constructor 
    UQ()
    {
      
    }
    // destructor 
    ~UQ()
    {
      
    }
};


#endif

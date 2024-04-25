// -*- model: c++; -*-


#ifndef SP_H
#define SP_H


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

    Eigen::MatrixXd FisherInfo(const Eigen::MatrixXd& d, const Rcpp::List& par, 
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
////////////////////////////////////////////////////////////////////////////////
             // END OF CLASS //
////////////////////////////////////////////////////////////////////////////////

#endif



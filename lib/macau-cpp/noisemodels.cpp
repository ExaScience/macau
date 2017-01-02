#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <cmath>

#include "noisemodels.h"
#include "macau.h"

using namespace Eigen;

////  AdaptiveGaussianNoise  ////
void AdaptiveGaussianNoise::init() {
  double se = 0.0;
  const Eigen::SparseMatrix<double> &train = macau.Y;
  const double mean_value = macau.mean_rating;

#pragma omp parallel for schedule(dynamic, 4) reduction(+:se)
  for (int k = 0; k < train.outerSize(); ++k) {
    for (SparseMatrix<double>::InnerIterator it(train,k); it; ++it) {
      se += square(it.value() - mean_value);
    }
  }

  var_total = se / train.nonZeros();
  if (var_total <= 0.0 || std::isnan(var_total)) {
    // if var cannot be computed using 1.0
    var_total = 1.0;
  }
  // Var(noise) = Var(total) / (SN + 1)
  alpha     = (sn_init + 1.0) / var_total;
  alpha_max = (sn_max + 1.0) / var_total;
}

void AdaptiveGaussianNoise::update()
{
  double sumsq = 0.0;
  const Eigen::SparseMatrix<double> &train = macau.Y;
  const double mean_value = macau.mean_rating;

#pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
  for (int j = 0; j < train.outerSize(); j++) {
    auto Vj = macau.samples[1].col(j);
    for (SparseMatrix<double>::InnerIterator it(train, j); it; ++it) {
      double Yhat = Vj.dot( macau.samples[0].col(it.row()) ) + mean_value;
      sumsq += square(Yhat - it.value());
    }
  }
  // (a0, b0) correspond to a prior of 1 sample of noise with full variance
  double a0 = 0.5;
  double b0 = 0.5 * var_total;
  double aN = a0 + train.nonZeros() / 2.0;
  double bN = b0 + sumsq / 2.0;
  alpha = rgamma(aN, 1.0 / bN);
  if (alpha > alpha_max) {
    alpha = alpha_max;
  }
}

 // Evaluation metrics
void FixedGaussianNoise::evalModel(Eigen::SparseMatrix<double> & Ytest, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var) {
   auto rmse = eval_rmse(Ytest, n, predictions, predictions_var, macau.samples[0], macau.samples[1], macau.mean_rating);
   rmse_test = rmse.second;
   rmse_test_onesample = rmse.first;
}

void AdaptiveGaussianNoise::evalModel(Eigen::SparseMatrix<double> & Ytest, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var) {
   auto rmse = eval_rmse(Ytest, n, predictions, predictions_var, macau.samples[0], macau.samples[1], macau.mean_rating);
   rmse_test = rmse.second;
   rmse_test_onesample = rmse.first;
}

inline double nCDF(double val) {return 0.5 * erfc(-val * M_SQRT1_2);}

void ProbitNoise::evalModel(Eigen::SparseMatrix<double> & Ytest, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var) {
  const unsigned N = Ytest.nonZeros();
  Eigen::VectorXd pred(N);
  Eigen::VectorXd test(N);

// #pragma omp parallel for schedule(dynamic,8) reduction(+:se, se_avg) <- dark magic :)
  for (int k = 0; k < Ytest.outerSize(); ++k) {
    int idx = Ytest.outerIndexPtr()[k];
    for (Eigen::SparseMatrix<double>::InnerIterator it(Ytest,k); it; ++it) {
     pred[idx] = nCDF( macau.samples[0].col(it.col()).dot( macau.samples[1].col(it.row())));
     test[idx] = it.value();

      // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
      double pred_avg;
      if (n == 0) {
        pred_avg = pred[idx];
      } else {
        double delta = pred[idx] - predictions[idx];
        pred_avg = (predictions[idx] + delta / (n + 1));
        predictions_var[idx] += delta * (pred[idx] - pred_avg);
      }
      predictions[idx++] = pred_avg;
   }
  }
  auc_test_onesample = auc(pred,test);
  auc_test = auc(predictions, test);
}

std::pair<double, double> ProbitNoise::sample(int n, int m) {
    double y = macau.Y.coeffRef(n,m);
    const VectorXd &u = macau.samples[0].col(n);
    const VectorXd &v = macau.samples[1].col(n);
    double z = (2 * y - 1) * fabs(v.dot(u) + bmrandn_single());
    return std::make_pair(1, z);
}

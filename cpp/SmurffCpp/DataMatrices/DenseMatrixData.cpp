#include "DenseMatrixData.h"

namespace smurff {

DenseMatrixData::DenseMatrixData(Matrix Y)
   : FullMatrixData<Matrix>(Y)
{
    this->name = "DenseMatrixData [fully known]";
}

//d is an index of column in U matrix
void DenseMatrixData::getMuLambda(const SubModel& model, uint32_t mode, int d, Vector& rr, Matrix& MM) const
{
    auto &Y = this->Y(mode).row(d);
    auto Vf = *model.CVbegin(mode);
    auto &ns = noise();

    for(int r = 0; r<Y.cols(); ++r) 
    {
        const auto &row = Vf.row(r);
        PVec<> pos = this->pos(mode, d, r);
        double noisy_val = ns.sample(model, pos, Y(r));
        rr.noalias() += row * noisy_val; // rr = rr + (V[m] * noisy_y[d]) 
    }

    MM.noalias() += ns.getAlpha() * VV[mode]; // MM = MM + VV[m]
}

double DenseMatrixData::train_rmse(const SubModel& model) const
{
   return std::sqrt(sumsq(model) / this->size());
}

double DenseMatrixData::var_total() const
{
   double cwise_mean = this->sum() / this->size();
   double se = (Y().array() - cwise_mean).square().sum();
   
   double var = se / this->size();
   if (var <= 0.0 || std::isnan(var))
   {
      // if var cannot be computed using 1.0
      var = 1.0;
   }

   return var;
}

// for the adaptive gaussian noise
double DenseMatrixData::sumsq(const SubModel& model) const
{
   double sumsq = 0.0;

   #pragma omp parallel for schedule(guided) reduction(+:sumsq)
   for (int j = 0; j < this->ncol(); j++) 
   {
      for (int i = 0; i < this->nrow(); i++) 
      {
         sumsq += std::pow(model.predict({i,j}) - this->Y()(i,j), 2);
      }
   }

   return sumsq;
}
} // end namespace smurff

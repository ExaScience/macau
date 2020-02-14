#pragma once

#include <memory>
#include <set>

#include <SmurffCpp/Types.h>

#include <SmurffCpp/Utils/PVec.hpp>
#include <Utils/ThreadVector.hpp>
#include <Utils/Error.h>

#include <SmurffCpp/Configs/Config.h>

namespace smurff {

class Step;

class Data;

class SubModel;

template<class T>
class VMatrixExprIterator;

template<class T>
class ConstVMatrixExprIterator;

template<class T>
class VMatrixIterator;

template<class T>
class ConstVMatrixIterator;

class Model : public std::enable_shared_from_this<Model>
{
private:
   std::vector<std::shared_ptr<Matrix>> m_factors; //vector of U matrices
   std::vector<std::shared_ptr<Matrix>> m_link_matrices; //vector of U matrices
   std::vector<std::shared_ptr<Vector>> m_mus; //vector of mu vectors

   bool m_collect_aggr;
   std::vector<std::shared_ptr<Matrix>> m_aggr_sum; //vector of aggr summed m_factors matrices
   std::vector<std::shared_ptr<Matrix>> m_aggr_dot; //vector of aggr dot m_factors matrices
   std::vector<int> m_num_aggr; //number of aggregated samples in above vectors

   int m_num_latent; //size of latent dimention for U matrices
   PVec<> m_dims; //dimensions of train data

   // to make predictions faster
   mutable thread_vector<Array1D> Pcache;

public:
   Model();

public:
   //initialize U matrices in the model (random/zero)
   void init(int num_latent, const PVec<>& dims, ModelInitTypes model_init_type, bool save_model, bool collect_aggr = true);

   void setLinkMatrix(int mode, std::shared_ptr<Matrix>, std::shared_ptr<Vector>);

public:
   //dot product of i'th columns in each U matrix
   //pos - vector of column indices
   double predict(const PVec<>& pos) const;

   // for each row in feature matrix f: compute latent vector from feature vector
   template<typename FeatMatrix>
   std::shared_ptr<Matrix> predict_latent(int mode, const FeatMatrix& f);

   // for each row in feature matrix f: predict full column based on feature vector
   template<typename FeatMatrix>
   const Matrix predict(int mode, const FeatMatrix& f);

public:
   //return f'th U matrix in the model
   Matrix &U(uint32_t f);

   const Matrix &U(uint32_t f) const;

   //return V matrices in the model opposite to mode
   VMatrixIterator<Matrix> Vbegin(std::uint32_t mode);
   
   VMatrixIterator<Matrix> Vend();

   ConstVMatrixIterator<Matrix> CVbegin(std::uint32_t mode) const;
   
   ConstVMatrixIterator<Matrix> CVend() const;

   //return i'th column of f'th U matrix in the model
   Matrix::ConstColXpr col(int f, int i) const;

public:
   //number of dimentions in train data
   std::uint64_t nmodes() const;

   //size of latent dimention
   int nlatent() const;

   //sum of number of columns in each U matrix in the model
   int nsamples() const;

public:
   //vector if dimention sizes of train data
   const PVec<>& getDims() const;

public:
   //returns SubModel proxy class with offset to the first column of each U matrix in the model
   SubModel full();

public:
   void updateAggr(int m);
   void updateAggr(int m, int n);

public:
   // output to file
   void save(std::shared_ptr<const Step> sf) const;
   bool m_save_model = true;
   bool m_save_aggr = false;
   void restore(std::shared_ptr<const Step> sf, int skip_mode = -1);

   std::ostream& info(std::ostream &os, std::string indent) const;
   std::ostream& status(std::ostream &os, std::string indent) const;
};




// SubModel is a proxy class that allows to access i'th column of each U matrix in the model
class SubModel
{
private:
   const Model &m_model;
   const PVec<> m_off;
   const PVec<> m_dims;

public:
   SubModel(const Model &m, const PVec<> o, const PVec<> d)
      : m_model(m), m_off(o), m_dims(d) {}

   SubModel(const SubModel &m, const PVec<> o, const PVec<> d)
      : m_model(m.m_model), m_off(o + m.m_off), m_dims(d) {}

   SubModel(const Model &m)
      : m_model(m), m_off(m.nmodes()), m_dims(m.getDims()) {}

public:
   Matrix::ConstBlockXpr U(int f) const;

   ConstVMatrixExprIterator<Matrix::ConstBlockXpr> CVbegin(std::uint32_t mode) const;
   ConstVMatrixExprIterator<Matrix::ConstBlockXpr> CVend() const;

public:
   //dot product of i'th columns in each U matrix
   double predict(const PVec<> &pos) const
   {
      return m_model.predict(m_off + pos);
   }

   //size of latent dimention
   int nlatent() const
   {
      return m_model.nlatent();
   }

   //number of dimentions in train data
   std::uint64_t nmodes() const
   {
      return m_model.nmodes();
   }
};


template<typename FeatMatrix>
std::shared_ptr<Matrix> Model::predict_latent(int mode, const FeatMatrix& f)
{
   THROWERROR_ASSERT_MSG(m_link_matrices.at(mode),
      "No link matrix available in mode " + std::to_string(mode));

   const auto &beta = *m_link_matrices.at(mode);
   const auto &mu = *m_mus.at(mode);

   auto ret = std::make_shared<Matrix>(nlatent(), f.rows());
   *ret = beta * f.transpose();
   ret->colwise() += mu;
   #if 0
   std::cout << "beta =\n" << beta.transpose() << std::endl;
   std::cout << "f =\n" << f << std::endl;
   std::cout << "beta * f.transpose() =\n" << beta * f.transpose() << std::endl;
   std::cout << "Umean: " << Umean << std::endl;
   std::cout << "U: " << U(mode) << std::endl;
   std::cout << "ret: " << *ret << std::endl;
   #endif

   return ret;
}

template<typename FeatMatrix>
const Matrix Model::predict(int mode, const FeatMatrix& f)
{
   THROWERROR_ASSERT_MSG(nmodes() == 2,
      "Only implemented for modes == 2");

   auto latent = predict_latent(mode, f);

   int othermode = (mode+1) % 2;

   #if 0
   std::cout << "predicted latent: " << *latent << std::endl;
   std::cout << "other U: " << U(othermode) << std::endl;
   std::cout << "ret: " << latent->transpose() * U(othermode) << std::endl;
   #endif

   return latent->transpose() * U(othermode);
}

};

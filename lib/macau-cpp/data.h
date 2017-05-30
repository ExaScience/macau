#ifndef DATA_H
#define DATA_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <numeric>
#include <memory>

#include "noisemodels.h"
#include "matrix_io.h"
#include "utils.h"

namespace Macau {

struct Data {
    Data() : center_mode(CENTER_INVALID) {}

    // init
    virtual void init_base() = 0;
    virtual void center(double upper_mean) = 0;
    virtual void init();

    // helper functions for noise
    virtual double sumsq(const SubModel &) const = 0;
    virtual double var_total() const = 0;

    // update noise and precision/mean
    virtual double train_rmse(const SubModel &) const = 0;
    virtual void update(const SubModel &model) { noise->update(model); }
    virtual void get_pnm(const SubModel &,int,int,VectorNd &, MatrixNNd &) = 0;
    virtual void update_pnm(const SubModel &,int) = 0;

    //-- print info
    virtual std::ostream &info(std::ostream &os, std::string indent);

    // set noise models
    FixedGaussianNoise &setPrecision(double p);
    AdaptiveGaussianNoise &setAdaptivePrecision(double sn_init, double sn_max);
    ProbitNoise &setProbit();

    // virtual functions data-related
    virtual int              nmode() const = 0;
    virtual int              nnz()   const = 0;
    virtual int              size()  const = 0;
    virtual int              nna()   const = 0;
    virtual std::vector<int> dims()  const = 0;
    virtual double           sum()   const = 0;

    // mean & centering
    double cwise_mean = NAN, global_mean = NAN;
    double mean(int m, int c) const { assert(mean_computed); return mode_mean.at(m)(c); }
    virtual double compute_mode_mean(int,int) = 0;
            void compute_mode_mean();
    virtual double offset_to_mean(std::vector<int> pos) const = 0;

    virtual void setCenterMode(std::string c);
    enum { CENTER_INVALID = -10, CENTER_NONE = -3, CENTER_GLOBAL = -2, CENTER_VIEW = -1, CENTER_COLS = 0, CENTER_ROWS = 1}
    center_mode;

    // helper for predictions
    double predict(const std::vector<int> &pos, const SubModel &model) const;

    // name
    std::string                  name;

    // noise model for this dataset
    std::unique_ptr<INoiseModel> noise;

  protected:
    std::vector<Eigen::VectorXd>  mode_mean;
    bool mean_computed = false;
    bool centered = false;
};

struct MatrixData: public Data {
            int nmode()     const override { return 2; }
    virtual int nrow()      const = 0;
    virtual int ncol()      const = 0;
            int size()      const override { return nrow() * ncol(); }
    std::vector<int> dims() const override { return {ncol(), nrow()}; }
    std::ostream &info(std::ostream &os, std::string indent) override;
};

struct MatricesData: public MatrixData {
    MatricesData() { name = "MatricesData"; }

    void init_base() override;
    void setCenterMode(std::string c) override;

    void center(double) override;
    double compute_mode_mean(int,int) override;
    double offset_to_mean(std::vector<int> pos) const override;

    // add data
    MatrixData &add(int, int, const std::unique_ptr<MatrixData>);

    // helper functions for noise
    // but 
    double sumsq(const SubModel &) const override { assert(false); return NAN; }
    double var_total() const override { assert(false); return NAN; }
    double train_rmse(const SubModel &) const override;

    // update noise and precision/mean
    void get_pnm(const SubModel &,int,int,VectorNd &, MatrixNNd &) override;
    void update_pnm(const SubModel &model, int mode) override;
  
    //-- print info
    std::ostream &info(std::ostream &os, std::string indent) override;

    // virtual functions data-related

    int  nnz()  const override { return std::accumulate(matrices.begin(), matrices.end(), 0,
            [](int s, const std::pair<const std::pair<int,int>, std::unique_ptr<MatrixData>> &m) -> int { return  s + m.second->nnz(); });  }           
    int  nrow() const override { return std::accumulate(rowdims.begin(), rowdims.end(), 0,
            [](int s, const std::pair<int,int> &p) -> int { return  s + p.second; }); }           
    int  ncol() const override { return std::accumulate(coldims.begin(), coldims.end(), 0,
            [](int s, const std::pair<int,int> &p) -> int { return  s + p.second; }); }           
    double sum() const override { return std::accumulate(matrices.begin(), matrices.end(), .0,
            [](double s, const std::pair<const std::pair<int,int>, std::unique_ptr<MatrixData>> &m) -> double { return  s + m.second->sum(); });  }        
    int  nna()  const override { return std::accumulate(matrices.begin(), matrices.end(), 0,
            [](int s, const std::pair<const std::pair<int,int>, std::unique_ptr<MatrixData>> &m) -> int { return  s + m.second->nna(); });  }           

    // specific stuff 
    std::vector<int> bdims(int brow, int bcol) const;
    std::vector<int> boffs(int brow, int bcol) const;
    SubModel submodel(const SubModel &model, int brow, int bcol); 

  private:
    std::map<std::pair<int,int>, std::unique_ptr<MatrixData>> matrices;
    std::map<int,int> rowdims, coldims;
};


template<typename YType>
struct MatrixDataTempl : public MatrixData {
    MatrixDataTempl(YType Y) : Y(Y) {}

    //init and center
    void init_base() override;

    int    nrow()  const override { return Y.rows(); }
    int    ncol()  const override { return Y.cols(); }
    int    nnz()   const override { return Y.nonZeros(); }
    double sum()   const override { return Y.sum(); }

    double offset_to_mean(std::vector<int> pos) const override;

    double var_total() const override;
    double sumsq(const SubModel &) const override;

    YType Y;
    std::vector<YType> Yc; // centered versions

};

struct ScarceMatrixData : public MatrixDataTempl<SparseMatrixD> {
    //-- c'tor
    ScarceMatrixData(SparseMatrixD Y)
        : MatrixDataTempl<SparseMatrixD>(Y) 
    {
        name = "ScarceMatrixData [with NAs]";
    }

    void init_base() override;
    void center(double) override;
    double compute_mode_mean(int,int) override;

    double train_rmse(const SubModel &) const override;

    std::ostream &info(std::ostream &os, std::string indent) override;

    void get_pnm(const SubModel &,int,int,VectorNd &, MatrixNNd &) override;
    void update_pnm(const SubModel &,int) override {}

    int    nna()             const override { return size() - nnz(); }

  private:
    int num_empty[2] = {0,0};

};

struct ScarceBinaryMatrixData : public ScarceMatrixData {
    //-- c'tor
    ScarceBinaryMatrixData(SparseMatrixD &Y) : ScarceMatrixData(Y) 
    {
        name = "ScarceBinaryMatrixData [containing 0,1,NA]";
    }

    void get_pnm(const SubModel &,int,int,VectorNd &, MatrixNNd &) override;
    void update_pnm(const SubModel &,int) override {};

    int    nna()   const override { return size() - nnz(); }

};

template<class YType>
struct FullMatrixData : public MatrixDataTempl<YType> {
    //-- c'tor
    FullMatrixData(YType Y) : MatrixDataTempl<YType>(Y)
    {
        this->name = "MatrixData [fully known]";
    }


    void get_pnm(const SubModel &,int,int,VectorNd &, MatrixNNd &) override;
    void update_pnm(const SubModel &,int) override;

    int    nna()   const override { return 0; }

  private:
    Eigen::MatrixXd VV[2];

    double compute_mode_mean(int,int) override;
};

struct DenseMatrixData : public FullMatrixData<Eigen::MatrixXd> {
    //-- c'tor
    DenseMatrixData(Eigen::MatrixXd Y) : FullMatrixData<Eigen::MatrixXd>(Y)
    {
        this->name = "DenseMatrixData [fully known]";
    }
    void center(double) override;
    double train_rmse(const SubModel &) const override;
};

struct SparseMatrixData : public FullMatrixData<Eigen::SparseMatrix<double>> {
    //-- c'tor
    SparseMatrixData(Eigen::SparseMatrix<double> Y) : FullMatrixData<Eigen::SparseMatrix<double>>(Y)
    {
        this->name = "SparseMatrixData [fully known]";
    }
    void center(double) override;
    double train_rmse(const SubModel &) const override;
};

}; // end namespace Macau

#endif

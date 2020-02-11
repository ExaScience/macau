#pragma once

#include <memory>

#include <SmurffCpp/ResultItem.h>
#include <SmurffCpp/Configs/MatrixConfig.h>
#include <SmurffCpp/DataTensors/SparseMode.h>

namespace smurff {

class StepFile;
class RootFile;

class Model;
class Data;

template<typename Item, typename Compare>
double calc_auc(const std::vector<Item> &predictions,
                double threshold,
                const Compare &compare)
{
    auto sorted_predictions = predictions;
    std::sort(sorted_predictions.begin(), sorted_predictions.end(), compare);

    int num_positive = 0;
    int num_negative = 0;
    double auc = .0;

    for(auto &t : sorted_predictions)
    {
        int is_positive = t.val > threshold;
        int is_negative = !is_positive;
        num_positive += is_positive;
        num_negative += is_negative;
        auc += is_positive * num_negative;
    }

    auc /= num_positive;
    auc /= num_negative;
    return auc;
}

template<typename Item>
double calc_auc(const std::vector<Item> &predictions, double threshold)
{
   return calc_auc(predictions, threshold, [](const Item &a, const Item &b) { return a.pred < b.pred;});
}

class Result
{
public:
   //c'tor with sparse TensorConfig
   Result(std::shared_ptr<TensorConfig> Y, int nsamples = 0);

   //fill with dense value
   Result(PVec<> lo, PVec<> hi, double value, int nsamples = 0);

   //empty c'tor
   Result();

public:
   //sparse representation of test matrix
   std::vector<ResultItem> m_predictions;

   //dimensions of Ytest
   PVec<> m_dims;

   //-- prediction metrics
   void update(std::shared_ptr<const Model> model, bool burnin);

public:
   double rmse_avg = NAN;
   double rmse_1sample = NAN;
   double auc_avg = NAN;
   double auc_1sample = NAN;
   int sample_iter = 0;
   int burnin_iter = 0;

   // general

public:
   void save(std::shared_ptr<const StepFile> sf) const;
   void restore(std::shared_ptr<const StepFile> sf);

private:
   template<typename Accessor>
   std::shared_ptr<const SparseMatrix> toMatrix(const Accessor &acc) const;

public:
   void init();

public:
   void toCsv(std::string filename) const;
   std::ostream &info(std::ostream &os, std::string indent);

   //-- for binary classification
   int total_pos = -1;
   bool classify = false;
   double threshold;

   //-- save predictions to file?
   bool m_save_pred = true;

   void setThreshold(double t)
   {
      threshold = t; classify = true;
   }

   void setSavePred(bool v)
   {
      m_save_pred = v;
   }

public:
   bool isEmpty() const;
};

};
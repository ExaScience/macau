#pragma once

#include <memory>

#include <Eigen/Sparse>
#include <Eigen/Core>

#include <SmurffCpp/Utils/PVec.hpp>
#include <SmurffCpp/Sessions/ISession.h>
#include <SmurffCpp/Model.h>

namespace smurff {

class RootFile;
class Result;
struct ResultItem;

class PredictSession : public ISession
{
private:
    std::shared_ptr<RootFile> m_model_rootfile;
    std::shared_ptr<RootFile> m_pred_rootfile;
    Config m_config;
    bool m_has_config;

    std::shared_ptr<Result> m_result;
    std::vector<std::shared_ptr<StepFile>>::reverse_iterator m_pos;

    double m_secs_per_iter;
    double m_secs_total;
    int m_iter;

    std::vector<std::shared_ptr<StepFile>> m_stepfiles;

    int m_num_latent;
    PVec<> m_dims;
    bool m_is_init;

private:
    std::shared_ptr<Model> restoreModel(const std::shared_ptr<StepFile> &);
    std::shared_ptr<Model> restoreModel(int i);

public:
    int    getNumSteps()  const { return m_stepfiles.size(); } 
    int    getNumLatent() const { return m_num_latent; } 
    PVec<> getModelDims() const { return m_dims; } 

public:
    // ISession interface 
    void run() override;
    bool step() override;
    void init() override;

    std::shared_ptr<StatusItem> getStatus() const override;
    std::shared_ptr<Result>     getResult() const override;

    std::shared_ptr<RootFile> getRootFile() const override {
        return m_pred_rootfile;
    }

private:
    void save();

    std::shared_ptr<RootFile> getModelRoot() const {
        return m_model_rootfile;
    }

  public:
    PredictSession(const Config &config);
    PredictSession(std::shared_ptr<RootFile> rf, const Config &config);
    PredictSession(std::shared_ptr<RootFile> rf);

    std::ostream& info(std::ostream &os, std::string indent) const override;

    // predict one element - based on position only
    ResultItem predict(PVec<> Ytest);
    
    ResultItem predict(PVec<> Ytest, const StepFile &sf);

    // predict one element - based on ResultItem
    void predict(ResultItem &);
    void predict(ResultItem &, const StepFile &sf);

    // predict all elements in Ytest
    std::shared_ptr<Result> predict(std::shared_ptr<TensorConfig> Y);
    void predict(Result &, const StepFile &);

    // predict element or elements based on sideinfo
    template <class Feat>
    std::vector<std::shared_ptr<Eigen::MatrixXd>> predict(int mode, const Feat &f);
};

// predict element or elements based on sideinfo
template <class Feat>
std::vector<std::shared_ptr<Eigen::MatrixXd>> PredictSession::predict(int mode, const Feat &f)
{
    std::vector<std::shared_ptr<Eigen::MatrixXd>> ret;

    for (int step=0; step<getNumSteps(); step++)
    {
        const auto &sf = m_stepfiles.at(step);
        auto predictions = std::make_shared<Eigen::MatrixXd>(restoreModel(sf)->predict(mode, f));
        ret.push_back(predictions);
    }

    return ret;
}

} // end namespace smurff
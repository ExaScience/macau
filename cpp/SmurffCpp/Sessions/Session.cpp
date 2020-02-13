#include "Session.h"


#include <fstream>
#include <string>
#include <iomanip>

#include <Utils/omp_util.h>
#include <SmurffCpp/Utils/Distribution.h>
#include <SmurffCpp/Utils/MatrixUtils.h>
#include <Utils/counters.h>
#include <Utils/Error.h>
#include <Utils/StringUtils.h>
#include <SmurffCpp/Configs/Config.h>

#include <SmurffCpp/DataMatrices/DataCreator.h>
#include <SmurffCpp/Priors/PriorFactory.h>

#include <SmurffCpp/result.h>
#include <SmurffCpp/StatusItem.h>

namespace smurff {
Session::Session()
    : m_model(std::make_shared<Model>()), m_pred()
{
    name = "Session";
}

void Session::fromRootPath(std::string rootPath)
{
    // open root file
    m_rootFile = std::make_shared<RootFile>(rootPath);

    //restore config
    m_rootFile->restoreConfig(m_config);

    m_config.validate();

    //base functionality
    setFromBase();
}

void Session::fromConfig(const Config &cfg)
{
    cfg.validate();

    // assign config
    m_config = cfg;

    if (!cfg.getRootName().empty())
    {
        // open root file
        m_rootFile = std::make_shared<RootFile>(cfg.getRootName());
    }
    else if (m_config.getSaveFreq() || m_config.getCheckpointFreq())
    {

        // create root file
        m_rootFile = std::make_shared<RootFile>(m_config.getSavePrefix() + "root.h5", true);

        //save config
        m_rootFile->saveConfig(m_config);
    }

    //base functionality
    setFromBase();
}

void Session::setFromBase()
{
    std::shared_ptr<Session> this_session = shared_from_this();

    // initialize pred
    if (m_config.getTest())
    {
        m_pred = std::make_shared<Result>(m_config.getTest());
        m_pred->setSavePred(m_config.getSavePred());
        if (m_config.getClassify())
            m_pred->setThreshold(m_config.getThreshold());
    }
    else 
    {
        m_pred = std::make_shared<Result>();
    }

    // initialize data

    data_ptr = m_config.getTrain()->create(std::make_shared<DataCreator>(this_session));

    // initialize priors

    std::shared_ptr<IPriorFactory> priorFactory = this->create_prior_factory();
    for (std::size_t i = 0; i < m_config.getPriorTypes().size(); i++)
        this->addPrior(priorFactory->create_prior(this_session, i));
}

void Session::addPrior(std::shared_ptr<ILatentPrior> prior)
{
   prior->setMode(m_priors.size());
   m_priors.push_back(prior);
}

void Session::init()
{
    //init omp
    threads::init(m_config.getVerbose(), m_config.getNumThreads());

    //initialize random generator
    initRng();

    //init performance counters
    perf_data_init();

    //initialize test set
    if (m_pred) m_pred->init();

    //initialize train matrix (centring and noise model)
    data().init();

    //initialize model (samples)
    model().init(m_config.getNumLatent(), data().dim(), m_config.getModelInitType(), m_config.getSaveModel());

    //initialize priors
    for (auto &p : m_priors)
        p->init();

    //write info to console
    if (m_config.getVerbose())
        info(std::cout, "");

    //restore session (model, priors)
    bool resume = restore(m_iter);

    //print session status to console
    if (m_config.getVerbose())
    {
        printStatus(std::cout, resume);
    }

    is_init = true;
}

void Session::run()
{
    init();
    while (step())
        ;
}

bool Session::step()
{
    COUNTER("step");
    THROWERROR_ASSERT_MSG(is_init, "Session::init() needs to be called before ::step()")

    // go to the next iteration
    m_iter++;

    bool isStep = m_iter < m_config.getBurnin() + m_config.getNSamples();

    if (isStep)
    {
        THROWERROR_ASSERT(is_init);

        auto starti = tick();
        for (auto &p : m_priors)
        {
            p->sample_latents();
            p->update_prior();
        }
        
        data().update(model());
        auto endi = tick();

        //WARNING: update is an expensive operation because of sort (when calculating AUC)
        m_pred->update(m_model, m_iter < m_config.getBurnin());

        m_secs_per_iter = endi - starti;
        m_secs_total += m_secs_per_iter;

        printStatus(std::cout);

        save();
    }

    return isStep;
}

std::ostream &Session::info(std::ostream &os, std::string indent) const
{
    os << indent << name << " {\n";
    os << indent << "  Data: {" << std::endl;
    data().info(os, indent + "    ");
    os << indent << "  }" << std::endl;
    os << indent << "  Model: {" << std::endl;
    model().info(os, indent + "    ");
    os << indent << "  }" << std::endl;
    os << indent << "  Priors: {" << std::endl;
    for (auto &p : m_priors)
        p->info(os, indent + "    ");
    os << indent << "  }" << std::endl;
    os << indent << "  Result: {" << std::endl;
    m_pred->info(os, indent + "    ");
    os << indent << "  }" << std::endl;
    os << indent << "  Config: {" << std::endl;
    m_config.info(os, indent + "    ");
    os << indent << "  }" << std::endl;
    os << indent << "}\n";
    return os;
}

void Session::save()
{
    //do not save if 'never save' mode is selected
    if (!m_config.getSaveFreq() &&
        !m_config.getCheckpointFreq())
        return;

    std::int32_t isample = m_iter - m_config.getBurnin() + 1;

    //save if checkpoint threshold overdue
    if (m_config.getCheckpointFreq() && (tick() - m_lastCheckpointTime) >= m_config.getCheckpointFreq())
    {
        std::int32_t icheckpoint = m_iter + 1;

        //save this iteration
        saveInternal(icheckpoint, true);

        //remove previous iteration if required (initial m_lastCheckpointIter is -1 which means that it does not exist)
        m_rootFile->removeOldCheckpoints();

        //upddate counters
        m_lastCheckpointTime = tick();
        m_lastCheckpointIter = m_iter;
    }

    //save model during sampling stage
    if (m_config.getSaveFreq() && isample > 0)
    {
        //save_freq > 0: check modulo - do not save if not a save iteration
        if (m_config.getSaveFreq() > 0 && (isample % m_config.getSaveFreq()) != 0)
        {
            // don't save
        }
        //save_freq < 0: save last iter - do not save if (final model) mode is selected and not a final iteration
        else if (m_config.getSaveFreq() < 0 && isample < m_config.getNSamples())
        {
            // don't save
        }
        else
        {
            //do save this iteration
            saveInternal(isample, false);
        }
    }
}

void Session::saveInternal(int iteration, bool checkpoint)
{
    std::shared_ptr<StepFile> stepFile = m_rootFile->createStepFile(iteration, checkpoint);

    if (m_config.getVerbose())
    {
        std::cout << "-- Saving model, predictions,... into '" << m_rootFile->getFullPath() << "'." << std::endl;
    }
    double start = tick();

    stepFile->save(m_model, m_pred, m_priors);

    double stop = tick();
    if (m_config.getVerbose())
    {
        std::cout << "-- Done saving model. Took " << stop - start << " seconds." << std::endl;
    }
}

bool Session::restore(int &iteration)
{
    std::shared_ptr<StepFile> stepFile = nullptr;
    if (m_rootFile)
    {
        stepFile = m_rootFile->openLastCheckpoint();
    }

    if (!stepFile)
    {
        //if there is nothing to restore - start from initial iteration
        iteration = -1;

        //to keep track at what time we last checkpointed
        m_lastCheckpointTime = tick();
        m_lastCheckpointIter = -1;
        return false;
    }
    else
    {
        if (m_config.getVerbose())
        {
            std::cout << "-- Restoring model, predictions,... from '" << m_rootFile->getFullPath() << "'." << std::endl;
        }

        stepFile->restore(m_model, m_pred, m_priors);

        //restore last iteration index
        if (stepFile->isCheckpoint())
        {
            iteration = stepFile->getIsample() - 1; //restore original state

            //to keep track at what time we last checkpointed
            m_lastCheckpointTime = tick();
            m_lastCheckpointIter = iteration;
        }
        else
        {
            iteration = stepFile->getIsample() + m_config.getBurnin() - 1; //restore original state

            //to keep track at what time we last checkpointed
            m_lastCheckpointTime = tick();
            m_lastCheckpointIter = iteration;
        }

        return true;
    }
}

std::shared_ptr<Result> Session::getResult() const
{
   return m_pred;
}

std::shared_ptr<StatusItem> Session::getStatus() const
{
    std::shared_ptr<StatusItem> ret = std::make_shared<StatusItem>();

    if (m_iter < 0)
    {
        ret->phase = "Initial";
        ret->iter = m_iter + 1;
        ret->phase_iter = 0;
    }
    else if (m_iter < m_config.getBurnin())
    {
        ret->phase = "Burnin";
        ret->iter = m_iter + 1;
        ret->phase_iter = m_config.getBurnin();
    }
    else
    {
        ret->phase = "Sample";
        ret->iter = m_iter - m_config.getBurnin() + 1;
        ret->phase_iter = m_config.getNSamples();
    }

    for (int i = 0; i < (int)model().nmodes(); ++i)
    {
        ret->model_norms.push_back(model().U(i).norm());
    }

    ret->train_rmse = data().train_rmse(model());

    ret->rmse_avg = m_pred->rmse_avg;
    ret->rmse_1sample = m_pred->rmse_1sample;

    ret->auc_avg = m_pred->auc_avg;
    ret->auc_1sample = m_pred->auc_1sample;

    ret->elapsed_iter = m_secs_per_iter;
    ret->elapsed_total = m_secs_total;

    ret->nnz_per_sec = (double)(data().nnz()) / m_secs_per_iter;
    ret->samples_per_sec = (double)(model().nsamples()) / m_secs_per_iter;

    return ret;
}

void Session::printStatus(std::ostream &output, bool resume)
{
    if (!m_config.getVerbose())
        return;

    auto status_item = getStatus();

    std::string resumeString = resume ? "Continue from " : std::string();

    if (m_config.getVerbose() > 0)
    {
        if (m_iter < 0)
        {
            output << " ====== Initial phase ====== " << std::endl;
        }
        else if (m_iter < m_config.getBurnin() && m_iter == 0)
        {
            output << " ====== Sampling (burning phase) ====== " << std::endl;
        }
        else if (m_iter == m_config.getBurnin())
        {
            output << " ====== Burn-in complete, averaging samples ====== " << std::endl;
        }

        output << resumeString << status_item->asString() << std::endl;

        if (m_config.getVerbose() > 1)
        {
            output << std::fixed << std::setprecision(4) << "  RMSE train: " << status_item->train_rmse << std::endl;
            output << "  Priors:" << std::endl;

            for (const auto &p : m_priors)
                p->status(output, "     ");

            output << "  Model:" << std::endl;
            model().status(output, "    ");
            output << "  Noise:" << std::endl;
            data().status(output, "    ");
        }

        if (m_config.getVerbose() > 2)
        {
            output << "  Compute Performance: " << status_item->samples_per_sec << " samples/sec, " << status_item->nnz_per_sec << " nnz/sec" << std::endl;
        }
    }
}

std::string StatusItem::getCsvHeader()
{
    return "phase;iter;phase_len;rmse_avg;rmse_1samp;train_rmse;auc_avg;auc_1samp;elapsed_1samp;elapsed_total";
}

std::string StatusItem::asCsvString() const
{
    char ret[1024];
    snprintf(ret, 1024, "%s;%d;%d;%.4f;%.4f;%.4f;%.4f;:%.4f;%0.1f;%0.1f",
             phase.c_str(), iter, phase_iter, rmse_avg, rmse_1sample, train_rmse,
             auc_1sample, auc_avg, elapsed_iter, elapsed_total);

    return ret;
}

void Session::initRng()
{
    //init random generator
    if (m_config.getRandomSeedSet())
        init_bmrng(m_config.getRandomSeed());
    else
        init_bmrng();
}

std::shared_ptr<IPriorFactory> Session::create_prior_factory() const
{
    return std::make_shared<PriorFactory>();
}
} // end namespace smurff

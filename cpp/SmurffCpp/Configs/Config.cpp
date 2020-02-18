#include "Config.h"

#ifdef _WINDOWS
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <set>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <cstdio>
#include <string>
#include <memory>

#include <Utils/Error.h>
#include <SmurffCpp/Utils/TensorUtils.h>
#include <SmurffCpp/IO/INIFile.h>
#include <SmurffCpp/DataMatrices/Data.h>
#include <SmurffCpp/IO/GenericIO.h>
#include <SmurffCpp/IO/MatrixIO.h>
#include <Utils/StringUtils.h>

#define NONE_TAG "none"

#define GLOBAL_SECTION_TAG "global"
#define TRAIN_SECTION_TAG "train"
#define TEST_SECTION_TAG "test"

#define NUM_PRIORS_TAG "num_priors"
#define PRIOR_PREFIX "prior"
#define NUM_AUX_DATA_TAG "num_aux_data"
#define AUX_DATA_PREFIX "aux_data"
#define SAVE_PREFIX_TAG "save_prefix"
#define SAVE_EXTENSION_TAG "save_extension"
#define SAVE_FREQ_TAG "save_freq"
#define SAVE_PRED_TAG "save_pred"
#define SAVE_MODEL_TAG "save_model"
#define CHECKPOINT_FREQ_TAG "checkpoint_freq"
#define VERBOSE_TAG "verbose"
#define BURNING_TAG "burnin"
#define NSAMPLES_TAG "nsamples"
#define NUM_LATENT_TAG "num_latent"
#define NUM_THREADS_TAG "num_threads"
#define RANDOM_SEED_SET_TAG "random_seed_set"
#define RANDOM_SEED_TAG "random_seed"
#define INIT_MODEL_TAG "init_model"
#define CLASSIFY_TAG "classify"
#define THRESHOLD_TAG "threshold"

#define POSTPROP_PREFIX "prop_posterior"
#define LAMBDA_TAG "Lambda"
#define MU_TAG "mu"

namespace smurff {

PriorTypes stringToPriorType(std::string name)
{
   if(name == PRIOR_NAME_DEFAULT)
      return PriorTypes::default_prior;
   else if(name == PRIOR_NAME_MACAU)
      return PriorTypes::macau;
   else if(name == PRIOR_NAME_MACAU_ONE)
      return PriorTypes::macauone;
   else if(name == PRIOR_NAME_SPIKE_AND_SLAB)
      return PriorTypes::spikeandslab;
   else if(name == PRIOR_NAME_NORMALONE)
      return PriorTypes::normalone;
   else if(name == PRIOR_NAME_NORMAL)
      return PriorTypes::normal;
   else
   {
      THROWERROR("Invalid prior type");
   }
}

std::string priorTypeToString(PriorTypes type)
{
   switch(type)
   {
      case PriorTypes::default_prior:
         return PRIOR_NAME_DEFAULT;
      case PriorTypes::macau:
         return PRIOR_NAME_MACAU;
      case PriorTypes::macauone:
         return PRIOR_NAME_MACAU_ONE;
      case PriorTypes::spikeandslab:
         return PRIOR_NAME_SPIKE_AND_SLAB;
      case PriorTypes::normal:
         return PRIOR_NAME_NORMAL;
      case PriorTypes::normalone:
         return PRIOR_NAME_NORMALONE;
      default:
      {
         THROWERROR("Invalid prior type");
      }
   }
}

ModelInitTypes stringToModelInitType(std::string name)
{
   if(name == MODEL_INIT_NAME_RANDOM)
      return ModelInitTypes::random;
   else if (name == MODEL_INIT_NAME_ZERO)
      return ModelInitTypes::zero;
   else
   {
      THROWERROR("Invalid model init type " + name);
   }
}

std::string modelInitTypeToString(ModelInitTypes type)
{
   switch(type)
   {
      case ModelInitTypes::random:
         return MODEL_INIT_NAME_RANDOM;
      case ModelInitTypes::zero:
         return MODEL_INIT_NAME_ZERO;
      default:
      {
         THROWERROR("Invalid model init type");
      }
   }
}

//config
ActionTypes Config::ACTION_DEFAULT_VALUE = ActionTypes::none;
int Config::BURNIN_DEFAULT_VALUE = 200;
int Config::NSAMPLES_DEFAULT_VALUE = 800;
int Config::NUM_LATENT_DEFAULT_VALUE = 96;
int Config::NUM_THREADS_DEFAULT_VALUE = 0; // as many as you want
ModelInitTypes Config::INIT_MODEL_DEFAULT_VALUE = ModelInitTypes::zero;
const char* Config::SAVE_PREFIX_DEFAULT_VALUE = "";
const char* Config::SAVE_EXTENSION_DEFAULT_VALUE = ".ddm";
int Config::SAVE_FREQ_DEFAULT_VALUE = 0;
bool Config::SAVE_PRED_DEFAULT_VALUE = true;
bool Config::SAVE_MODEL_DEFAULT_VALUE = true;
int Config::CHECKPOINT_FREQ_DEFAULT_VALUE = 0;
int Config::VERBOSE_DEFAULT_VALUE = 0;
const char* Config::STATUS_DEFAULT_VALUE = "";
bool Config::ENABLE_BETA_PRECISION_SAMPLING_DEFAULT_VALUE = true;
double Config::THRESHOLD_DEFAULT_VALUE = 0.0;
int Config::RANDOM_SEED_DEFAULT_VALUE = 0;

Config::Config()
{
   m_action = Config::ACTION_DEFAULT_VALUE;
   m_model_init_type = Config::INIT_MODEL_DEFAULT_VALUE;

   m_save_prefix = Config::SAVE_PREFIX_DEFAULT_VALUE;
   m_save_extension = Config::SAVE_EXTENSION_DEFAULT_VALUE;
   m_save_freq = Config::SAVE_FREQ_DEFAULT_VALUE;
   m_save_pred = Config::SAVE_PRED_DEFAULT_VALUE;
   m_save_model = Config::SAVE_MODEL_DEFAULT_VALUE;
   m_checkpoint_freq = Config::CHECKPOINT_FREQ_DEFAULT_VALUE;

   m_random_seed_set = false;
   m_random_seed = Config::RANDOM_SEED_DEFAULT_VALUE;

   m_verbose = Config::VERBOSE_DEFAULT_VALUE;
   m_burnin = Config::BURNIN_DEFAULT_VALUE;
   m_nsamples = Config::NSAMPLES_DEFAULT_VALUE;
   m_num_latent = Config::NUM_LATENT_DEFAULT_VALUE;
   m_num_threads = Config::NUM_THREADS_DEFAULT_VALUE;

   m_threshold = Config::THRESHOLD_DEFAULT_VALUE;
   m_classify = false;
}

std::string Config::getSavePrefix() const
{
    auto &pfx = m_save_prefix;
    if (pfx == Config::SAVE_PREFIX_DEFAULT_VALUE || pfx.empty())
    {
#ifdef _WINDOWS
       char templ[1024];
	   static int temp_counter = 0;
       snprintf(templ, 1023, "%s\\smurff.%03d", getenv("TEMP"), temp_counter++);
       CreateDirectory(templ, NULL);
       pfx = templ;
#else
        char templ[1024] = "/tmp/smurff.XXXXXX";
        pfx = mkdtemp(templ);
#endif
    }

    if (*pfx.rbegin() != '/') 
        pfx += "/";

    return m_save_prefix;
}

const std::shared_ptr<SideInfoConfig>& Config::getSideInfoConfig(int mode) const
{
  auto iter = m_sideInfoConfigs.find(mode);
  THROWERROR_ASSERT(iter != m_sideInfoConfigs.end());
  return iter->second;
}

Config& Config::addSideInfoConfig(int mode, std::shared_ptr<SideInfoConfig> c)
{
    m_sideInfoConfigs[mode] = c;

    // automagically update prior type 
    // normal(one) prior -> macau(one) prior
    if ((int)m_prior_types.size() > mode)
    {
      PriorTypes &pt = m_prior_types[mode];
           if (pt == PriorTypes::normal) pt = PriorTypes::macau;
      else if (pt == PriorTypes::normalone) pt = PriorTypes::macauone;
    }

    return *this;
}

bool Config::validate() const
{
   if (!m_train || !m_train->getNNZ())
   {
      THROWERROR("Missing train data");
   }

   auto train_pos = PVec<>(m_train->getNModes());
   if (!m_train->hasPos())
   {
       m_train->setPos(train_pos);
   }
   else if (m_train->getPos() != train_pos)
   {
       THROWERROR("Train should be at upper position (all zeros)");
   }

   if (m_test && !m_test->getNNZ())
   {
      THROWERROR("Missing test data");
   }

   if (m_test && m_test->getDims() != m_train->getDims())
   {
      THROWERROR("Train and test data should have the same dimensions");
   }

   if(getPriorTypes().size() != m_train->getNModes())
   {
      THROWERROR("Number of priors should equal to number of dimensions in train data");
   }

   if (m_train->getNModes() > 2)
   {

      if (!m_auxData.empty())
      {
         //it is advised to check macau and macauone priors implementation
         //as well as code in PriorFactory that creates macau priors

         //this check does not directly check that input data is Tensor (it only checks number of dimensions)
         //however TensorDataFactory will do an additional check throwing an exception
         THROWERROR("Aux data is not supported for TensorData");
      }
   }

   for (auto p : m_sideInfoConfigs)
   {
      int mode = p.first;
      auto &configItem = p.second;
      const auto &sideInfo = configItem->getSideInfo();
      THROWERROR_ASSERT(sideInfo);

      if (sideInfo->getDims()[0] != m_train->getDims()[mode])
      {
         std::stringstream ss;
         ss << "Side info should have the same number of rows as size of dimension " << mode << " in train data";
         THROWERROR(ss.str());
      }
   }

   for(auto& ad1 : getData())
   {
      if (!ad1->hasPos())
      {
         std::stringstream ss;
         ss << "Data \"" << ad1->info() << "\" is missing position info";
         THROWERROR(ss.str());
      }

      const auto& dim1 = ad1->getDims();
      const auto& pos1 = ad1->getPos();

      for(auto& ad2 : getData())
      {
         if (ad1 == ad2)
            continue;

         if (!ad2->hasPos())
         {
            std::stringstream ss;
            ss << "Data \"" << ad2->info() << "\" is missing position info";
            THROWERROR(ss.str());
         }

         const auto& dim2 = ad2->getDims();
         const auto& pos2 = ad2->getPos();

         if (pos1 == pos2)
         {
            std::stringstream ss;
            ss << "Data \"" << ad1->info() <<  "\" and \"" << ad2->info() << "\" at same position";
            THROWERROR(ss.str());
         }

         // if two data blocks are aligned in a certain dimension
         // this dimension should be equal size
         for (std::size_t i = 0; i < pos1.size(); ++i)
         {
            if (pos1.at(i) == pos2.at(i) && (dim1.at(i) != dim2.at(i)))
            {
               std::stringstream ss;
               ss << "Data \"" << ad1->info() << "\" and \"" << ad2->info() << "\" different in size in dimension " << i;
               THROWERROR(ss.str());
            }
         }
      }
   }

   for(std::size_t i = 0; i < m_prior_types.size(); i++)
   {
      PriorTypes pt = m_prior_types[i];
      switch (pt)
      {
         case PriorTypes::normal:
         case PriorTypes::normalone:
         case PriorTypes::spikeandslab:
         case PriorTypes::default_prior:
            THROWERROR_ASSERT_MSG(!hasSideInfo(i), priorTypeToString(pt) + " prior in dimension " + std::to_string(i) + " cannot have side info");
            break;
         case PriorTypes::macau:
         case PriorTypes::macauone:
            THROWERROR_ASSERT_MSG(hasSideInfo(i), priorTypeToString(pt) + " prior in dimension " + std::to_string(i) + " needs side info");
            break;
         default:
            THROWERROR("Unknown prior");
            break;
      }
   }

   std::set<std::string> save_extensions = { ".csv", ".ddm" };

   if (save_extensions.find(m_save_extension) == save_extensions.end())
   {
      THROWERROR("Unknown output extension: " + m_save_extension + " (expected \".csv\" or \".ddm\")");
   }

   m_train->getNoiseConfig().validate();

   // validate propagated posterior
   for(uint64_t i=0; i<getTrain()->getNModes(); ++i)
   {
       if (hasPropagatedPosterior(i))
       {
           THROWERROR_ASSERT_MSG(
               getMuPropagatedPosterior(i)->getNCol() == getTrain()->getDims().at(i),
               "mu of propagated posterior in mode " + std::to_string(i) + 
               " should have same number of columns as train in mode"
           );
           THROWERROR_ASSERT_MSG(
               getLambdaPropagatedPosterior(i)->getNCol() == getTrain()->getDims().at(i),
               "Lambda of propagated posterior in mode " + std::to_string(i) + 
               " should have same number of columns as train in mode"
           );
           THROWERROR_ASSERT_MSG(
               (int)getMuPropagatedPosterior(i)->getNRow() == getNumLatent(),
               "mu of propagated posterior in mode " + std::to_string(i) + 
               " should have num-latent rows"
           );
           THROWERROR_ASSERT_MSG(
               (int)getLambdaPropagatedPosterior(i)->getNRow() == getNumLatent() * getNumLatent(),
               "mu of propagated posterior in mode " + std::to_string(i) +
                   " should have num-latent^2 rows"
           );
       }
   }

   return true;
}

void Config::save(std::string fname) const
{
   INIFile ini;

   //write global options section
   auto &global_section = ini.addSection(GLOBAL_SECTION_TAG);

   //count data
   global_section.put(NUM_PRIORS_TAG, m_prior_types.size());
   global_section.put(NUM_AUX_DATA_TAG, m_auxData.size());

   //priors data
   for(const auto &pt : m_prior_types)
      global_section.add(PRIOR_PREFIX, priorTypeToString(pt));

   //save data
   global_section.put(SAVE_PREFIX_TAG, m_save_prefix);
   global_section.put(SAVE_EXTENSION_TAG, m_save_extension);
   global_section.put(SAVE_FREQ_TAG, m_save_freq);
   global_section.put(SAVE_PRED_TAG, m_save_pred);
   global_section.put(SAVE_MODEL_TAG, m_save_model);
   global_section.put(CHECKPOINT_FREQ_TAG, m_checkpoint_freq);

   //general data
   global_section.put(VERBOSE_TAG, m_verbose);
   global_section.put(BURNING_TAG, m_burnin);
   global_section.put(NSAMPLES_TAG, m_nsamples);
   global_section.put(NUM_LATENT_TAG, m_num_latent);
   global_section.put(NUM_THREADS_TAG, m_num_threads);
   global_section.put(RANDOM_SEED_SET_TAG, m_random_seed_set);
   global_section.put(RANDOM_SEED_TAG, m_random_seed);
   global_section.put(INIT_MODEL_TAG, modelInitTypeToString(m_model_init_type));

   //probit prior data
   global_section.put(CLASSIFY_TAG, m_classify);
   global_section.put(THRESHOLD_TAG, m_threshold);

   //write train data section
   TensorConfig::save_tensor_config(ini, TRAIN_SECTION_TAG, -1, m_train);

   //write test data section
   TensorConfig::save_tensor_config(ini, TEST_SECTION_TAG, -1, m_test);

   //write macau prior configs section
   for (auto p : m_sideInfoConfigs)
   {
       int mode = p.first;
       auto &configItem = p.second;
       configItem->save(ini, mode);
   }

   //write aux data section
   for (std::size_t sIndex = 0; sIndex < m_auxData.size(); sIndex++)
   {
      TensorConfig::save_tensor_config(ini, AUX_DATA_PREFIX, sIndex, m_auxData.at(sIndex));
   }

   //write posterior propagation
   for (std::size_t pIndex = 0; pIndex < m_prior_types.size(); pIndex++)
   {
       if (hasPropagatedPosterior(pIndex))
       {
           auto section = INIFile::add_index(POSTPROP_PREFIX, pIndex);
           ini.put(section, MU_TAG, getMuPropagatedPosterior(pIndex)->getFilename());
           ini.put(section, LAMBDA_TAG, getLambdaPropagatedPosterior(pIndex)->getFilename());
       }
   }

   ini.write(fname);
}

bool Config::restore(std::string fname)
{
   THROWERROR_FILE_NOT_EXIST(fname);

   INIFile reader;
   reader.read(fname);

   //restore train data
   setTest(TensorConfig::restore_tensor_config(reader, TEST_SECTION_TAG));

   //restore test data
   setTrain(TensorConfig::restore_tensor_config(reader, TRAIN_SECTION_TAG));

   //restore global data
   auto &global_section = reader.getSection(GLOBAL_SECTION_TAG);

   //restore priors
   std::size_t num_priors = global_section.get<int>(NUM_PRIORS_TAG, 0);
   std::vector<std::string> pNames;
   for(std::size_t pIndex = 0; pIndex < num_priors; pIndex++)
   {
      pNames.push_back(global_section.get<std::string>(addIndex(PRIOR_PREFIX, pIndex),  PRIOR_NAME_DEFAULT));
   }
   setPriorTypes(pNames);

   //restore macau prior configs section
   for (std::size_t mPriorIndex = 0; mPriorIndex < num_priors; mPriorIndex++)
   {
      auto sideInfoConfig = std::make_shared<SideInfoConfig>();
      if (sideInfoConfig->restore(reader, mPriorIndex))
         m_sideInfoConfigs[mPriorIndex] = sideInfoConfig;
   }

   //restore aux data
   std::size_t num_aux_data = global_section.get<int>(NUM_AUX_DATA_TAG, 0);
   for(std::size_t pIndex = 0; pIndex < num_aux_data; pIndex++)
   {
      m_auxData.push_back(TensorConfig::restore_tensor_config(reader, addIndex(AUX_DATA_PREFIX, pIndex)));
   }

   // restore posterior propagated data
   for(std::size_t pIndex = 0; pIndex < num_priors; pIndex++)
   {
       auto mu = std::shared_ptr<MatrixConfig>();
       auto lambda = std::shared_ptr<MatrixConfig>();

       {
           std::string filename = reader.get<std::string>(addIndex(POSTPROP_PREFIX, pIndex), MU_TAG, NONE_TAG);
           if (filename != NONE_TAG)
           {
               mu = matrix_io::read_matrix(filename, false);
               mu->setFilename(filename);
           }
       }

       {
           std::string filename = reader.get<std::string>(addIndex(POSTPROP_PREFIX, pIndex), LAMBDA_TAG, NONE_TAG);
           if (filename != NONE_TAG)
           {
               lambda = matrix_io::read_matrix(filename, false);
               lambda->setFilename(filename);
           }
       }

       if (mu && lambda)
       {
           addPropagatedPosterior(pIndex, mu, lambda);
       }
   }


   //restore save data
   m_save_prefix = global_section.get<std::string>(SAVE_PREFIX_TAG, Config::SAVE_PREFIX_DEFAULT_VALUE);
   m_save_extension = global_section.get<std::string>(SAVE_EXTENSION_TAG, Config::SAVE_EXTENSION_DEFAULT_VALUE);
   m_save_freq = global_section.get<int>(SAVE_FREQ_TAG, Config::SAVE_FREQ_DEFAULT_VALUE);
   m_save_pred = global_section.get<bool>(SAVE_PRED_TAG, Config::SAVE_PRED_DEFAULT_VALUE);
   m_save_model = global_section.get<bool>(SAVE_MODEL_TAG, Config::SAVE_MODEL_DEFAULT_VALUE);
   m_checkpoint_freq = global_section.get<int>(CHECKPOINT_FREQ_TAG, Config::CHECKPOINT_FREQ_DEFAULT_VALUE);

   //restore general data
   m_verbose = global_section.get<int>(VERBOSE_TAG, Config::VERBOSE_DEFAULT_VALUE);
   m_burnin = global_section.get<int>(BURNING_TAG, Config::BURNIN_DEFAULT_VALUE);
   m_nsamples = global_section.get<int>(NSAMPLES_TAG, Config::NSAMPLES_DEFAULT_VALUE);
   m_num_latent = global_section.get<int>(NUM_LATENT_TAG, Config::NUM_LATENT_DEFAULT_VALUE);
   m_num_threads = global_section.get<int>(NUM_THREADS_TAG, Config::NUM_THREADS_DEFAULT_VALUE);
   m_random_seed_set = global_section.get<bool>(RANDOM_SEED_SET_TAG,  false);
   m_random_seed = global_section.get<int>(RANDOM_SEED_TAG, Config::RANDOM_SEED_DEFAULT_VALUE);
   m_model_init_type = stringToModelInitType(global_section.get<std::string>(INIT_MODEL_TAG, modelInitTypeToString(Config::INIT_MODEL_DEFAULT_VALUE)));

   //restore probit prior data
   m_classify = global_section.get<bool>(CLASSIFY_TAG,  false);
   m_threshold = global_section.get<double>(THRESHOLD_TAG, Config::THRESHOLD_DEFAULT_VALUE);

   return true;
}

std::ostream& Config::info(std::ostream &os, std::string indent) const
{
   os << indent << "  Iterations: " << getBurnin() << " burnin + " << getNSamples() << " samples\n";

   if (getSaveFreq() != 0 || getCheckpointFreq() != 0)
   {
      if (getSaveFreq() > 0)
      {
          os << indent << "  Save model: every " << getSaveFreq() << " iteration\n";
      }
      else if (getSaveFreq() < 0)
      {
          os << indent << "  Save model after last iteration\n";
      }

      if (getCheckpointFreq() > 0)
      {
          os << indent << "  Checkpoint state: every " << getCheckpointFreq() << " seconds\n";
      }

      os << indent << "  Save prefix: " << getSavePrefix() << "\n";
      os << indent << "  Save extension: " << getSaveExtension() << "\n";
   }
   else
   {
      os << indent << "  Save model: never\n";
   }

   return os;
}
} // end namespace smurff

#include <iostream>
#include <fstream>
#include <memory>

#include <SmurffCpp/Utils/HDF5Group.h>
#include <SmurffCpp/Utils/StringUtils.h>

#include "SideInfoConfig.h"

namespace smurff {

static const std::string SIDE_INFO_PREFIX = "side_info";
static const std::string TOL_TAG = "tol";
static const std::string DIRECT_TAG = "direct";
static const std::string THROW_ON_CHOLESKY_ERROR_TAG = "throw_on_cholesky_error";
static const std::string NUMBER_TAG = "nr";

const bool   SideInfoConfig::DIRECT_DEFAULT_VALUE = true;
const double SideInfoConfig::BETA_PRECISION_DEFAULT_VALUE = 10.0;
const double SideInfoConfig::TOL_DEFAULT_VALUE = 1e-6;

SideInfoConfig::SideInfoConfig(const Matrix &data, const NoiseConfig &ncfg)
   : DataConfig(data, ncfg)
{
   m_tol = SideInfoConfig::TOL_DEFAULT_VALUE;
   m_direct = SideInfoConfig::DIRECT_DEFAULT_VALUE;
   m_throw_on_cholesky_error = false;
}

SideInfoConfig::SideInfoConfig(const SparseMatrix &data, const NoiseConfig &ncfg)
   : DataConfig(data, false, ncfg)
{
   m_tol = SideInfoConfig::TOL_DEFAULT_VALUE;
   m_direct = SideInfoConfig::DIRECT_DEFAULT_VALUE;
   m_throw_on_cholesky_error = false;
}

void SideInfoConfig::save(HDF5Group& cfg_file, std::size_t prior_index) const
{
   std::string sectionName = addIndex(SIDE_INFO_PREFIX, prior_index);

   //macau config params
   cfg_file.put(sectionName, TOL_TAG, m_tol);
   cfg_file.put(sectionName, DIRECT_TAG, m_direct);
   cfg_file.put(sectionName, THROW_ON_CHOLESKY_ERROR_TAG, m_throw_on_cholesky_error);

   //data
   DataConfig::save(cfg_file, sectionName);
}

bool SideInfoConfig::restore(const HDF5Group& cfg_file, std::size_t prior_index)
{
   std::string sectionName = addIndex(SIDE_INFO_PREFIX, prior_index);

   if (!cfg_file.hasSection(sectionName))
   {
       return false;
   }

   //restore side info properties
   m_tol = cfg_file.get(sectionName, TOL_TAG, SideInfoConfig::TOL_DEFAULT_VALUE);
   m_direct = cfg_file.get(sectionName, DIRECT_TAG, false);
   m_throw_on_cholesky_error = cfg_file.get(sectionName, THROW_ON_CHOLESKY_ERROR_TAG, false);

   DataConfig::restore(cfg_file, sectionName);

   return true;
}

} // end namespace smurff

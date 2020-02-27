#include "DataConfig.h"

#include <numeric>

#include <SmurffCpp/Utils/PVec.hpp>
#include <SmurffCpp/Utils/ConfigFile.h>
#include <Utils/Error.h>
#include <Utils/StringUtils.h>

namespace smurff {

static const std::string POS_TAG = "pos";
static const std::string FILE_TAG = "file";
static const std::string DATA_TAG = "data";
static const std::string DENSE_TAG = "dense";
static const std::string SCARCE_TAG = "scarce";
static const std::string SPARSE_TAG = "sparse";
static const std::string TYPE_TAG = "type";

static const std::string NONE_VALUE("none");

static const std::string NOISE_MODEL_TAG = "noise_model";
static const std::string PRECISION_TAG = "precision";
static const std::string SN_INIT_TAG = "sn_init";
static const std::string SN_MAX_TAG = "sn_max";
static const std::string NOISE_THRESHOLD_TAG = "noise_threshold";

DataConfig::DataConfig ( const Matrix &m
                       , const NoiseConfig& noiseConfig
                       , PVec<> pos
                       )
   : m_noiseConfig(noiseConfig)
   , m_isDense(true)
   , m_isScarce(false)
   , m_isMatrix(true)
   , m_dims({ (std::uint64_t)m.rows(), (std::uint64_t)m.cols() })
   , m_nnz(m.nonZeros())
   , m_pos(pos)
   , m_dense_matrix_data(m)
{
   check();
}

DataConfig::DataConfig ( const SparseMatrix &m
                       , bool isScarce
                       , const NoiseConfig& noiseConfig
                       , PVec<> pos
                       )
   : m_noiseConfig(noiseConfig)
   , m_isDense(false)
   , m_isScarce(isScarce)
   , m_isMatrix(true)
   , m_dims({ (std::uint64_t)m.rows(), (std::uint64_t)m.cols() })
   , m_nnz(m.nonZeros())
   , m_pos(pos)
   , m_sparse_matrix_data(m)
{
   check();
}

DataConfig::DataConfig ( const DenseTensor &m
                       , const NoiseConfig& noiseConfig
                       , PVec<> pos
                       )
   : m_noiseConfig(noiseConfig)
   , m_isDense(true)
   , m_isMatrix(false)
   , m_dims(m.getDims())
   , m_nnz(m.getNNZ())
   , m_pos(pos)
   , m_dense_tensor_data(m)
{
   check();
}

DataConfig::DataConfig ( const SparseTensor &m
                       , bool isScarce
                       , const NoiseConfig& noiseConfig
                       , PVec<> pos
                       )
   : m_noiseConfig(noiseConfig)
   , m_isDense(false)
   , m_isScarce(isScarce)
   , m_isMatrix(false)
   , m_dims(m.getDims())
   , m_nnz(m.getNNZ())
   , m_pos(pos)
   , m_sparse_tensor_data(m)
{
    check();
}


DataConfig::~DataConfig()
{
}

void DataConfig::check() const
{
   THROWERROR_ASSERT(m_dims.size() > 0);

   if (isDense())
   {
       THROWERROR_ASSERT(m_nnz == std::accumulate(m_dims.begin(), m_dims.end(), 1ULL, std::multiplies<std::uint64_t>()));
   }
}

//
// other methods
//

const Matrix &DataConfig::getDenseMatrixData() const
{
   THROWERROR_ASSERT(isDense() && isMatrix());
   return m_dense_matrix_data;
}

const SparseMatrix &DataConfig::getSparseMatrixData() const
{
   THROWERROR_ASSERT(!isDense() && isMatrix());
   return m_sparse_matrix_data;
}

const SparseTensor &DataConfig::getSparseTensorData() const
{
   THROWERROR_ASSERT(!isDense() && !isMatrix());
   return m_sparse_tensor_data;
}

const DenseTensor &DataConfig::getDenseTensorData() const
{
   THROWERROR_ASSERT(isDense() && !isMatrix());
   return m_dense_tensor_data;
}

Matrix &DataConfig::getDenseMatrixData()
{
   THROWERROR_ASSERT(!hasData());
   return m_dense_matrix_data;
}

SparseMatrix &DataConfig::getSparseMatrixData()
{
   THROWERROR_ASSERT(!hasData());
   return m_sparse_matrix_data;
}

SparseTensor &DataConfig::getSparseTensorData() 
{
   THROWERROR_ASSERT(!hasData());
   return m_sparse_tensor_data;
}

DenseTensor &DataConfig::getDenseTensorData() 
{
   THROWERROR_ASSERT(!hasData());
   return m_dense_tensor_data;
}

bool DataConfig::isDense() const
{
   return m_isDense;
}

bool DataConfig::isScarce() const
{
   return m_isScarce;
}

bool DataConfig::isMatrix() const
{
   return m_isMatrix;
}

std::uint64_t DataConfig::getNNZ() const
{
   return m_nnz;
}

std::uint64_t DataConfig::getNModes() const
{
   return m_dims.size();
}

const std::vector<std::uint64_t>& DataConfig::getDims() const
{
   return m_dims;
}

const NoiseConfig& DataConfig::getNoiseConfig() const
{
   return m_noiseConfig;
}

void DataConfig::setNoiseConfig(const NoiseConfig& value)
{
   m_noiseConfig = value;
}

void DataConfig::setFilename(const std::string &f)
{
    m_filename = f;
}

const std::string &DataConfig::getFilename() const
{
    return m_filename;
}

void DataConfig::setPos(const PVec<>& p)
{
   m_pos = p;
}

bool DataConfig::hasPos() const
{
    return m_pos.size();
}

const PVec<>& DataConfig::getPos() const
{
   THROWERROR_ASSERT(hasPos());
   return m_pos;
}

std::ostream& DataConfig::info(std::ostream& os) const
{
   if (!m_dims.size())
   {
      os << "0";
   }
   else
   {
      os << m_dims.operator[](0);
      for (std::size_t i = 1; i < m_dims.size(); i++)
         os << " x " << m_dims.operator[](i);
   }
   if (getFilename().size())
   {
        os << " \"" << getFilename() << "\"";
   }
   if (hasPos())
   {
        os << " @[" << getPos() << "]";
   }
   return os;
}

std::string DataConfig::info() const
{
    std::stringstream ss;
    info(ss);
    return ss.str();
}

void DataConfig::save(ConfigFile& cfg_file, const std::string& sectionName) const
{
   //write tensor config position
   if (hasPos())
   {
      std::stringstream ss;
      ss << getPos();
      cfg_file.put(sectionName, POS_TAG, ss.str());
   }

   if (isMatrix() && isDense()) 
      cfg_file.write(sectionName, DATA_TAG, getDenseMatrixData());
   else if (isMatrix() && !isDense()) 
      cfg_file.write(sectionName, DATA_TAG, getSparseMatrixData());
   else if (!isMatrix() && !isDense())
      cfg_file.write(sectionName, DATA_TAG, getSparseTensorData());
   else if (!isMatrix() && isDense())
      cfg_file.write(sectionName, DATA_TAG, getDenseTensorData());
   else 
      THROWERROR_NOTIMPL();

   //write tensor config type
   std::string type_str = isDense() ? DENSE_TAG : isScarce() ? SCARCE_TAG : SPARSE_TAG;
   cfg_file.put(sectionName, TYPE_TAG, type_str);

   //write noise config
   auto &noise_config = getNoiseConfig();
   if (noise_config.getNoiseType() != NoiseTypes::unset)
   {
      cfg_file.put(sectionName, NOISE_MODEL_TAG, noiseTypeToString(noise_config.getNoiseType()));
      cfg_file.put(sectionName, PRECISION_TAG, noise_config.getPrecision());
      cfg_file.put(sectionName, SN_INIT_TAG, noise_config.getSnInit());
      cfg_file.put(sectionName, SN_MAX_TAG, noise_config.getSnMax());
      cfg_file.put(sectionName, NOISE_THRESHOLD_TAG, noise_config.getThreshold());
   }

}

bool DataConfig::restore(const ConfigFile& cfg_file, const std::string& sectionName)
{
   //restore position
   std::string pos_str = cfg_file.get(sectionName, POS_TAG, NONE_VALUE);
   if (pos_str != NONE_VALUE)
   {
      std::vector<int> tokens;
      split(pos_str, tokens, ',');

      //assign position
      setPos(PVec<>(tokens));
   }

   //restore type
   m_isDense = cfg_file.get(sectionName, TYPE_TAG, DENSE_TAG) == DENSE_TAG;
   m_isScarce = cfg_file.get(sectionName, TYPE_TAG, SCARCE_TAG) == SCARCE_TAG;

   //restore filename and content
   std::string filename = cfg_file.get(sectionName, FILE_TAG, NONE_VALUE);

   if (isMatrix() && isDense())
      cfg_file.read(sectionName, DATA_TAG, getDenseMatrixData());
   else if (isMatrix() && !isDense())
      cfg_file.read(sectionName, DATA_TAG, getSparseMatrixData());
   else if (!isMatrix() && !isDense())
      cfg_file.read(sectionName, DATA_TAG, getSparseTensorData());
   else if (!isMatrix() && isDense())
      cfg_file.read(sectionName, DATA_TAG, getDenseTensorData());
   else
      THROWERROR_NOTIMPL();

   //restore noise model
   NoiseConfig noise;

   NoiseTypes noiseType = stringToNoiseType(cfg_file.get(sectionName, NOISE_MODEL_TAG, noiseTypeToString(NoiseTypes::unset)));
   if (noiseType != NoiseTypes::unset)
   {
      noise.setNoiseType(noiseType);
      noise.setPrecision(cfg_file.get(sectionName, PRECISION_TAG, NoiseConfig::PRECISION_DEFAULT_VALUE));
      noise.setSnInit(cfg_file.get(sectionName, SN_INIT_TAG, NoiseConfig::ADAPTIVE_SN_INIT_DEFAULT_VALUE));
      noise.setSnMax(cfg_file.get(sectionName, SN_MAX_TAG, NoiseConfig::ADAPTIVE_SN_MAX_DEFAULT_VALUE));
      noise.setThreshold(cfg_file.get(sectionName, NOISE_THRESHOLD_TAG, NoiseConfig::PROBIT_DEFAULT_VALUE));
   }

   //assign noise model
   setNoiseConfig(noise);

   return true;
}

} // end namespace smurff

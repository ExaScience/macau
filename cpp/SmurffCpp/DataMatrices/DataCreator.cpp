#include "DataCreator.h"

#include "DataCreatorBase.h"

#include <SmurffCpp/DataMatrices/MatricesData.h>

//noise classes
#include <SmurffCpp/Configs/NoiseConfig.h>
#include <SmurffCpp/Noises/NoiseFactory.h>

#include <Utils/Error.h>
#include <SmurffCpp/Utils/PVec.hpp>

namespace smurff {

std::shared_ptr<Data> DataCreator::create(std::shared_ptr<const MatrixConfig> mc) const
{
   auto& aux_matrices = getSession().getConfig().getAuxData();

   //create creator
   std::shared_ptr<DataCreatorBase> creatorBase = std::make_shared<DataCreatorBase>();

   //create single matrix
   if (aux_matrices.empty())
      return mc->create(creatorBase);

   //multiple matrices
   NoiseConfig ncfg(NoiseTypes::unused);
   std::shared_ptr<MatricesData> local_data_ptr(new MatricesData());
   local_data_ptr->setNoiseModel(NoiseFactory::create_noise_model(ncfg));
   local_data_ptr->add(PVec<>({0,0}), mc->create(creatorBase));

   for(auto &m : aux_matrices)
   {
      local_data_ptr->add(m->getPos(), m->create(creatorBase));
   }

   return local_data_ptr;
}

std::shared_ptr<Data> DataCreator::create(std::shared_ptr<const TensorConfig> tc) const
{

   //we need TensorsData class to utilize aux data
   const auto& auxDataSet = getSession().getConfig().getAuxData();
   if (!auxDataSet.empty())
   {
      THROWERROR("Tensor config does not support aux data");
   }

   //create creator
   std::shared_ptr<DataCreatorBase> creatorBase = std::make_shared<DataCreatorBase>();

   return tc->create(creatorBase);
}

std::shared_ptr<Data> DataCreator::create(std::shared_ptr<const DataConfig> dc) const
{
   auto& aux_matrices = getSession().getConfig().getAuxData();

   //create creator
   std::shared_ptr<DataCreatorBase> creatorBase = std::make_shared<DataCreatorBase>();

   //create single matrix
   if (aux_matrices.empty())
      return dc->create(creatorBase);

   if (dc->isMatrix())
   {
      //multiple matrices
      NoiseConfig ncfg(NoiseTypes::unused);
      std::shared_ptr<MatricesData> local_data_ptr(new MatricesData());
      local_data_ptr->setNoiseModel(NoiseFactory::create_noise_model(ncfg));
      local_data_ptr->add(PVec<>({0,0}), dc->create(creatorBase));

      for(auto &m : aux_matrices)
      {
         local_data_ptr->add(m->getPos(), m->create(creatorBase));
      }

      return local_data_ptr;
   }
   else
   {
      THROWERROR("Tensor config does not support aux data");
   }

   return std::shared_ptr<Data>();
   
}
} // end namespace smurff

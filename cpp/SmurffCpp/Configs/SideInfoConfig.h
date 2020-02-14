#pragma once

#include <vector>
#include <memory>
#include <string>

#include "MatrixConfig.h"

class INIFile;

namespace smurff
{
   class SideInfoConfig
   {
   public:
      static double BETA_PRECISION_DEFAULT_VALUE;
      static double TOL_DEFAULT_VALUE;
   private:
      double m_tol;
      bool m_direct;
      bool m_throw_on_cholesky_error;

      std::shared_ptr<MatrixConfig> m_sideInfo; //side info matrix for macau and macauone prior

   public:
      SideInfoConfig();

   public:
      std::shared_ptr<MatrixConfig> getSideInfo() const
      {
         return m_sideInfo;
      }

      void setSideInfo(std::shared_ptr<MatrixConfig> value)
      {
         m_sideInfo = value;
      }

      double getTol() const
      {
         return m_tol;
      }

      void setTol(double value)
      {
         m_tol = value;
      }

      bool getDirect() const
      {
         return m_direct;
      }

      void setDirect(bool value)
      {
         m_direct = value;
      }

      bool getThrowOnCholeskyError() const
      {
         return m_throw_on_cholesky_error;
      }

      void setThrowOnCholeskyError(bool value)
      {
         m_throw_on_cholesky_error = value;
      }

   public:
      void save(INIFile& writer, std::size_t prior_index, std::size_t config_item_index) const;

      bool restore(const INIFile& reader, std::size_t prior_index, std::size_t config_item_index);
   };
}

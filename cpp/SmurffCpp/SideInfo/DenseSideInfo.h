#pragma once

#include <memory>

#include <SmurffCpp/Types.h>

#include <SmurffCpp/Configs/MatrixConfig.h>

#include "ISideInfo.h"


namespace smurff {

   class DenseSideInfo : public ISideInfo
   {
   private:
      std::shared_ptr<Matrix> m_side_info;

   public:
      DenseSideInfo(const std::shared_ptr<MatrixConfig> &);

   public:
      int cols() const override;

      int rows() const override;

   public:
      std::ostream& print(std::ostream &os) const override;

      bool is_dense() const override;

   public:
      //linop

      void compute_uhat(Matrix& uhat, Matrix& beta) override;

      void At_mul_A(Matrix& out) override;

      Matrix A_mul_B(Matrix& A) override;

      int solve_blockcg(Matrix& X, double reg, Matrix& B, double tol, const int blocksize, const int excess, bool throw_on_cholesky_error = false) override;

      Vector col_square_sum() override;

      void At_mul_Bt(Vector& Y, const int col, Matrix& B) override;

      void add_Acol_mul_bt(Matrix& Z, const int col, Vector& b) override;

      //only for tests
   public:
      std::shared_ptr<Matrix> get_features();
   };

}

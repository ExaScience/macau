#pragma once

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Types.h>
#include <Eigen/IterativeLinearSolvers>

#include <SmurffCpp/Utils/MatrixUtils.h>
#include <Utils/Error.h>
#include <Utils/counters.h>

#include <SmurffCpp/SideInfo/SparseSideInfo.h>

namespace smurff {
namespace linop {
  class AtA;
} }

namespace Eigen {
namespace internal {
  // AtA looks-like a SparseMatrix, so let's inherits its traits:
  template<>
  struct traits<smurff::linop::AtA> :  public Eigen::internal::traits<smurff::SparseMatrix>
  {};
}
}

namespace smurff
{
namespace linop
{

// Example of a matrix-free wrapper from a user type to Eigen's compatible type
// For the sake of simplicity, this example simply wrap a Eigen::SparseMatrix.
class AtA : public Eigen::EigenBase<AtA>
{
public:
  // Required typedefs, constants, and method:
  typedef double Scalar;
  typedef double RealScalar;
  typedef int StorageIndex;
  enum
  {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };
  Index rows() const { return m_A.cols(); }
  Index cols() const { return m_A.cols(); }
  template <typename Rhs>
  Eigen::Product<AtA, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs> &x) const
  {
    return Eigen::Product<AtA, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
  }
  // Custom API:
  AtA(const SparseMatrix &A, double reg) : m_A(A), m_reg(reg) {}

  const SparseMatrix &m_A;
  double m_reg;
};

} // namespace linop
} // namespace smurff

// Implementation of AtA * Eigen::DenseVector though a specialization of internal::generic_product_impl:
namespace Eigen {
namespace internal {
  template<typename Rhs>
  struct generic_product_impl<smurff::linop::AtA, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
  : generic_product_impl_base<smurff::linop::AtA,Rhs,generic_product_impl<smurff::linop::AtA,Rhs> >
  {
    typedef typename Product<smurff::linop::AtA,Rhs>::Scalar Scalar;
    template<typename Dest>
    static void scaleAndAddTo(Dest& dst, const smurff::linop::AtA& lhs, const Rhs& rhs, const Scalar& alpha)
    {
      // This method should implement "dst += alpha * lhs * rhs" inplace,
      dst += alpha * ((lhs.m_A.transpose() * (lhs.m_A * rhs)) + lhs.m_reg * rhs);
    }
  };
}
}

namespace smurff
{
namespace linop
{

template<typename T>
int  solve_blockcg(Matrix & X, T & t, double reg, Matrix & B, double tol, const int blocksize, const int excess, bool throw_on_cholesky_error = false);
template<typename T>
int  solve_blockcg(Matrix & X, T & t, double reg, Matrix & B, double tol, bool throw_on_cholesky_error = false);

inline void AtA_mul_B(Matrix & out, SparseSideInfo & A, double reg, Matrix & B);
inline void AtA_mul_B(Matrix & out, Matrix & A, double reg, Matrix & B);

inline void makeSymmetric(Matrix &A)
{
  A = A.selfadjointView<Eigen::Lower>();
}

/** good values for solve_blockcg are blocksize=32 an excess=8 */
template<typename T>
inline int solve_blockcg(Matrix & X, T & K, double reg, Matrix & B, double tol, const int blocksize, const int excess, bool throw_on_cholesky_error) {
  if (B.rows() <= excess + blocksize) {
    return solve_blockcg(X, K, reg, B, tol, throw_on_cholesky_error);
  }
  // split B into blocks of size <blocksize> (+ excess if needed)
  Matrix Xblock, Bblock;
  int max_iter = 0;
  for (int i = 0; i < B.rows(); i += blocksize) {
    int nrows = blocksize;
    if (i + blocksize + excess >= B.rows()) {
      nrows = B.rows() - i;
    }
    Bblock.resize(nrows, B.cols());
    Xblock.resize(nrows, X.cols());

    Bblock = B.block(i, 0, nrows, B.cols());
    int niter = solve_blockcg(Xblock, K, reg, Bblock, tol, throw_on_cholesky_error);
    max_iter = std::max(niter, max_iter);
    X.block(i, 0, nrows, X.cols()) = Xblock;
  }

  return max_iter;
}

//
//-- Solves the system (K' * K + reg * I) * X = B for X for m right-hand sides
//   K = d x n matrix
//   I = n x n identity
//   X = n x m matrix
//   B = n x m matrix
//
template<typename T>
inline int solve_blockcg(Matrix & X, T & K, double reg, Matrix & B, double tol, bool throw_on_cholesky_error) {
  // initialize
  const int nfeat = B.cols();
  const int nrhs  = B.rows();
  double tolsq = tol*tol;

  if (nfeat != K.cols()) {THROWERROR("B.cols() must equal K.cols()");}

  Vector norms(nrhs), inorms(nrhs); 
  norms.setZero();
  inorms.setZero();
  #pragma omp parallel for schedule(static)
  for (int rhs = 0; rhs < nrhs; rhs++) 
  {
    double sumsq = 0.0;
    for (int feat = 0; feat < nfeat; feat++) 
    {
      sumsq += B(rhs, feat) * B(rhs, feat);
    }
    norms(rhs)  = std::sqrt(sumsq);
    inorms(rhs) = 1.0 / norms(rhs);
  }
  Matrix R(nrhs, nfeat);
  Matrix P(nrhs, nfeat);
  Matrix Ptmp(nrhs, nfeat);
  X.setZero();
  // normalize R and P:
  #pragma omp parallel for schedule(static) collapse(2)
  for (int feat = 0; feat < nfeat; feat++) 
  {
    for (int rhs = 0; rhs < nrhs; rhs++) 
    {
      R(rhs, feat) = B(rhs, feat) * inorms(rhs);
      P(rhs, feat) = R(rhs, feat);
    }
  }
  Matrix* RtR = new Matrix(nrhs, nrhs);
  Matrix* RtR2 = new Matrix(nrhs, nrhs);

  Matrix KP(nrhs, nfeat);
  Matrix PtKP(nrhs, nrhs);
  //Eigen::Matrix<double, N, N> A;
  //Eigen::Matrix<double, N, N> Psi;
  Matrix A;
  Matrix Psi;

  //A_mul_At_combo(*RtR, R);
  *RtR = R * R.transpose();
  makeSymmetric(*RtR);

  const int nblocks = (int)ceil(nfeat / 64.0);

  // CG iteration:
  int iter = 0;
  for (iter = 0; iter < 1000; iter++) {
    // KP = K * P
    ////double t1 = tick();
    AtA_mul_B(KP, K, reg, P);
    ////double t2 = tick();

    PtKP = P * KP.transpose();

    auto chol_PtKP = PtKP.llt();
    THROWERROR_ASSERT_MSG(!throw_on_cholesky_error || chol_PtKP.info() != Eigen::NumericalIssue, "Cholesky Decomposition failed! (Numerical Issue)");
    THROWERROR_ASSERT_MSG(!throw_on_cholesky_error || chol_PtKP.info() != Eigen::InvalidInput, "Cholesky Decomposition failed! (Invalid Input)");
    A = chol_PtKP.solve(*RtR);

    A.transposeInPlace();
    ////double t3 = tick();

    
    #pragma omp parallel for schedule(guided)
    for (int block = 0; block < nblocks; block++) 
    {
      int col = block * 64;
      int bcols = std::min(64, nfeat - col);
      // X += A' * P
      X.block(0, col, nrhs, bcols).noalias() += A *  P.block(0, col, nrhs, bcols);
      // R -= A' * KP
      R.block(0, col, nrhs, bcols).noalias() -= A * KP.block(0, col, nrhs, bcols);
    }
    ////double t4 = tick();

    // convergence check:
    //A_mul_At_combo(*RtR2, R);
    *RtR2 = R * R.transpose();
    makeSymmetric(*RtR2);

    Vector d = RtR2->diagonal();
    //std::cout << "[ iter " << iter << "] " << std::scientific << d.transpose() << " (max: " << d.maxCoeff() << " > " << tolsq << ")" << std::endl;
    //std::cout << iter << ":" << std::scientific << d.transpose() << std::endl;
    if ( (d.array() < tolsq).all()) {
      break;
    } 

    // Psi = (R R') \ R2 R2'
    auto chol_RtR = RtR->llt();
    THROWERROR_ASSERT_MSG(!throw_on_cholesky_error || chol_RtR.info() != Eigen::NumericalIssue, "Cholesky Decomposition failed! (Numerical Issue)");
    THROWERROR_ASSERT_MSG(!throw_on_cholesky_error || chol_RtR.info() != Eigen::InvalidInput, "Cholesky Decomposition failed! (Invalid Input)");
    Psi  = chol_RtR.solve(*RtR2);
    Psi.transposeInPlace();
    ////double t5 = tick();

    // P = R + Psi' * P (P and R are already transposed)
    #pragma omp parallel for schedule(guided)
    for (int block = 0; block < nblocks; block++) 
    {
      int col = block * 64;
      int bcols = std::min(64, nfeat - col);
      Matrix xtmp(nrhs, bcols);
      xtmp = Psi *  P.block(0, col, nrhs, bcols);
      P.block(0, col, nrhs, bcols) = R.block(0, col, nrhs, bcols) + xtmp;
    }

    // R R' = R2 R2'
    std::swap(RtR, RtR2);
    ////double t6 = tick();
    ////printf("t2-t1 = %.3f, t3-t2 = %.3f, t4-t3 = %.3f, t5-t4 = %.3f, t6-t5 = %.3f\n", t2-t1, t3-t2, t4-t3, t5-t4, t6-t5);
  }
  
  if (iter == 1000)
  {
    Vector d = RtR2->diagonal().cwiseSqrt();
    std::cerr << "warning: block_cg: could not find a solution in 1000 iterations; residual: ["
              << d.transpose() << " ].all() > " << tol << std::endl;
  }


  // unnormalizing X:
  #pragma omp parallel for schedule(static) collapse(2)
  for (int feat = 0; feat < nfeat; feat++) 
  {
    for (int rhs = 0; rhs < nrhs; rhs++) 
    {
      X(rhs, feat) *= norms(rhs);
    }
  }
  delete RtR;
  delete RtR2;
  return iter;
}

inline void AtA_mul_B(Matrix & out, Matrix & A, double reg, Matrix & B) {
	out.noalias() = (A.transpose() * (A * B.transpose())).transpose() + reg * B;
}

inline void AtA_mul_B(Matrix& out, SparseSideInfo& A, double reg, Matrix& B) {
  out.noalias() = (A.Ft * (A.F * B.transpose())).transpose() + reg * B;
}

}}
#include "catch.hpp"

#include <SmurffCpp/Types.h>
#include <SmurffCpp/Types.h>

#include <SmurffCpp/IO/TensorIO.h>
#include <SmurffCpp/Utils/TensorUtils.h>
#include <SmurffCpp/Utils/MatrixUtils.h>

namespace smurff {

static NoiseConfig fixed_ncfg(NoiseTypes::fixed);

TEST_CASE("tensor_io/read_dense_float64_bin | tensor_io/write_dense_float64_bin")
{
   std::vector<std::uint64_t> tensorConfigDims = { 3, 4 };
   std::vector<double> tensorConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   std::shared_ptr<TensorConfig> tensorConfig(new TensorConfig(tensorConfigDims, tensorConfigValues, fixed_ncfg));

   std::stringstream matrixConfigStream;
   tensor_io::write_dense_float64_bin(matrixConfigStream, tensorConfig);

   std::shared_ptr<TensorConfig> actualTensorConfig = tensor_io::read_dense_float64_bin(matrixConfigStream);

   Matrix actualMatrix = matrix_utils::dense_to_eigen(*actualTensorConfig);

   Matrix expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("tensor_io/read_sparse_float64_bin | tensor_io/write_sparse_float64_bin")
{
   std::vector<std::uint64_t> tensorConfigDims = { 3, 4 };
   std::vector<std::vector<std::uint32_t>> tensorConfigColumns = { 
      { 0, 0, 0, 0, 2, 2, 2, 2 }, 
      { 0, 1, 2, 3, 0, 1, 2, 3 } 
   };
   std::vector<double> tensorConfigValues = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<TensorConfig> tensorConfig(new TensorConfig(tensorConfigDims, tensorConfigColumns, tensorConfigValues, fixed_ncfg, false));

   std::stringstream tensorStream;
   tensor_io::write_sparse_float64_bin(tensorStream, tensorConfig);

   std::shared_ptr<TensorConfig> actualTensorConfig = tensor_io::read_sparse_float64_bin(tensorStream, false);
   SparseMatrix actualMatrix = matrix_utils::sparse_to_eigen(*actualTensorConfig);

   SparseMatrix expectedMatrix(3, 4);
   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets;
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 1, 2));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 2, 3));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 3, 4));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 0, 9));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 1, 10));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 2, 11));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 3, 12));
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("tensor_io/read_sparse_binary_bin | tensor_io/write_sparse_binary_bin")
{
   std::vector<std::uint64_t> tensorConfigDims = { 3, 4 };
   std::vector<std::vector<std::uint32_t>> tensorConfigColumns = {
      { 0, 0, 0, 0, 2, 2, 2, 2 },
      { 0, 1, 2, 3, 0, 1, 2, 3 }
   };
   std::shared_ptr<TensorConfig> tensorConfig(new TensorConfig(tensorConfigDims, tensorConfigColumns, fixed_ncfg, false));

   std::stringstream tensorStream;
   tensor_io::write_sparse_binary_bin(tensorStream, tensorConfig);

   std::shared_ptr<TensorConfig> actualTensorConfig = tensor_io::read_sparse_binary_bin(tensorStream, false);
   SparseMatrix actualMatrix = matrix_utils::sparse_to_eigen(*actualTensorConfig);

   SparseMatrix expectedMatrix(3, 4);
   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets;
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 1, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 2, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 3, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 1, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 2, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 3, 1));
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("tensor_io/read_dense_float64_csv | tensor_io/write_dense_float64_csv")
{
   std::vector<std::uint64_t> tensorConfigDims = { 3, 4 };
   std::vector<double> tensorConfigValues = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
   std::shared_ptr<TensorConfig> tensorConfig(new TensorConfig(tensorConfigDims, tensorConfigValues, fixed_ncfg));

   std::stringstream matrixConfigStream;
   tensor_io::write_dense_float64_csv(matrixConfigStream, tensorConfig);

   std::shared_ptr<TensorConfig> actualTensorConfig = tensor_io::read_dense_float64_csv(matrixConfigStream);
   Matrix actualMatrix = matrix_utils::dense_to_eigen(*actualTensorConfig);

   Matrix expectedMatrix(3, 4);
   expectedMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}

TEST_CASE("tensor_io/read_sparse_float64_tns | tensor_io/write_sparse_float64_tns")
{
   std::vector<std::uint64_t> tensorConfigDims = { 3, 4 };
   std::vector<std::vector<std::uint32_t>> tensorConfigColumns = {
       { 0, 0, 0, 0, 2, 2, 2, 2 },
       { 0, 1, 2, 3, 0, 1, 2, 3 }
   };
   std::vector<double> tensorConfigValues = { 1, 2, 3, 4, 9, 10, 11, 12 };
   std::shared_ptr<TensorConfig> tensorConfig(new TensorConfig(tensorConfigDims, tensorConfigColumns, tensorConfigValues, fixed_ncfg, false));

   std::stringstream tensorStream;
   tensor_io::write_sparse_float64_tns(tensorStream, tensorConfig);

   std::shared_ptr<TensorConfig> actualTensorConfig = tensor_io::read_sparse_float64_tns(tensorStream, false);
   SparseMatrix actualMatrix = matrix_utils::sparse_to_eigen(*actualTensorConfig);

   SparseMatrix expectedMatrix(3, 4);
   std::vector<Eigen::Triplet<double> > expectedMatrixTriplets;
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 0, 1));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 1, 2));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 2, 3));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(0, 3, 4));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 0, 9));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 1, 10));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 2, 11));
   expectedMatrixTriplets.push_back(Eigen::Triplet<double>(2, 3, 12));
   expectedMatrix.setFromTriplets(expectedMatrixTriplets.begin(), expectedMatrixTriplets.end());

   REQUIRE(matrix_utils::equals(actualMatrix, expectedMatrix));
}
} // end namespace smurff

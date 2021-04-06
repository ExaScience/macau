#include <cstdio>
#include <fstream>

#include <boost/filesystem/operations.hpp>

#include "catch.hpp"

#include <SmurffCpp/Types.h>

#include <SmurffCpp/Configs/Config.h>
#include <SmurffCpp/Sessions/TrainSession.h>
#include <SmurffCpp/Predict/PredictSession.h>
#include <SmurffCpp/Utils/MatrixUtils.h>
#include <SmurffCpp/Utils/StateFile.h>
#include <SmurffCpp/result.h>

#include "Tests.h"

#define TAG_MATRIX_TESTS "[matrix][random]"

namespace fs = boost::filesystem;

namespace smurff {
namespace test {

static Config& prepareResultDir(Config &config, const std::string &dir)
{
  fs::path output_dir("tests_output");
  fs::create_directory(output_dir);

  std::string save_dir(dir);
  save_dir.erase(std::remove_if(save_dir.begin(), save_dir.end(),
                           [](char c) {
                              const std::string special_chars("\\/:*\"<>|");
                              return special_chars.find(c) != std::string::npos;
                           }
            ), save_dir.end());
  
  output_dir /= fs::path(save_dir);
  fs::path output_filename = output_dir / "output.h5";
 
  config.setSaveFreq(1);
  config.setSaveName(output_filename.string());
  fs::remove_all(output_dir);
  fs::create_directory(output_dir);
  return config;
}

TEST_CASE("OverwriteHDF5")
{
  Config config = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal});
  prepareResultDir(config, Catch::getResultCapture().getCurrentTestName() + "_train");

  // this one should work just fine
  TrainSession(config).run();

  // the second run should complain -- trying to create th HDF5 file that already exists
  TrainSession(config).run();
}

TEST_CASE("PredictSession/BPMF")
{
  Config config = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::normal, PriorTypes::normal});
  prepareResultDir(config, Catch::getResultCapture().getCurrentTestName() + "_train");
  std::string model_file = config.getSaveName();

  std::shared_ptr<ISession> trainSession = std::make_shared<TrainSession>(config);
  trainSession->run();

  // std::cout << "Prediction from TrainSession RMSE: " << trainSession->getRmseAvg() <<
  // std::endl;

  {
    prepareResultDir(config, Catch::getResultCapture().getCurrentTestName() + "_predict");
    const Config &const_config = config;
    PredictSession s(model_file); // restore from file just generated by TrainSession

    // test predict from TensorConfig
    auto result = s.predict(const_config.getTest());

    // test predict using smurff::predict function
    Model m;
    for (int step=0; step < s.getNumSteps(); step++)
    {
        s.restoreModel(m, step);
        auto preds =  predict_matrix(const_config.getTest().getSparseMatrixData(), { m.U(0), m.U(1)});
        for ( const auto &result_item : result->m_predictions)
        {
            double p1 = result_item.pred_all.at(step);
            double p2 = preds.coeff(result_item.coords.at(0), result_item.coords.at(1));
            checkValue(p1, p2, rmse_epsilon);
        }
    }

    // std::cout << "Prediction from StateFile RMSE: " << result->rmse_avg <<
    // std::endl;
    checkValue(trainSession->getRmseAvg(), result->rmse_avg, rmse_epsilon);
  }
}


TEST_CASE("TrainSession/TensorBPMF")
{
  Config config = genConfig(trainSparseTensor3d, testSparseTensor3d, {PriorTypes::normal, PriorTypes::normal, PriorTypes::normal});
  prepareResultDir(config, Catch::getResultCapture().getCurrentTestName() + "_train");
  std::string model_file = config.getSaveName();

  std::shared_ptr<ISession> trainSession = std::make_shared<TrainSession>(config);
  trainSession->run();

  // std::cout << "Prediction from TrainSession RMSE: " << trainSession->getRmseAvg() <<
  // std::endl;

  {
    prepareResultDir(config, Catch::getResultCapture().getCurrentTestName() + "_predict");
    PredictSession s(model_file); // restore from file just generated by TrainSession

    // test predict from TensorConfig
    auto result = s.predict(config.getTest());

    // std::cout << "Prediction from StateFile RMSE: " << result->rmse_avg <<
    // std::endl;
    checkValue(trainSession->getRmseAvg(), result->rmse_avg, rmse_epsilon);
  }
}

//=================================================================

TEST_CASE("PredictSession/Features/1", TAG_MATRIX_TESTS) {
  const SideInfoConfig rowSideInfoDenseMatrixConfig = makeSideInfoConfig(rowSideDenseMatrix);

  Config config = genConfig(trainDenseMatrix, testSparseMatrix, {PriorTypes::macau, PriorTypes::normal});
  config.addSideInfo(0, rowSideInfoDenseMatrixConfig);
  prepareResultDir(config, Catch::getResultCapture().getCurrentTestName());
  std::string model_file = config.getSaveName();

  std::shared_ptr<ISession> trainSession = std::make_shared<TrainSession>(config);
  trainSession->run();

  PredictSession predict_session(model_file);

  const auto &sideInfoMatrix = rowSideInfoDenseMatrixConfig.getDenseMatrixData();

#if 0
    std::cout << "sideInfo =\n" << sideInfoMatrix << std::endl;
#endif

  for (int r = 0; r < sideInfoMatrix.rows(); r++) {
#if 0
        std::cout << "=== row " << r << " ===\n";
#endif

    auto predictions = predict_session.predict(0, sideInfoMatrix.row(r));
#if 0
        int i = 0;
        for (auto P : predictions)
        {
            std::cout << "p[" << i++ << "] = " << P->transpose() << std::endl;
        }
#endif
  }
}

TEST_CASE("PredictSession/Features/2", TAG_MATRIX_TESTS) {
  /*
       BetaPrecision: 1.00
  U = np.array([ [ 1, 2, -1, -2  ] ])
  V = np.array([ [ 2, 2, 1, 2 ] ])
  U*V =
    [[ 2,  2,  1,  2],
     [ 4,  4,  2,  4],
     [-2, -2, -1, -2],
     [-4, -4, -2, -4]])
  */

  std::vector<std::uint32_t> trainMatrixConfigRows = {0, 0, 1, 1, 2, 2};
  std::vector<std::uint32_t> trainMatrixConfigCols = {0, 1, 2, 3, 0, 1};
  std::vector<double> trainMatrixConfigVals = {2, 2, 2, 4, -2, -2};
  SparseMatrix trainMatrix = matrix_utils::sparse_to_eigen(SparseTensor(
      { 4, 4 }, { trainMatrixConfigRows, trainMatrixConfigCols }, trainMatrixConfigVals));

  std::vector<std::uint32_t> testMatrixConfigRows = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  std::vector<std::uint32_t> testMatrixConfigCols = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  std::vector<double> testMatrixConfigVals = {2, 2, 1, 2, 4, 4, 2, 4, -2, -2, -1, -2, -4, -4, -2, -4};
  SparseMatrix testMatrix = matrix_utils::sparse_to_eigen(SparseTensor({ 4, 4 } , { testMatrixConfigRows, testMatrixConfigCols }, testMatrixConfigVals));

  SideInfoConfig rowSideInfoConfig;
  {
    NoiseConfig nc(NoiseTypes::sampled);
    nc.setPrecision(10.0);

    std::vector<std::uint32_t> rowSideInfoSparseMatrixConfigRows = {0, 1, 2, 3};
    std::vector<std::uint32_t> rowSideInfoSparseMatrixConfigCols = {0, 0, 0, 0};
    std::vector<double> rowSideInfoSparseMatrixConfigVals = {2, 4, -2, -4};

    auto data = SparseTensor({ 4, 1 }, { rowSideInfoSparseMatrixConfigRows, rowSideInfoSparseMatrixConfigCols },
                                   rowSideInfoSparseMatrixConfigVals);

    rowSideInfoConfig.setData(matrix_utils::sparse_to_eigen(data), false);
    rowSideInfoConfig.setNoiseConfig(nc);
    rowSideInfoConfig.setDirect(true);
  }
  const SideInfoConfig &si = rowSideInfoConfig;

  Config config = genConfig(trainMatrix, testMatrix, {PriorTypes::macau, PriorTypes::normal});
  config.addSideInfo(0, si);
  NoiseConfig trainNoise(NoiseTypes::fixed);
  trainNoise.setPrecision(1.);
  config.getTrain().setNoiseConfig(trainNoise);
  prepareResultDir(config, Catch::getResultCapture().getCurrentTestName());
  std::string model_file = config.getSaveName();

  std::shared_ptr<ISession> trainSession = std::make_shared<TrainSession>(config);
  trainSession->run();

  PredictSession predict_session_in(model_file);
  auto in_matrix_predictions = predict_session_in.predict(config.getTest())->m_predictions;

  PredictSession predict_session_out(model_file);
  const auto &sideInfoMatrix = si.getSparseMatrixData();
  int d = config.getTrain().getDims()[0];
  for (int r = 0; r < d; r++) {
    auto feat = sideInfoMatrix.row(r).transpose();
    auto out_of_matrix_predictions = predict_session_out.predict(0, feat);
    // Vector out_of_matrix_averages =
    // out_of_matrix_predictions->rowwise().mean();

#undef DEBUG_OOM_PREDICT
#ifdef DEBUG_OOM_PREDICT
    for (auto p : in_matrix_predictions) {
      if (p.coords[0] == r) {
        std::cout << "in: " << p << std::endl;
        std::cout << "  out: " << out_of_matrix_averages.row(p.coords[1]) << std::endl;
      }
    }
#endif
  }
}

} // namespace test
} // namespace smurff

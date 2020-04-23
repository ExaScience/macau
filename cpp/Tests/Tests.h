#include <vector>

#include <SmurffCpp/Types.h>

namespace smurff {

struct ResultItem;
class DataConfig;

namespace test {

// noise
extern smurff::NoiseConfig fixed_ncfg;

// dense train data
extern smurff::Matrix trainDenseMatrix;
extern smurff::DenseTensor trainDenseTensor2d;
extern smurff::DenseTensor trainDenseTensor3d;

// sparse train data
extern smurff::SparseMatrix trainSparseMatrix;
extern smurff::SparseTensor trainSparseTensor2d;

// sparse test data
extern smurff::SparseMatrix testSparseMatrix;
extern smurff::SparseTensor testSparseTensor2d;
extern smurff::SparseTensor testSparseTensor3d;

// aux data
extern smurff::DataConfig rowAuxDense;
extern smurff::DataConfig colAuxDense;

// side info
extern smurff::Matrix rowSideDenseMatrix;
extern smurff::Matrix colSideDenseMatrix;
extern smurff::Matrix rowSideDenseMatrix3d;

extern smurff::SparseMatrix rowSideSparseMatrix;
extern smurff::SparseMatrix colSideSparseMatrix;

void REQUIRE_RESULT_ITEMS(const std::vector<smurff::ResultItem> &actualResultItems,
                          const std::vector<smurff::ResultItem> &expectedResultItems);

template<class M>
SideInfoConfig makeSideInfoConfig(const M &data, bool direct = true) {
  smurff::NoiseConfig sampled_ncfg(NoiseTypes::sampled);
  sampled_ncfg.setPrecision(10.0);
  SideInfoConfig picfg(data, sampled_ncfg);
  picfg.setDirect(direct);
  return picfg;
}

template <class Train, class Test> Config genConfig(const Train &train, const Test &test, std::vector<PriorTypes> priors) {
  Config config;
  config.setPriorTypes(priors);
  config.setBurnin(100);
  config.setNSamples(200);
  config.setVerbose(1);
  config.setRandomSeed(0xdeadbeef);
  config.setNumThreads(1);
  config.setNumLatent(4);

  config.getTrain().setData(train);
  config.getTrain().setNoiseConfig(fixed_ncfg);
  config.getTest().setData(test);

  return config;
}
} // namespace test
} // namespace smurff

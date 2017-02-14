#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iomanip>

#include "mvnormal.h"
#include "macau.h"
#include "chol.h"
#include "linop.h"
#include "noisemodels.h"
#include "latentprior.h"

extern "C" {
  #include <sparse.h>
}

using namespace std; 
using namespace Eigen;

namespace Macau {

ILatentPrior::ILatentPrior(BaseSession &m, int p, std::string name)
    : macau(m), pos(p), U(m.model->U(pos)), V(m.model->V(pos)),
      name(name), rrs(VectorNd::Zero(m.model->num_latent)),
                  MMs(MatrixNNd::Zero(m.model->num_latent, m.model->num_latent))
{
} 

// utility
Factors &ILatentPrior::model() const
{
    return *macau.model; 
}

int ILatentPrior::num_latent() const
{ 
    return model().num_latent; 
}

INoiseModel &ILatentPrior::noise() const
{
    return *macau.noise; 
}
std::ostream &ILatentPrior::printInitStatus(std::ostream &os, std::string indent) 
{
    os << indent << pos << ": " << name << "\n";
    return os;
}


/**
 *  base class NormalPrior 
 */

NormalPrior::NormalPrior(BaseSession &m, int p, std::string name)
    : ILatentPrior(m, p, name) 
{
    const int K = num_latent();
    mu.resize(K);
    mu.setZero();

    Lambda.resize(K, K);
    Lambda.setIdentity();
    Lambda *= 10;

    // parameters of Inv-Whishart distribution
    WI.resize(K, K);
    WI.setIdentity();
    mu0.resize(K);
    mu0.setZero();
    b0 = 2;
    df = K;
}

void NormalPrior::sample_latents() {

    // FIXME: include siblings!!!!
    tie(mu, Lambda) = CondNormalWishart(U, mu0, b0, WI, df);

    ILatentPrior::sample_latents();
}

void NormalPrior::addSibling(BaseSession &b)
{
    addSiblingTempl<NormalPrior>(b);
}


void NormalPrior::sample_latent(int n)
{
    const auto &mu_u = getMu(n);
    const double alpha = noise().getAlpha();

    VectorNd &rr = rrs.local();
    MatrixNNd &MM = MMs.local();

    rr.setZero();
    MM.setZero();

    // add pnm
    pnm(n,rr,MM);

    // add noise
    rr.array() *= alpha;
    MM.array() *= alpha;

    // add hyperparams
    rr.noalias() += Lambda * mu_u;
    MM.noalias() += Lambda;

    Eigen::LLT<MatrixXd> chol = MM.llt();
    if(chol.info() != Eigen::Success) {
        throw std::runtime_error("Cholesky Decomposition failed!");
    }

    chol.matrixL().solveInPlace(rr);
    rr.noalias() += nrandn(num_latent());
    chol.matrixU().solveInPlace(rr);
    U.col(n).noalias() = rr;
}

void NormalPrior::savePriorInfo(std::string prefix) {
  writeToCSVfile(prefix + "-" + std::to_string(pos) + "-latentmean.csv", mu);
}


/*
 * Master Prior
 */

template<class Prior>
MasterPrior<Prior>::MasterPrior(BaseSession &m, int p) 
    : Prior(m, p)
{
    this->name = "Master" + this->name;
}

template<class Prior>
void MasterPrior<Prior>::init() 
{
    for(auto &s : slaves) s.init();
}

template<class Prior>
std::ostream &MasterPrior<Prior>::printInitStatus(std::ostream &os, std::string indent) 
{
    Prior::printInitStatus(os, indent);
    os << indent << "with slaves {\n";
    for(auto &s : slaves) s.printInitStatus(os, indent + "  ");
    os << indent << "}\n";
    return os;

}

template<typename P1, typename P2>
std::pair<P1, P2> &operator+=(std::pair<P1, P2> &a, const std::pair<P1, P2> &b) {
    a.first += b.first;
    a.second += b.second;
    return a;
}

template<class Prior>
void MasterPrior<Prior>::pnm(int n, VectorNd &rr, MatrixNNd &MM) 
{
    // first the master
    Prior::pnm(n, rr, MM);

    // then the slaves
    assert(slaves.size() > 0 && "No slaves");
    for(auto &s : slaves) {
        auto &slave_prior = s.priors.at(this->pos);
        slave_prior->pnm(n, rr, MM);
    }
}

template<class Prior>
void MasterPrior<Prior>::sample_latents() {
    assert(slaves.size() > 0 && "No slaves");
    for(auto &s : this->slaves) s.step();
    Prior::sample_latents();
}

template<class Prior>
template<class Model>
Model& MasterPrior<Prior>::addSlave()
{
    slaves.push_back(BaseSession());
    auto &slave_macau = slaves.back();
    slave_macau.name = "Slave " + std::to_string(slaves.size());
    slave_macau.setPrecision(1.0); // FIXME
    Model *n = new Model(this->num_latent());
    slave_macau.model.reset(n);
    for(auto &p : this->macau.priors) {
        if (p->pos == this->pos) slave_macau.template addPrior<SlavePrior>();
        else p->addSibling(slave_macau);
    }
    return *n;
}

template<class Prior>
double MasterPrior<Prior>::getLinkNorm() {
    assert(slaves.size() > 0 && "No slaves");
    double ret = .0;
    for(auto &s : this->slaves) ret += s.model->V(this->pos).norm();
    return ret;
}

template class MasterPrior<NormalPrior>;
template DenseDenseMF &MasterPrior<NormalPrior>::addSlave();
template SparseDenseMF &MasterPrior<NormalPrior>::addSlave();
template SparseMF &MasterPrior<NormalPrior>::addSlave();

template class MasterPrior<SpikeAndSlabPrior>;
template DenseDenseMF &MasterPrior<SpikeAndSlabPrior>::addSlave();
template SparseDenseMF &MasterPrior<SpikeAndSlabPrior>::addSlave();
template SparseMF &MasterPrior<SpikeAndSlabPrior>::addSlave();

} // end namespace Macau

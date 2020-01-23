#include <mpi.h>

#include "MPISession.h"
#include "MPIPriorFactory.h"

#include <SmurffCpp/Priors/ILatentPrior.h>

#include <SmurffCpp/Utils/Error.h>

namespace smurff {

MPISession::MPISession()
{
   name = "MPISession";

   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
}

void MPISession::run()
{
   if (world_rank == 0)
   {
      Session::run();
   }
   else
   {
      bool work_done = false;

      for(auto &p : m_priors)
         work_done |= p->run_slave();

      THROWERROR_ASSERT(work_done);
   }
}

std::shared_ptr<IPriorFactory> MPISession::create_prior_factory() const
{
   return std::make_shared<MPIPriorFactory>();
}

//create mpi session
//parses args with setFromArgs, then internally calls setFromConfig (to validate, save, set config)
std::shared_ptr<ISession> create_mpi_session(int argc, char** argv)
{
   std::shared_ptr<MPISession> session(new MPISession());
   session->fromConfig(parse_options(argc, argv));
   return session;
}
} // end namespace smurff

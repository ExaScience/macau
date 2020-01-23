#include "UnusedNoise.h"

#include <Utils/Error.h>

namespace smurff {

UnusedNoise::UnusedNoise()
: INoiseModel()
{
}

double UnusedNoise::getAlpha() const 
{
   THROWERROR_NOTIMPL();
}

double UnusedNoise::sample(const SubModel& model, const PVec<> &pos, double val)
{
   THROWERROR_NOTIMPL();
}

std::ostream& UnusedNoise::info(std::ostream& os, std::string indent)
{
   os << "Noisemodel is not used here." << std::endl;
   return os;
}

std::string UnusedNoise::getStatus()
{
   return std::string("Unused");
}
} // end namespace smurff

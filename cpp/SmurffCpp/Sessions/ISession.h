#pragma once

#include <vector>
#include <memory>

#include <SmurffCpp/StatusItem.h>
#include <SmurffCpp/ResultItem.h>
#include <SmurffCpp/Configs/Config.h>

namespace smurff {
   class StateFile;
   class Result;

   class ISession
   {
   protected:
      ISession(){};

   public:
      virtual ~ISession(){}

   public:
      virtual void run() = 0;
      virtual bool step() = 0;
      virtual bool interrupted() { return false; }
      virtual void init() = 0;

      const Config &getConfig() const { return m_config; }
      virtual StatusItem getStatus() const = 0;
      virtual const Result &getResult() const = 0;

      double getRmseAvg() { return getStatus().rmse_avg; }
      const std::vector<ResultItem> & getResultItems() const;

    public:
      virtual std::ostream &info(std::ostream &, std::string indent) const = 0;
      std::string infoAsString() 
      {
          std::ostringstream ss;
          info(ss, "");
          return ss.str();
      }

   protected:
      Config m_config;
   };

}

#pragma once

#include <memory>

#include "IDataCreator.h"

#include <SmurffCpp/Sessions/Session.h>

namespace smurff
{
   class DataCreator : public IDataCreator
   {
   private:
      std::weak_ptr<Session> m_session;

      Session &getSession() const { return *m_session.lock(); }

   public:
      DataCreator(std::shared_ptr<Session> session)
         : m_session(session)
      {
      }

   public:
      std::shared_ptr<Data> create(std::shared_ptr<const DataConfig> dc) const override;
   };
}

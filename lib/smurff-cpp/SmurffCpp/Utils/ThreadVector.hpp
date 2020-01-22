#pragma once

#include <algorithm>
#include <numeric>
#include <vector>
#include <cassert>

#include "omp_util.h"

namespace smurff
{
   template<typename T>
   class thread_vector
   {
       public:
           thread_vector(const T &t = T())
           {
               init(t);
           }
           template<typename F>
           T combine(F f) const {
               return std::accumulate(_m.begin(), _m.end(), _i, f);
           }
           T combine() const {
               return std::accumulate(_m.begin(), _m.end(), _i, std::plus<T>());
           }
   
           T &local() {
               return _m.at(threads::get_thread_num());
           }
           void reset() {
               _m.resize(threads::get_max_threads());
               for(auto &t: _m) t = _i;
           }
           template<typename F>
           T combine_and_reset(F f) const {
               T ret = combine(f);
               reset();
               return ret;
           }
           T combine_and_reset() {
               T ret = combine();
               reset();
               return ret;
           }
           void init(const T &t) {
               _i = t;
               reset();
           }
           void init(const std::vector<T> &v) {
               assert((int)v.size() == threads::get_max_threads());
               _m = v;
           }

           typedef typename std::vector<T>::const_iterator const_iterator;

           const_iterator begin() const
           {
               return _m.begin();
           }

           const_iterator end() const
           {
               return _m.end();
           }

       private:
           std::vector<T> _m;
           T _i;
   };
}

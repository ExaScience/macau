from libcpp cimport bool
from libcpp.memory cimport shared_ptr

from MatrixConfig cimport MatrixConfig

cdef extern from "<SmurffCpp/Configs/SideInfoConfig.h>" namespace "smurff":
    cdef cppclass SideInfoConfig:
        MacauPriorConfigItem() except +
        void setTol(double value)
        void setDirect(bool value)

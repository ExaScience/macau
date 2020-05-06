#include "omp_util.h"

#include <iostream>

#include <SmurffCpp/Utils/Error.h>

#if defined(USE_ARRAYFIRE)
#include "arrayfire.h"
#elif defined(VIENNACL_WITH_OPENCL)
#include "viennacl/ocl/backend.hpp"
#endif

namespace threads
{


    // _OPENMP will be enabled if -fopenmp flag is passed to the compiler (use cmake release build)
    #if defined(_OPENMP)

    #include <omp.h>

    static int  m_verbose = 0;

    int get_num_threads()
    {
        return omp_get_num_threads(); 
    }

    int get_max_threads()
    {
        return omp_get_max_threads();
    }

    int get_thread_num()
    {
        return omp_get_thread_num(); 
    }


    void init(int verbose, int num_threads) 
    {
        m_verbose = verbose;
        static int default_num_threads = get_max_threads();

        if (num_threads > 0)
        {
            omp_set_num_threads(num_threads);
        }
        else
        {
            omp_set_num_threads(default_num_threads);
        }

        if (verbose)
        {
            std::cout << "Using OpenMP with up to " << get_max_threads() << " threads.\n";
        }
    }

    #else

    void init(int verbose, int) 
    { 
        if (verbose)
        {
            std::cout << "No threading library used.\n";
        }

    }

    int  get_num_threads() { return 1; }
    int  get_max_threads() { return 1; }
    int  get_thread_num() { return 0; } 

    #endif // _OPENMP
} //end namespace threads

namespace opencl
{
    static int m_device;

    void init(int verbose, int device_idx)
    {
#if defined(USE_ARRAYFIRE)
            af::setDevice(device_idx);
#elif defined(VIENNACL_WITH_OPENCL)
            const std::vector<viennacl::ocl::device> devices = viennacl::ocl::platform().devices();
            viennacl::ocl::setup_context(0, devices[device_idx]);
            viennacl::ocl::switch_context(0);
            if (verbose) {
                std::cout << "Using OpenCL Device:\n" << devices[device_idx].info() << std::endl;
            }
#endif

        m_device = device_idx;


        if (verbose)
        {
#if defined(USE_ARRAYFIRE)
            af::info();
#elif defined(VIENNACL_WITH_OPENCL)
            const std::vector<viennacl::ocl::device> devices = viennacl::ocl::platform().devices();
            std::cout << "Using OpenCL Device:\n"
                      << devices[m_device].info() << std::endl;
#endif
        }
    }
} // namespace opencl
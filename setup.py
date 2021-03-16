from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

import os
import sys
import subprocess
import platform

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: C++",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS"
]

# taken from github.com/pybind/cmake_example
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='', extra_cmake_args='', extra_build_args=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.extra_cmake_args = extra_cmake_args.split()
        self.extra_build_args = extra_build_args.split()

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = 'Debug' if self.debug else 'Release'

        os.makedirs(self.build_temp, exist_ok = True)
        cmake_args = [
                        '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                        '-DPYTHON_EXECUTABLE=' + sys.executable,
                        '-DENABLE_PYTHON=ON',
                        ]
        build_args = ['--config', cfg, '--parallel']
        if "CPU_COUNT" in os.environ:
            build_args.append(os.environ.get("CPU_COUNT"))

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

        cmake_args += ext.extra_cmake_args 
        build_args += ['--'] + ext.extra_build_args
      
        cmake_srcdir = ext.sourcedir

        if not skip_cmake:
            subprocess.check_call(['cmake', cmake_srcdir] + cmake_args, cwd=self.build_temp)

        subprocess.check_call(['cmake', '--build', '.' ] + build_args, cwd=self.build_temp)

        if install_binaries:
            subprocess.check_call(['cmake', '--build', '.', '--target', 'install'] + build_args, cwd=self.build_temp)

extra_cmake_args = ''
extra_build_args = ''
install_binaries = False
skip_cmake = False

if "--skip-cmake" in sys.argv:
    skip_cmake = True
    sys.argv.remove("--skip-cmake")

if "--extra-cmake-args" in sys.argv:
    index = sys.argv.index('--extra-cmake-args')
    sys.argv.pop(index)
    extra_cmake_args += " " + sys.argv.pop(index)

if "CMAKE_ARGS" in os.environ:
    extra_cmake_args += " " + os.environ.get("CMAKE_ARGS")

if "--extra-build-args" in sys.argv:
    index = sys.argv.index('--extra-build-args')
    sys.argv.pop(index)
    extra_build_args += " " + sys.argv.pop(index)

if "BUILD_ARGS" in os.environ:
    extra_build_args += " " + os.environ.get("BUILD_ARGS")

try:
    index = sys.argv.index('--install-binaries')
    sys.argv.pop(index)
    install_binaries = True
except ValueError:
    pass

setup(
    name = 'smurff',
    package_dir={'smurff':'python/smurff'},
    packages = [ 'smurff' ],
    scripts = [ 'python/py_smurff' ],
    use_scm_version={
        'version_scheme': 'post-release',
        'local_scheme': 'dirty-tag'
    },
    url = "http://github.com/ExaScience/smurff",
    ext_modules=[CMakeExtension('smurff/wrapper', '.', extra_cmake_args, extra_build_args)],
    cmdclass=dict(build_ext=CMakeBuild), 
    zip_safe = False,
    license = "MIT",
    description = 'Bayesian Factorization Methods',
    long_description = """
        Highly optimized and parallelized methods for
        Bayesian Factorization, including BPMF and smurff. The package uses
        optimized OpenMP/C++ code with a pybind11gq$ wrapper to factorize large
        scale matrices. smurff method provides also the ability to incorporate
        high-dimensional side information to the factorization.
    """,
    author = "Tom Vander Aa",
    author_email = "Tom.VanderAa@imec.be",
    classifiers = CLASSIFIERS,
    keywords = "bayesian factorization machine-learning high-dimensional side-information",
    install_requires = [ 'numpy', 'scipy', 'pandas', 'scikit-learn', 'h5sparse-tensor' ],
    setup_requires=['setuptools_scm', 'pybind11' ],
    tests_require=['parameterized' ],
)


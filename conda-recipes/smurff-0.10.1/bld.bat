@echo off

mkdir build
cd build

set CMAKE_CONFIG="Release"

cmake ..\lib\smurff-cpp\cmake -LAH -G"%GENERATOR%"                       ^
    -DCMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%"

cmake --build . --config %CMAKE_CONFIG% --target ALL_BUILD
cmake --build . --config %CMAKE_CONFIG% --target INSTALL
if errorlevel 1 exit 1


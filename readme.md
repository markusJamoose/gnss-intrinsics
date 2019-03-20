## Description

An open source implementation of a Code Division Multiple Access (CDMA) software correlator library that leverages Single Instruction Multiple Data (SIMD) is presented.  We initially discuss the key aspects involved in the correlation operation for software radio applications. Afterward, we present the state of the art Application Programming Interface (API) that provides SIMD capable methods for each of the components in a correlation operation, including the first of its kind parallelized code and carrier generation using lookup tables and SIMD instructions. The library is developed using SIMD Intrinsic Instructions, which are a C type nomenclature offering access to the assembly instructions originally designed for the SIMD extensions in the processor. This design paradigm presents an advantage in terms of readability and simplified code development to accommodate future modifications. Recorded data was used with a standalone Global Navigation Satellite System (GNSS) software receiver where the methods hereby presented were tested and profiled to validate theoretical assumptions.

## Dependencies

**Raw signal file:** The GPS_and_GIOVE_A-NN-fs16_3676-if4_1304.bin raw signal file can be found here:
1. [https://drive.google.com/file/d/1zrM0wjSQgH8hpJgw6_xE-E2yr8mNTEkn/view?usp=sharing], or
2. [https://drive.google.com/open?id=1Jz07sTIdCOC557pjuQ5ODp1BuQ8h0-O8]

## Source Code

The source code available is organized as follows:
1. **src**: contains the SIMD library functions including multiplication, accumulation, and code and carrier generation with AVX2 and AVX512 support.
2. **doc**: contains the Doxygen style documentation of available functions in HTML format, with links to function parameters and return values if applicable. In order to generate the documentation user needs to run the `make doc` command.
3. **prof**: contains sample code on how to use the library functions in a simulated, standalone tracking scenario.
4. **plot**: contains binary files generated by `prof` source code to display tracking results values and validate new function additions or usage.
5. **build**: contains build directory for application using standard makefile components.
6. **data**: contains data dependencies needed to run the application
7. **install**: contains generated executables from `prof` source code and utility scripts to profile the code.

## Build code

For compilation the code uses traditional `makefile` syntaxes and nomenclature.
```
# Change directory to build
$ cd build/

# Compiles all executables available, this includes documentation
$ make

# clean generated executables
$ make clean
```

## Profile code
In Linux platforms, to enable profiling, obtain root privileges and run
```
# Obtain root priviledges and run
$root@root-pc:~# echo 0 > /proc/sys/kernel/yama/ptrace_scope
```
This process is optional and allows user to verify working implementations of the code. Profiling has been automatized by running the `install/profile_cpu_times.py` script. Note that this implementation requires a working station with Intel VTune Amplifier installed on it. To profile, simply run:
```
# Change directory to install directory. Assumes executables present here
$ cd install/

# Configure script and run applications
$ python profile_cpu_times.py
```

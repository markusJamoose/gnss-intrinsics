#!/bin/python
#
# This scripts runs a vtune amplifier program x amount of times and finds the
# mean, std, max, and min of the CPU Times found
#
# Created by Jake Johnson on March 10, 2018
# Last Modified March 10, 2018
#
# This program does not use advanced-hotspots.  When using advanced-hotspots, the
# CPU time increases up to the elapsed time for some reason.  But advanced-hotspots
# does give the CPU cycle count
#
# Make sure to run this in a directory where multiple vtune results can be stored
#


import subprocess
import numpy
import re

CPU_Times = []

f= open("guru99.txt","w+")
exe = ["avx_si32_avx2", "nom_si32_avx2", "avx_fl32_avx2", "nom_fl32_avx2", "avx_si32_avx512", "nom_si32_avx512", "avx_fl32_avx512", "nom_fl32_avx512", "reg_standalone"]
for j in range(0,9):
    print("Running: ")
    print(exe[j])
    print("\n")
    # Reset CPU Times to store only latest results
    CPU_Times = []
    for i in range(0,50):
	base_command = """/opt/intel/vtune_amplifier_2018.1.0.535340/bin64/amplxe-cl -collect hotspots -quiet -knob sampling-interval=1 -app-working-dir /home/gnssi9/dami7269/gnss-intrinsics/prof/ -- /home/gnssi9/dami7269/gnss-intrinsics/prof/"""
	command = base_command + exe[j]

        output = subprocess.check_output(command, shell=True)

        # Go through output add desired values to array
        for line in output.splitlines():
            if re.search('CPU Time:', line):
                CPU_Times.append(float(line.split()[2]))
                print('Analysis ' + str(i) + ' complete: CPU Time: ' + line.split()[2] + '\n')

    mean_CPU_Times = numpy.mean(CPU_Times)
    std_CPU_Times = numpy.std(CPU_Times)
    maximum_CPU_Times = numpy.amax(CPU_Times)
    minimum_CPU_Times = numpy.amin(CPU_Times)

    print("CPU_Times: ")
    print(CPU_Times)
    print("mean_CPU_Times: " + str(mean_CPU_Times))
    print("std_CPU_Times: " + str(std_CPU_Times))
    print("maximum_CPU_Times: " + str(maximum_CPU_Times))
    print("minimum_CPU_Times: " + str(minimum_CPU_Times))

    f.write(exe[j])
    f.write(str(CPU_Times))
    f.write("\n\n")

f.close()

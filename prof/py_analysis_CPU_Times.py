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
exe = ["reg_standalone", "avx_si32_avx2", "nom_si32_avx2", "avx_fl32_avx2", "nom_fl32_avx2", "avx_si32_avx512", "nom_si32_avx512", "avx_fl32_avx512", "nom_fl32_avx512"]
for j in range(0,8)
    for i in range(0,50):

        output = subprocess.check_output("""/opt/intel/vtune_amplifier_2018.1.0.535340/bin64/amplxe-cl -collect hotspots -quiet -knob sampling-interval=1 -app-working-dir /home/gnssi9/dami7269/gnss-intrinsics/prof/ -- /home/gnssi9/dami7269/gnss-intrinsics/prof/a.out""", shell=True)

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
    f.write("\n")
    f.write(CPU_Times)
    f.write("\n")

f.close()
# Code derived from:

# https://stackoverflow.com/questions/4172448/is-it-possible-to-break-a-long-line-to-multiple-lines-in-python
# https://stackoverflow.com/questions/10358547/grep-for-contents-after-pattern
# https://stackoverflow.com/questions/89228/calling-an-external-command-in-python
# https://stackoverflow.com/questions/11566967/python-raise-child-exception-oserror-errno-2-no-such-file-or-directory
# https://stackoverflow.com/questions/8659275/how-to-store-the-result-of-an-executed-shell-command-in-a-variable-in-python
# https://docs.python.org/3/library/subprocess.html#replacing-shell-pipeline

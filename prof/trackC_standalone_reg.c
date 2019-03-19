/*!
 *  \file trackC_standalone_reg.c
 *  \brief      Simulates the tracking stage of a receiver using nominal C code
 *  \details    Profiles code when using:
 1. Carrier wave generation by means of DLUT method.
 2. Pseudorandom code generation by means of DLUT method.
 3. Down-conversion of the received signal by nominal multiplication.
 4. Multiplication of baseband signal with a local replica of
ranging code using with nominal C operations
 5. Accumulation to generate the correlation value with nominal C operations
 *  \author    Damian Miralles
 *  \author    Jake Johnson
 *  \date      Jan 23, 2018
 *  \pre       Make sure you have .bin files containing data and lookup tables.
 *  \code{.sh}
# Sample compilation script
$ gcc -I ../src/ trackC_standalone_reg.c -g
 -mavx2 -lm -o reg_standalone -O3
 *  \endcode
 */

#include "read_bin.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Sin and Cos Function approximations
#define gps_sin(x) (((x > 31416) || (x < 0 && x > -31416)) ? -1 : +1)
#define gps_cos(x)                                                             \
  (((x > 15708 && x < 47124) || (x < -15708 && x > -47124)) ? -1 : 1)

int main() {

  // Declarations
  int i, loopcount, angle, blksize, pCode, eCode, lCode, channelNr,
      totalChannels;
  int vsmCount, vsmInterval, PRN, dataAdaptCoeff;
  double remCodePhase, remCarrPhase, codePhaseStep;
  double earlyLateSpc, seekvalue, samplingFreq, trigarg, carrCos, carrSin,
      carrFreq;
  double I_E, Q_E, I_P, Q_P, I_L, Q_L, mixedcarrSin, mixedcarrCos, baseCode,
      mixedcarrSinLHCP;
  double mixedcarrCosLHCP, carrNco, oldCarrNco, tau1carr, tau2carr, carrError,
      oldCarrError;
  double PDIcarr, codeNco, oldCodeNco, tau1code, tau2code, codeError,
      oldCodeError, PDIcode;
  double codeFreq, codeFreqBasis, carrFreqBasis, absoluteSample, codeLength;
  double pwr, pwrSum, pwrSqrSum, pwrMean, pwrVar, pwrAvgSqr, pwrAvg, noiseVar,
      CNo, accInt, *pos;
  char *rawSignal, *rawSignalI, *rawSignalQ;
  char trackingStatusUpdated[100], arg[20];
  long int codePeriods;
  const double pi = 3.1415926535;

  FILE *fpdata, *fpdataLHCP;
  clock_t time_1, time_2, time_3, time_4;

  // Initialization
  remCodePhase = 0;
  remCarrPhase = 0;
  oldCarrNco = 0;
  oldCarrError = 0;
  carrError = 0;
  oldCarrError = 0;
  oldCodeNco = 0;
  oldCodeError = 0;
  absoluteSample = 0;
  vsmCount = 0;
  pwr = 0;
  CNo = 0;
  pwrSum = 0;
  pwrSqrSum = 0;

  // Get all the vectors/integers/strings from .bin files

  double caCode[1025]; //**
  getcaCodeFromFile("../data/caCode.bin", caCode);
  blksize = getIntFromFile("../data/blksize.bin");
  codePhaseStep = getDoubleFromFile("../data/codePhaseStep.bin");
  remCodePhase = getDoubleFromFile("../data/remCodePhase.bin");
  earlyLateSpc = getDoubleFromFile("../data/earlyLateSpc.bin");
  samplingFreq = getDoubleFromFile("../data/samplingFreq.bin");
  remCarrPhase = getDoubleFromFile("../data/remCarrPhase.bin");
  carrFreq = getDoubleFromFile("../data/carrFreq.bin");
  char fileid[] = "../data/GPS_and_GIOVE_A-NN-fs16_3676-if4_1304.bin";
  seekvalue = getDoubleFromFile("../data/skipvalue.bin");
  tau1carr = getDoubleFromFile("../data/tau1carr.bin");
  tau2carr = getDoubleFromFile("../data/tau2carr.bin");
  PDIcarr = getDoubleFromFile("../data/PDIcarr.bin");
  carrFreqBasis = getDoubleFromFile("../data/carrFreqBasis.bin");
  tau1code = getDoubleFromFile("../data/tau1code.bin");
  tau2code = getDoubleFromFile("../data/tau2code.bin");
  PDIcode = getDoubleFromFile("../data/PDIcode.bin");
  codeFreq = 1023002.79220779;
  codeFreqBasis = 1023002.79220779;
  codeLength = getDoubleFromFile("../data/codeLength.bin");
  codePeriods = (long int)getIntFromFile("../data/codePeriods.bin");
  // I removed the new line and added \n... this might cause issues
  char trackingStatus[] = "Tracking: Ch 1 of 8 \n PRN:22";
  dataAdaptCoeff = getIntFromFile("../data/dataAdaptCoeff.bin");
  vsmInterval = getIntFromFile("../data/VSMinterval.bin");
  accInt = getDoubleFromFile("../data/accTime.bin");

  // Declare outputs

  double carrFreq_output[codePeriods];
  double codeFreq_output[codePeriods];
  double absoluteSample_output[codePeriods];
  double codeError_output[codePeriods];
  double codeNco_output[codePeriods];
  double carrError_output[codePeriods];
  double carrNco_output[codePeriods];
  double I_E_output[codePeriods];
  double I_P_output[codePeriods];
  double I_L_output[codePeriods];
  double Q_E_output[codePeriods];
  double Q_P_output[codePeriods];
  double Q_L_output[codePeriods];
  double VSMIndex[codePeriods / vsmInterval];
  double VSMValue[codePeriods / vsmInterval];

  // Allocate memory for the signal
  rawSignal = calloc(dataAdaptCoeff * blksize, sizeof(char));
  rawSignalI = calloc(blksize, sizeof(char));
  rawSignalQ = calloc(blksize, sizeof(char));

  // Open the file for reading the data and fseek if required
  fpdata = fopen(fileid, "rb");
  fseek(fpdata, dataAdaptCoeff * seekvalue, SEEK_SET);

  for (loopcount = 0; loopcount < codePeriods; loopcount++) {

    I_E = 0;
    Q_E = 0;
    I_P = 0;
    Q_P = 0;
    I_L = 0;
    Q_L = 0;
    i = 0;

    codePhaseStep = codeFreq / samplingFreq;
    // blksize is supposed to be ceiling'd but for some reason compiler won't
    // find <math.h> header... >:(
    blksize = ceil((codeLength - remCodePhase) / codePhaseStep);

    i = fread(rawSignal, sizeof(char), dataAdaptCoeff * blksize, fpdata);

    // An error check should be added here to see if the required amount of data
    // can be read

    if (dataAdaptCoeff == 2) {
      for (i = 0; i < blksize; i++) {
        *(rawSignalI + i) = *(rawSignal + 2 * i);
        *(rawSignalQ + i) = *(rawSignal + 2 * i + 1);
      }
    }

    // Set up all the code phase tracking information
    // Define index into early,late and prompt codes

    for (i = 0; i < blksize; i++) {

      // Generate the carrier frequency to mix the signal to baseband
      trigarg = (2.0 * pi * carrFreq * (i / samplingFreq)) + remCarrPhase;
      angle = (int)(trigarg * 10000) % 62832;
      carrCos = (short)(round(8 * (gps_cos(angle))));
      carrSin = (short)(round(8 * (gps_sin(angle))));

      // First mix to baseband
      if (dataAdaptCoeff == 1) {
        mixedcarrSin = (carrSin * (*(rawSignal + i)));
        mixedcarrCos = (carrCos * (*(rawSignal + i)));
      } else if (dataAdaptCoeff == 2) {
        mixedcarrSin =
            (*(rawSignalI + i) * carrSin + *(rawSignalQ + i) * carrCos);
        mixedcarrCos =
            (*(rawSignalI + i) * carrCos - *(rawSignalQ + i) * carrSin);
      }

      baseCode = (i * codePhaseStep + remCodePhase);

      pCode = (int)(baseCode) < baseCode ? (baseCode + 1) : baseCode;
      eCode = (int)(baseCode - earlyLateSpc) < (baseCode - earlyLateSpc)
                  ? (baseCode - earlyLateSpc + 1)
                  : (baseCode - earlyLateSpc);
      lCode = (int)(baseCode + earlyLateSpc) < (baseCode + earlyLateSpc)
                  ? (baseCode + earlyLateSpc + 1)
                  : (baseCode + earlyLateSpc);

      // Now get early, late, and prompt values for each
      I_E += (*(caCode + eCode) * mixedcarrSin);
      Q_E += (*(caCode + eCode) * mixedcarrCos);
      I_P += (*(caCode + pCode) * mixedcarrSin);
      Q_P += (*(caCode + pCode) * mixedcarrCos);
      I_L += (*(caCode + lCode) * mixedcarrSin);
      Q_L += (*(caCode + lCode) * mixedcarrCos);
    }

    // Compute the VSM C/No
    pwr = I_P * I_P + Q_P * Q_P;
    pwrSum += pwr;
    pwrSqrSum += pwr * pwr;
    vsmCount++;

    if (vsmCount == vsmInterval) {
      pwrMean = pwrSum / vsmInterval;
      pwrVar = pwrSqrSum / vsmInterval - pwrMean * pwrMean;
      pwrAvgSqr = pwrMean * pwrMean - pwrVar;
      pwrAvgSqr = (pwrAvgSqr > 0) ? pwrAvgSqr : -pwrAvgSqr;
      pwrAvg = sqrt(pwrAvgSqr);
      noiseVar = 0.5 * (pwrMean - pwrAvg);
      CNo = (pwrAvg / accInt) / (2 * noiseVar);
      CNo = (CNo > 0) ? CNo : -CNo;
      CNo = 10 * log10(CNo);

      *(VSMIndex + loopcount / vsmInterval) = loopcount + 1;
      *(VSMValue + loopcount / vsmInterval) = CNo;

      vsmCount = 0;
      pwrSum = 0;
      pwrSqrSum = 0;
    }

    remCodePhase = ((remCodePhase) + (blksize)*codePhaseStep - 1023);
    trigarg = (2.0 * pi * carrFreq * (blksize / samplingFreq)) + remCarrPhase;
    remCarrPhase = trigarg - (2 * pi) * ((int)(trigarg / (2 * pi)));

    // Implement carrier loop discriminator (phase detector)
    carrError = atan(Q_P / I_P) / (2.0 * pi);

    // Implement carrier loop filter and generate NCO command
    carrNco = oldCarrNco + (tau2carr / tau1carr) * (carrError - oldCarrError) +
              carrError * (PDIcarr / tau1carr);
    oldCarrNco = carrNco;
    oldCarrError = carrError;

    // Modify carrier freq based on NCO command
    carrFreq = carrFreqBasis + carrNco;

    // Find DLL error and update code NCO -------------------------------------
    codeError = (sqrt(I_E * I_E + Q_E * Q_E) - sqrt(I_L * I_L + Q_L * Q_L)) /
                (sqrt(I_E * I_E + Q_E * Q_E) + sqrt(I_L * I_L + Q_L * Q_L));

    // Implement code loop filter and generate NCO command
    codeNco = oldCodeNco + (tau2code / tau1code) * (codeError - oldCodeError) +
              codeError * (PDIcode / tau1code);
    oldCodeNco = codeNco;
    oldCodeError = codeError;

    // Modify code freq based on NCO command
    codeFreq = codeFreqBasis - codeNco;
    absoluteSample = ftell(fpdata) / dataAdaptCoeff - remCodePhase;

    // Store values in output arrays         vvv Corresponding variable in
    // Matlab (trackResults)
    carrFreq_output[loopcount] = carrFreq; // codeFreq
    codeFreq_output[loopcount] = codeFreq;
    absoluteSample_output[loopcount] = absoluteSample; // absoluteSample
    codeError_output[loopcount] = codeError;           // dllDiscr
    codeNco_output[loopcount] = codeNco;               // dllDiscrFilt
    carrError_output[loopcount] = carrError;           // pllDiscr
    carrNco_output[loopcount] = carrNco;               // pllDiscrFilt
    I_E_output[loopcount] = I_E;                       // I_E
    I_P_output[loopcount] = I_P;                       // I_P
    I_L_output[loopcount] = I_L;                       // I_L
    Q_E_output[loopcount] = Q_E;                       // Q_E
    Q_P_output[loopcount] = Q_P;                       // Q_P
    Q_L_output[loopcount] = Q_L;                       // Q_L
  }
  // mexCallMATLAB(0,NULL,1, &hwb, "close");

  fclose(fpdata);

  // Write I_E_output to bin file
  FILE *fp = fopen("../plot/data_reg/I_E_output.bin", "wb");
  fwrite(I_E_output, sizeof *I_E_output, 37000, fp);

  // Write I_P_output to bin file
  fp = fopen("../plot/data_reg/I_P_output.bin", "wb");
  fwrite(I_P_output, sizeof *I_P_output, 37000, fp);

  // Write I_L_output to bin file
  fp = fopen("../plot/data_reg/I_L_output.bin", "wb");
  fwrite(I_L_output, sizeof *I_L_output, 37000, fp);

  // Write Q_E_output to bin file
  fp = fopen("../plot/data_reg/Q_E_output.bin", "wb");
  fwrite(Q_E_output, sizeof *Q_E_output, 37000, fp);

  // Write Q_P_output to bin file
  fp = fopen("../plot/data_reg/Q_P_output.bin", "wb");
  fwrite(Q_P_output, sizeof *Q_P_output, 37000, fp);

  // Write Q_L_output to bin file
  fp = fopen("../plot/data_reg/Q_L_output.bin", "wb");
  fwrite(Q_L_output, sizeof *Q_L_output, 37000, fp);

  return 0;
}

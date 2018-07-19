/*
 * This file is a version of trackC.c in Dr. Akos's matlab GNSS SDR simulation
 that
 * runs without MATLAB.  This version also uses AVX2 intrinsic functions with
 * 32-bit integers for addition 16-bit for multiplication
 * TODO: THIS CURRENTLY DOES NOT WORK

 *
 * Author: Jake Johnson
 * Date created: Jan 28, 2018
 * Last Modified: Jan 29, 2018
 *
 */

#include "avx2_intrinsics.h"
#include "read_bin.h" // For getting values from bin files
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
  double caCode[1025];
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
  codeFreq =
      1023002.79220779; // getDoubleFromFile("text_data_files/codeFreq.bin");
  codeFreqBasis =
      1023002.79220779; // getDoubleFromFile("text_data_files/codeFreqBasis.bin");
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

  const int lutSize = 256;       // [N=number of bits]
  int32_t sin_LUT_si32[lutSize]; // our sine wave LUT
  int32_t cos_LUT_si32[lutSize]; // our sine wave LUT
  const float delta_phi =
      (float)carrFreqBasis / samplingFreq * blksize; // phase increment

  int16_t sig_nco_si16[blksize]; // output buffer
  int32_t sig_nco_si32[blksize]; // output buffer
  float sig_nco_fl32[blksize];   // output buffer

  // Allocate memory for the signal
  rawSignal = calloc(dataAdaptCoeff * blksize, sizeof(char));

  // Open the file for reading the data and fseek if required
  fpdata = fopen(fileid, "rb");
  fseek(fpdata, dataAdaptCoeff * seekvalue, SEEK_SET);

  // Sine Look-up Table Generation
  for (int i = 0; i < lutSize; ++i) {
    sin_LUT_si32[i] = (int32_t)(10.0 * sinf(2.0f * pi * (float)i / lutSize));
    cos_LUT_si32[i] = (int32_t)(10.0 * cosf(2.0f * pi * (float)i / lutSize));
  }

  // START MAIN LOOP
  for (loopcount = 0; loopcount < codePeriods; loopcount++) {

    I_E = 0;
    Q_E = 0;
    I_P = 0;
    Q_P = 0;
    I_L = 0;
    Q_L = 0;
    i = 0;

    codePhaseStep = codeFreq / samplingFreq;
    blksize = ceil((codeLength - remCodePhase) / codePhaseStep);

    ///////////////////////// NEW CODE
    //////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Create blksize_arr
    double blksize_arr[blksize];
    for (i = 0; i < blksize; i++) {
      blksize_arr[i] = i;
    }

    ///////////////////////// END NEW CODE
    /////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    i = fread(rawSignal, sizeof(char), dataAdaptCoeff * blksize, fpdata);

    ///////////////////////// NEW CODE
    //////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // An error check should be added here to see if the required amount of data
    // can be read

    // instantiate vectors
    double trigarg_vec[blksize];
    double angle_vec[blksize];

    double carrCos_vec[blksize];
    double carrSin_vec[blksize];

    int32_t mixedcarrSin_vec[blksize];
    int32_t mixedcarrCos_vec[blksize];
    int32_t sin_nco_si32[blksize];
    int32_t cos_nco_si32[blksize];
    int32_t eCode_vec[blksize];
    int32_t lCode_vec[blksize];
    int32_t pCode_vec[blksize];

    avx2_nom_nco_si32(sin_nco_si32, sin_LUT_si32, blksize, remCarrPhase,
                      carrFreq, samplingFreq);
    avx2_nom_nco_si32(cos_nco_si32, cos_LUT_si32, blksize, remCarrPhase,
                      carrFreq, samplingFreq);

    // This loop is for parts of code I haven't brought out of loop or haven't
    // figured out how to
    for (i = 0; i < blksize; i++) {
      // Find PRN Values:
      baseCode = (i * codePhaseStep + remCodePhase);
      pCode = (int32_t)(baseCode) < baseCode ? (baseCode + 1) : baseCode;
      eCode = (int32_t)(baseCode - earlyLateSpc) < (baseCode - earlyLateSpc)
                  ? (baseCode - earlyLateSpc + 1)
                  : (baseCode - earlyLateSpc);
      lCode = (int32_t)(baseCode + earlyLateSpc) < (baseCode + earlyLateSpc)
                  ? (baseCode + earlyLateSpc + 1)
                  : (baseCode + earlyLateSpc);

      pCode_vec[i] = *(caCode + pCode);
      eCode_vec[i] = *(caCode + eCode);
      lCode_vec[i] = *(caCode + lCode);

      // // Generate the carrier frequency to mix the signal to baseband
      // trigarg_vec[i] =
      //     (2.0 * pi * carrFreq * (i / samplingFreq)) + remCarrPhase;
      // angle_vec[i] = (int)(trigarg_vec[i] * 10000) % 62832;
      // sin_nco_si32[i] = (int32_t)(round(8 * (gps_cos(angle_vec[i]))));
      // cos_nco_si32[i] = (int32_t)(round(8 * (gps_sin(angle_vec[i]))));
      //
      mixedcarrSin_vec[i] = sin_nco_si32[i] * rawSignal[i];
      mixedcarrCos_vec[i] = cos_nco_si32[i] * rawSignal[i];
    }

    // Sine AVX2 NCO Look-up Table Implementation

    // Mix to baseband
    // avx2_si32_x2_mul_si32(mixedcarrSin_vec, sin_nco_si32, (int32_t
    // *)rawSignal,
    //                       blksize);
    // avx2_si32_x2_mul_si32(mixedcarrCos_vec, cos_nco_si32, (int32_t
    // *)rawSignal,
    //                       blksize);

    // I_E
    double I_E = avx2_mul_and_acc_si32(eCode_vec, mixedcarrSin_vec, blksize);

    // I_L
    double I_L = avx2_mul_and_acc_si32(lCode_vec, mixedcarrSin_vec, blksize);

    // I_P
    double I_P = avx2_mul_and_acc_si32(pCode_vec, mixedcarrSin_vec, blksize);

    // Q_E
    double Q_E = avx2_mul_and_acc_si32(eCode_vec, mixedcarrCos_vec, blksize);

    // Q_L
    double Q_L = avx2_mul_and_acc_si32(lCode_vec, mixedcarrCos_vec, blksize);

    // Q_P
    double Q_P = avx2_mul_and_acc_si32(pCode_vec, mixedcarrCos_vec, blksize);

    //--------------------------------------------------------------------------

    ///////////////////////////////////// END NEW CODE
    ///////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

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
    /* COMMENTING OUT BECAUSE I_P == 0 at loopcount== */
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

  } // end for

  fclose(fpdata);

  // Write early, late, prompt values to bin files:-----------------------------

  // Write I_E_output to bin file
  FILE *fp =
      fopen("../plot/data_avx2_32i_add_mul_nom_lut/I_E_output.bin", "wb");
  fwrite(I_E_output, sizeof *I_E_output, 50000, fp);

  // Write I_P_output to bin file
  fp = fopen("../plot/data_avx2_32i_add_mul_nom_lut/I_P_output.bin", "wb");
  fwrite(I_P_output, sizeof *I_P_output, 50000, fp);

  // Write I_L_output to bin file
  fp = fopen("../plot/data_avx2_32i_add_mul_nom_lut/I_L_output.bin", "wb");
  fwrite(I_L_output, sizeof *I_L_output, 50000, fp);

  // Write Q_E_output to bin file
  fp = fopen("../plot/data_avx2_32i_add_mul_nom_lut/Q_E_output.bin", "wb");
  fwrite(Q_E_output, sizeof *Q_E_output, 50000, fp);

  // Write Q_P_output to bin file
  fp = fopen("../plot/data_avx2_32i_add_mul_nom_lut/Q_P_output.bin", "wb");
  fwrite(Q_P_output, sizeof *Q_P_output, 50000, fp);

  // Write Q_L_output to bin file
  fp = fopen("../plot/data_avx2_32i_add_mul_nom_lut/Q_L_output.bin", "wb");
  fwrite(Q_L_output, sizeof *Q_L_output, 50000, fp);

  //----------------------------------------------------------------------------

  return 0;
}

// compile: gcc -g -mavx -march=haswell trackC_standalone_AVX2.c -lm

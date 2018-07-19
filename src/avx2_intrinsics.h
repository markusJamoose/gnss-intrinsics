/*!
 *  \file avx2_intrinsics.h
 *  \brief      Provides c functions that implement Intel's AVX2 intrinsic
 * functions \details    Based off Volk kernel functions. Specifically the AVX
 * functions defined in: volk_32f_x2_multiply_32f.h volk_32f_accumulator_s32f.h
 * The files can be found at:
 *   https://github.com/gnuradio/volk/blob/master/kernels/volk/volk_32f_x2_multiply_32f.h
 *   https://github.com/gnuradio/volk/blob/master/kernels/volk/volk_32f_accumulator_s32f.h
 *  \author    Jake Johnson
 *  \version   4.1a
 *  \date      Jan 23, 2018
 *  \pre       Make sure you have .bin files containing data and lookup tables
 *  \bug       None reported
 *  \warning   None so far
 *  \copyright TBD
 */

/*
 * This file provides c functions that implement Intel's AVX2 intrinsic
 * functions
 *
 * Author: Jake
 * Date Created: Jan 23, 2018
 * Last Modified:  Jan 29, 2018
 *
 * Based off Volk kernel functions. Specifically the AVX functions defined in:
 *   volk_32f_x2_multiply_32f.h
 *   volk_32f_accumulator_s32f.h
 * The files can be found at:
 *   https://github.com/gnuradio/volk/blob/master/kernels/volk/volk_32f_x2_multiply_32f.h
 *   https://github.com/gnuradio/volk/blob/master/kernels/volk/volk_32f_accumulator_s32f.h
 *
 */

#include "immintrin.h"
#include <math.h>
#include <stdio.h>

/*!
 *  \brief     Multiply two short vectors using AVX2 Intrinsics
 *  \details   This class is used to demonstrate a number of section commands.
 *  \param[out] cVector Product vector storing the result of the multiplication
 *  \param[in] aVector Source vector with factors to multiply
 *  \param[in] bVector Source vector with factors to multiply
 *  \param[in] num_points Number of points to Multiply in the operation
 */

void avx2_nco_si32(int32_t *sig_nco, const int32_t *lut, const int blk_size,
                   const double rem_carr_phase, const double carr_freq,
                   const double samp_freq) {
  int inda;
  const unsigned int eight_points = blk_size / 8;
  const unsigned int nom_carr_step =
      (unsigned int)(carr_freq * (4294967296.0 / samp_freq) + 0.5);

  // Declarations for serial implementation
  unsigned int nom_carr_phase_base =
      (unsigned int)(rem_carr_phase * (4294967296.0 / (2.0 * M_PI)) + 0.5);
  unsigned int nom_carr_idx = 0;

  // Important variable declarations
  __m256i carr_phase_base = _mm256_set1_epi32(nom_carr_phase_base);
  __m256i carr_step_base =
      _mm256_set_epi32(7 * nom_carr_step, 6 * nom_carr_step, 5 * nom_carr_step,
                       4 * nom_carr_step, 3 * nom_carr_step, 2 * nom_carr_step,
                       1 * nom_carr_step, 0 * nom_carr_step);
  __m256i carr_idx = _mm256_set1_epi32(0);
  __m256i hex_ff =
      _mm256_set_epi32(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
  __m256i nco;
  __m256i carr_step_offset = _mm256_set1_epi32(8 * nom_carr_step);

  // First iteration happens outside the loop
  carr_phase_base = _mm256_add_epi32(carr_phase_base, carr_step_base);

  for (inda = 0; inda < eight_points; inda++) {
    // Shift packed 32-bit integers in a right by imm8 while shifting in zeros
    carr_idx = _mm256_srli_epi32(carr_phase_base, 24);
    // carr_idx = _mm256_and_si256(carr_idx, hex_ff);

    // Look in lut
    nco = _mm256_i32gather_epi32(lut, carr_idx, 4);

    // Delta step
    // carr_step_base = _mm256_add_epi32(carr_step_base, carr_step_offset);
    carr_phase_base = _mm256_add_epi32(carr_phase_base, carr_step_offset);

    // 5- Store values in output buffer
    _mm256_storeu_si256((__m256i *)sig_nco, nco);

    // 6- Update pointers
    sig_nco += 8;
  }

  inda = eight_points * 8;
  nom_carr_phase_base = (unsigned int)_mm256_extract_epi32(carr_phase_base, 7);

  // generate buffer of output
  for (; inda < blk_size; ++inda) {
    // Obtain integer index in 8:24 number
    nom_carr_idx = (nom_carr_phase_base >> 24) & 0xFF;
    // Look in lut
    *sig_nco++ = lut[nom_carr_idx]; // get sample value from LUT
    // Delta step
    nom_carr_phase_base += nom_carr_step;
  }
}

/*!
 *  \brief     Generates nominal NCO generation through LUT implementation
 *  \details   This class is used to demonstrate a number of section commands.
 *  \param[out] cVector Product vector storing the result of the multiplication
 *  \param[in] aVector Source vector with factors to multiply
 *  \param[in] bVector Source vector with factors to multiply
 *  \param[in] num_points Number of points to Multiply in the operation
 */

void avx2_nom_nco_si32(int32_t *sig_nco, const int32_t *LUT, const int blksize,
                       const double remCarrPhase, const double carrFreq,
                       const double sampFreq) {

  unsigned int carrPhaseBase =
      (remCarrPhase * (4294967296.0 / (2.0 * M_PI)) + 0.5);
  unsigned int carrStep = (carrFreq * (4294967296.0 / sampFreq) + 0.5);
  unsigned int carrIndex = 0;
  int inda;

  // Store this values for debug purposes only
  unsigned int carrPhaseBaseVec[blksize];
  unsigned int carrIndexVec[blksize];

  // for each sample
  for (inda = 0; inda < blksize; ++inda) {
    // Obtain integer index in 8:24 number
    carrIndexVec[inda] = carrIndex;
    carrIndex = (carrPhaseBase >> 24) & 0xFF;

    // Look in lut
    sig_nco[inda] = LUT[carrIndex];

    // Delta step
    carrPhaseBaseVec[inda] = carrPhaseBase;
    carrPhaseBase += carrStep;
  }
}

static inline double avx2_mul_and_acc_si32(const int *aVector,
                                           const int *bVector,
                                           unsigned int num_points) {

  int returnValue = 0;
  unsigned int number = 0;
  const unsigned int eigthPoints = num_points / 8;

  const int *aPtr = aVector;
  const int *bPtr = bVector;
  int tempBuffer[8];

  __m256i aVal, bVal, cVal;
  __m256i accumulator = _mm256_setzero_si256();

  for (; number < eigthPoints; number++) {

    // Load 256-bits of integer data from memory into dst. mem_addr does not
    // need to be aligned on any particular boundary.
    aVal = _mm256_loadu_si256((__m256i *)aPtr);
    bVal = _mm256_loadu_si256((__m256i *)bPtr);

    // TODO: More efficient way to exclude having this intermediate cVal
    // variable??
    cVal = _mm256_mullo_epi32(aVal, bVal);

    // accumulator += _mm256_mullo_epi16(aVal, bVal);
    accumulator = _mm256_add_epi32(accumulator, cVal);

    // Increment pointers
    aPtr += 8;
    bPtr += 8;
  }

  _mm256_storeu_si256((__m256i *)tempBuffer, accumulator);

  returnValue = tempBuffer[0];
  returnValue += tempBuffer[1];
  returnValue += tempBuffer[2];
  returnValue += tempBuffer[3];
  returnValue += tempBuffer[4];
  returnValue += tempBuffer[5];
  returnValue += tempBuffer[6];
  returnValue += tempBuffer[7];

  // Perform non SIMD leftover operations
  number = eigthPoints * 8;
  for (; number < num_points; number++) {
    returnValue += (*aPtr++) * (*bPtr++);
  }
  return returnValue;
}

static inline void avx2_si32_x2_mul_si32(int *cVector, const int *aVector,
                                         const int *bVector,
                                         unsigned int num_points) {

  unsigned int number = 0;
  const unsigned int eigthPoints = num_points / 8;

  int *cPtr = cVector;
  const int *aPtr = aVector;
  const int *bPtr = bVector;

  __m256i aVal, bVal, cVal;

  for (; number < eigthPoints; number++) {

    // Load 256-bits of integer data from memory into dst. mem_addr does not
    // need to be aligned on any particular boundary.
    aVal = _mm256_loadu_si256((__m256i *)aPtr);
    bVal = _mm256_loadu_si256((__m256i *)bPtr);

    // Multiply packed 16-bit integers in a and b, producing intermediate
    // signed 32-bit integers. Truncate each intermediate integer to the 18
    // most significant bits, round by adding 1, and store bits [16:1] to dst.
    cVal = _mm256_mullo_epi32(aVal, bVal);

    // Store 256-bits of integer data from a into memory. mem_addr does
    // not need to be aligned on any particular boundary.
    _mm256_storeu_si256((__m256i *)cPtr, cVal);

    // Increment pointers
    aPtr += 8;
    bPtr += 8;
    cPtr += 8;
  }

  number = eigthPoints * 8;
  for (; number < num_points; number++) {
    *cPtr++ = (*aPtr++) * (*bPtr++);
  }
}

static inline void avx2_mul_short(short *cVector, const short *aVector,
                                  const short *bVector,
                                  unsigned int num_points) {

  unsigned int number = 0;
  const unsigned int sixteenthPoints = num_points / 16;

  short *cPtr = cVector;
  const short *aPtr = aVector;
  const short *bPtr = bVector;

  __m256i aVal, bVal, cVal;

  for (; number < sixteenthPoints; number++) {

    // Load 256-bits of integer data from memory into dst. mem_addr does not
    // need to be aligned on any particular boundary.
    aVal = _mm256_loadu_si256((__m256i *)aPtr);
    bVal = _mm256_loadu_si256((__m256i *)bPtr);

    // Multiply packed 16-bit integers in a and b, producing intermediate
    // signed 32-bit integers. Truncate each intermediate integer to the 18
    // most significant bits, round by adding 1, and store bits [16:1] to dst.
    cVal = _mm256_mullo_epi16(aVal, bVal);

    // Store 256-bits of integer data from a into memory. mem_addr does
    // not need to be aligned on any particular boundary.
    _mm256_storeu_si256((__m256i *)cPtr, cVal);

    // Increment pointers
    aPtr += 16;
    bPtr += 16;
    cPtr += 16;
  }

  // Perform non SIMD leftover operations
  number = sixteenthPoints * 16;
  for (; number < num_points; number++) {
    *cPtr++ = (*aPtr++) * (*bPtr++);
  }
}

// Accumulate the elements of a vector and return result as a double
static inline double avx_accumulate_short(const short *inputBuffer,
                                          unsigned int num_points) {
  int returnValue = 0;
  unsigned int number = 0;
  const unsigned int sixteenthPoints = num_points / 16;

  const short *aPtr = inputBuffer;
  /*__VOLK_ATTR_ALIGNED(32)*/ short tempBuffer[16];

  __m256i accumulator = _mm256_setzero_si256();
  __m256i aVal = _mm256_setzero_si256();

  for (; number < sixteenthPoints; number++) {
    aVal = _mm256_loadu_si256((__m256i *)aPtr);
    accumulator = _mm256_adds_epi16(accumulator, aVal);
    aPtr += 16;
  }

  _mm256_storeu_si256((__m256i *)tempBuffer, accumulator);

  returnValue = tempBuffer[0];
  returnValue += tempBuffer[1];
  returnValue += tempBuffer[2];
  returnValue += tempBuffer[3];
  returnValue += tempBuffer[4];
  returnValue += tempBuffer[5];
  returnValue += tempBuffer[6];
  returnValue += tempBuffer[7];
  returnValue += tempBuffer[8];
  returnValue += tempBuffer[9];
  returnValue += tempBuffer[10];
  returnValue += tempBuffer[11];
  returnValue += tempBuffer[12];
  returnValue += tempBuffer[13];
  returnValue += tempBuffer[14];
  returnValue += tempBuffer[15];

  number = sixteenthPoints * 16;
  for (; number < num_points; number++) {
    returnValue += (*aPtr++);
  }
  return returnValue;
}

// Accumulate the elements of a vector
static inline double avx_accumulate_short_unsat(const short *inputBuffer,
                                                unsigned int num_points) {
  int returnValue = 0;
  unsigned int number = 0;
  const unsigned int sixteenthPoints = num_points / 16;

  const short *aPtr = inputBuffer;
  /*__VOLK_ATTR_ALIGNED(32)*/ short tempBuffer[16];

  __m256i accumulator = _mm256_setzero_si256();
  __m256i aVal = _mm256_setzero_si256();

  for (; number < sixteenthPoints; number++) {
    aVal = _mm256_loadu_si256((__m256i *)aPtr);
    accumulator = _mm256_adds_epi16(accumulator, aVal);
    aPtr += 16;
  }

  _mm256_storeu_si256((__m256i *)tempBuffer, accumulator);

  returnValue = tempBuffer[0];
  returnValue += tempBuffer[1];
  returnValue += tempBuffer[2];
  returnValue += tempBuffer[3];
  returnValue += tempBuffer[4];
  returnValue += tempBuffer[5];
  returnValue += tempBuffer[6];
  returnValue += tempBuffer[7];
  returnValue += tempBuffer[8];
  returnValue += tempBuffer[9];
  returnValue += tempBuffer[10];
  returnValue += tempBuffer[11];
  returnValue += tempBuffer[12];
  returnValue += tempBuffer[13];
  returnValue += tempBuffer[14];
  returnValue += tempBuffer[15];

  number = sixteenthPoints * 16;
  for (; number < num_points; number++) {
    returnValue += (*aPtr++);
  }
  return returnValue;
}

// Multiply and accumulate two vectors of short integers.  Return result in a
// double
// TODO: This maxes out at 524272 for some reason??? (i.e. won't return a value
// greater than 524272) Update: Probably because tempBuffer[] is a short array
static inline double avx2_mul_and_acc_short(const short *aVector,
                                            const short *bVector,
                                            unsigned int num_points) {

  int returnValue = 0;
  unsigned int number = 0;
  const unsigned int sixteenthPoints = num_points / 16;

  const short *aPtr = aVector;
  const short *bPtr = bVector;
  short tempBuffer[16];

  __m256i aVal, bVal, cVal;
  __m256i accumulator = _mm256_setzero_si256();

  for (; number < sixteenthPoints; number++) {

    // Load 256-bits of integer data from memory into dst. mem_addr does not
    // need to be aligned on any particular boundary.
    aVal = _mm256_loadu_si256((__m256i *)aPtr);
    bVal = _mm256_loadu_si256((__m256i *)bPtr);

    // TODO: More efficient way to exclude having this intermediate cVal
    // variable??
    cVal = _mm256_mullo_epi16(aVal, bVal);

    // accumulator += _mm256_mullo_epi16(aVal, bVal);
    accumulator = _mm256_adds_epi16(accumulator, cVal);

    // Increment pointers
    aPtr += 16;
    bPtr += 16;
  }

  _mm256_storeu_si256((__m256i *)tempBuffer, accumulator);

  returnValue = tempBuffer[0];
  returnValue += tempBuffer[1];
  returnValue += tempBuffer[2];
  returnValue += tempBuffer[3];
  returnValue += tempBuffer[4];
  returnValue += tempBuffer[5];
  returnValue += tempBuffer[6];
  returnValue += tempBuffer[7];
  returnValue += tempBuffer[8];
  returnValue += tempBuffer[9];
  returnValue += tempBuffer[10];
  returnValue += tempBuffer[11];
  returnValue += tempBuffer[12];
  returnValue += tempBuffer[13];
  returnValue += tempBuffer[14];
  returnValue += tempBuffer[15];

  // Perform non SIMD leftover operations
  number = sixteenthPoints * 16;
  for (; number < num_points; number++) {
    returnValue += (*aPtr++) * (*bPtr++);
  }
  return returnValue;
}

// Multiply two short vectors [16_bit multiplication]
// Store results as ints to perform 32 bit addition
static inline void avx2_mul_short_store_int(short *cVector,
                                            const short *aVector,
                                            const short *bVector,
                                            unsigned int num_points) {

  unsigned int number = 0;
  const unsigned int sixteenthPoints = num_points / 16;

  short *cPtr = cVector;
  const short *aPtr = aVector;
  const short *bPtr = bVector;

  __m256i aVal, bVal, cVal;

  for (; number < sixteenthPoints; number++) {

    // Load 256-bits of integer data from memory into dst. mem_addr does not
    // need to be aligned on any particular boundary.
    aVal = _mm256_loadu_si256((__m256i *)aPtr);
    bVal = _mm256_loadu_si256((__m256i *)bPtr);

    // Multiply packed 16-bit integers in a and b, producing intermediate
    // signed 32-bit integers. Truncate each intermediate integer to the 18
    // most significant bits, round by adding 1, and store bits [16:1] to dst.
    cVal = _mm256_mullo_epi16(aVal, bVal);

    // Store 256-bits of integer data from a into memory. mem_addr does
    // not need to be aligned on any particular boundary.
    _mm256_storeu_si256((__m256i *)cPtr, cVal);

    // Increment pointers
    aPtr += 16;
    bPtr += 16;
    cPtr += 16;
  }

  // Perform non SIMD leftover operations
  number = sixteenthPoints * 16;
  for (; number < num_points; number++) {
    *cPtr++ = (*aPtr++) * (*bPtr++);
  }
}

// Accumulate the elements of a vector [32 bit addition]
static inline double avx_accumulate_int(const int *inputBuffer,
                                        unsigned int num_points) {
  int returnValue = 0;
  unsigned int number = 0;
  const unsigned int eighthPoints = num_points / 8;

  const int *aPtr = inputBuffer;
  /*__VOLK_ATTR_ALIGNED(32)*/ int tempBuffer[8];

  __m256i accumulator = _mm256_setzero_si256();
  __m256i aVal = _mm256_setzero_si256();

  // no saturation used
  for (; number < eighthPoints; number++) {
    aVal = _mm256_loadu_si256((__m256i *)aPtr);
    accumulator = _mm256_add_epi32(accumulator, aVal);
    aPtr += 8;
  }

  _mm256_storeu_si256((__m256i *)tempBuffer, accumulator);

  returnValue = tempBuffer[0];
  returnValue += tempBuffer[1];
  returnValue += tempBuffer[2];
  returnValue += tempBuffer[3];
  returnValue += tempBuffer[4];
  returnValue += tempBuffer[5];
  returnValue += tempBuffer[6];
  returnValue += tempBuffer[7];

  number = eighthPoints * 8;
  for (; number < num_points; number++) {
    returnValue += (*aPtr++);
  }
  return returnValue;
}

/*!
 *  \file avx512_intrinsics.h
 *  \brief      Provides c functions that implement Intel's AVX512 intrinsic functions
 *  \details    Based off Volk kernel functions. Specifically the AVX functions defined in:
 *   volk_32f_x2_multiply_32f.h
 *   volk_32f_accumulator_s32f.h
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
 * This file provides c functions that implement Intel's AVX512 intrinsic functions
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


#include <stdio.h>
#include "immintrin.h"


// Multiply and accumulate two vectors of 16bit integers (16i).  Return accumulation result as a double
static inline double avx512_mul_and_acc_16i(const short *aVector, const short *bVector, unsigned int num_points)
{

   int returnValue = 0;
   unsigned int number = 0;
   const unsigned int thirtysecondthPoints = num_points / 32;


   const short* aPtr = aVector;
   const short* bPtr = bVector;
   short tempBuffer[32];

   __m512i aVal, bVal, cVal;
   __m512i accumulator = _mm512_setzero_si512();

   for(;number < thirtysecondthPoints; number++){

     // Load 512-bits of integer data from memory into dst. mem_addr does not
     // need to be aligned on any particular boundary.
     aVal = _mm512_loadu_si512((__m512i*)aPtr);
     bVal = _mm512_loadu_si512((__m512i*)bPtr);
     // TODO: More efficient way to exclude having this intermediate cVal variable??
     cVal = _mm512_mullo_epi16(aVal, bVal);

     accumulator = _mm512_adds_epi16(accumulator, cVal);

     // Increment pointers
     aPtr += 32;
     bPtr += 32;

   }

   _mm512_storeu_si512((__m512i*)tempBuffer, accumulator);

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
   returnValue += tempBuffer[16];
   returnValue += tempBuffer[17];
   returnValue += tempBuffer[18];
   returnValue += tempBuffer[19];
   returnValue += tempBuffer[20];
   returnValue += tempBuffer[21];
   returnValue += tempBuffer[22];
   returnValue += tempBuffer[23];
   returnValue += tempBuffer[24];
   returnValue += tempBuffer[25];
   returnValue += tempBuffer[26];
   returnValue += tempBuffer[27];
   returnValue += tempBuffer[28];
   returnValue += tempBuffer[29];
   returnValue += tempBuffer[30];
   returnValue += tempBuffer[31];

   // Perform non SIMD leftover operations
   number = thirtysecondthPoints * 32;
   for(;number < num_points; number++){
     returnValue += (*aPtr++) * (*bPtr++);
   }
   return returnValue;
}

/*
// Using this function as reference:
// Multiply and accumulate two vectors of short integers.  Return result in a double
// TODO: This maxes out at 524272 for some reason??? (i.e. won't return a value greater than 524272)
// Update: Probably because tempBuffer[] is a short array
static inline double avx2_mul_and_acc_short(const short *aVector, const short *bVector, unsigned int num_points)
{

  int returnValue = 0;
  unsigned int number = 0;
  const unsigned int sixteenthPoints = num_points / 16;


  const short* aPtr = aVector;
  const short* bPtr = bVector;
  short tempBuffer[16];

  __m256i aVal, bVal, cVal;
  __m256i accumulator = _mm256_setzero_si256();

  for(;number < sixteenthPoints; number++){

    // Load 256-bits of integer data from memory into dst. mem_addr does not
    // need to be aligned on any particular boundary.
    aVal = _mm256_loadu_si256((__m256i*)aPtr);
    bVal = _mm256_loadu_si256((__m256i*)bPtr);

    // TODO: More efficient way to exclude having this intermediate cVal variable??
    cVal = _mm256_mullo_epi16(aVal, bVal);

    //accumulator += _mm256_mullo_epi16(aVal, bVal);
    accumulator = _mm256_adds_epi16(accumulator, cVal);

    // Increment pointers
    aPtr += 16;
    bPtr += 16;

  }

  _mm256_storeu_si256((__m256i*)tempBuffer, accumulator);

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
  for(;number < num_points; number++){
    returnValue += (*aPtr++) * (*bPtr++);
  }
  return returnValue;
}
*/

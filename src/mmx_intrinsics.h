/*!
 *  \file mmx_intrinsics.h
 *  \brief      Provides c functions that implement Intel's MMX intrinsic functions
 *  \details    Based off Volk kernel functions. Specifically the functions defined in:
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
 * This file provides c functions that implement Intel's AVX2 intrinsic functions
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
#include "mmintrin.h"

/*!
 *  \brief     Multiply and accumulate two vectors of short integers with MMX intrinsic functions
 *  \details   This class is used to demonstrate a number of section commands.
 *  \param[out] int returnValue as result of multiplication and accumulation
 *  \param[in] aVector Source vector with factors to multiply
 *  \param[in] bVector Source vector with factors to multiply
 *  \param[in] num_points Number of points to Multiply in the operation
 */

static inline double mmx_mul_and_acc_short(const short *aVector, const short *bVector, unsigned int num_points)
{

  int returnValue = 0;
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  const short* aPtr = aVector;
  const short* bPtr = bVector;
  short tempBuffer[4];

  __m64 aVal, bVal, cVal;
  __m64 accumulator = _mm_setzero_si64();

  for(;number < quarterPoints; number++){

    // Load 256-bits of integer data from memory into dst. mem_addr does not
    // need to be aligned on any particular boundary.
    aVal = _m_from_int64(*aPtr);
    bVal = _m_from_int64(*bPtr);

    // TODO: More efficient way to exclude having this intermediate cVal variable??
    cVal = _mm_mullo_pi16(aVal, bVal);

    accumulator = _mm_adds_pi16(accumulator, cVal);

    // Increment pointers
    aPtr += 4;
    bPtr += 4;

  }

  //_mm256_storeu_si256((__m256i*)tempBuffer, accumulator);
  *tempBuffer = _m_to_int64(accumulator);

  returnValue = tempBuffer[0];
  returnValue += tempBuffer[1];
  returnValue += tempBuffer[2];
  returnValue += tempBuffer[3];

  // Perform non SIMD leftover operations
  number = quarterPoints * 4;
  for(;number < num_points; number++){
    returnValue += (*aPtr++) * (*bPtr++);
  }
  return returnValue;
}

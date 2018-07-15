/*!
 *  \file avx2_intrinsics.h
 *  \brief      Provides c functions that implement Intel's AVX2 intrinsic functions
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
#include "immintrin.h"

/*!
 *  \brief     Multiply two short vectors using AVX2 Intrinsics
 *  \details   This class is used to demonstrate a number of section commands.
 *  \param[out] cVector Product vector storing the result of the multiplication
 *  \param[in] aVector Source vector with factors to multiply
 *  \param[in] bVector Source vector with factors to multiply
 *  \param[in] num_points Number of points to Multiply in the operation
 */

 void avx2_nco_si32(int32_t * sig_nco, const int blksize, int32_t * LUT, const float delta_phi, const int lutSize){

 	const unsigned int eightPoints = blksize / 8;

 	__m256i idx 			= _mm256_set_epi32 (7, 6, 5, 4, 3, 2, 1, 0);
 	__m256i lut_sat 	     = _mm256_set1_epi32 (lutSize);
 	__m256i dphi 			= _mm256_set1_epi32((int)delta_phi);
 	__m256i phase_steps;
 	__m256i phase;
 	__m256i nco;

 	for(int number = 0;number < eightPoints; number++)
 	{
 		// 1- Find steps values
 		phase_steps = _mm256_set1_epi32 (number*8);
 		phase_steps = _mm256_add_epi32 (idx, phase_steps);

 		// 2- Find steps values
 		phase = _mm256_mul_epi32(dphi, phase_steps);

 		// 3- Keep phase bounded in LUT size
 		phase = _mm256_and_si256(phase, lut_sat);

 		// 4- Read LUT Values
 		nco = _mm256_i32gather_epi32 ( LUT, phase, 1);

 		// 5- Store values in output buffer
 		_mm256_storeu_si256((__m256i *)sig_nco, nco);

 		// 6- Update pointers
 		sig_nco += 8;
 	}

 	// Get latest phase increment of the buffer
 	float phase_s = 0;
 	int i = eightPoints * 8;

 	// generate buffer of output
 	for (; i < blksize; ++i)
 	{
 	    int phase_i = (int)phase_s;       // get integer part of our phase
 	    *sig_nco++  = LUT[phase_i];      // get sample value from LUT
 	    phase_s += delta_phi;             // increment phase
 	    if (phase_s >= (float)lutSize)	// handle wraparound
 	        phase_s -= (float)lutSize;
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

  void avx2_nom_nco_si32(int32_t * sig_nco, const int32_t * LUT, const int blksize, const double remCarrPhase, const double carrFreq, const double sampFreq){

    unsigned int carrPhaseBase = (remCarrPhase * (4294967296.0/ (2.0 * M_PI)) + 0.5);
  	unsigned int carrStep = (carrFreq * (4294967296.0 / sampFreq) + 0.5);
    unsigned int carrIndex = 0;
    int inda;

    // for each sample
    for (inda = 0; inda < blksize; ++inda) {
        // Obtain integer index in 8:24 number
        carrIndex = (carrPhaseBase >> 24) & 0xFF ;
        // Look in lut
        sig_nco[inda] = LUT[carrIndex];

        // Delta step
        carrPhaseBase += carrStep ;
    }

  }


 static inline double avx2_mul_and_acc_si32(const int *aVector, const int *bVector, unsigned int num_points)
 {

   int returnValue = 0;
   unsigned int number = 0;
   const unsigned int eigthPoints = num_points / 8;

   const int* aPtr = aVector;
   const int* bPtr = bVector;
   int tempBuffer[8];

   __m256i aVal, bVal, cVal;
   __m256i accumulator = _mm256_setzero_si256();

   for(;number < eigthPoints; number++){

     // Load 256-bits of integer data from memory into dst. mem_addr does not
     // need to be aligned on any particular boundary.
     aVal = _mm256_loadu_si256((__m256i*)aPtr);
     bVal = _mm256_loadu_si256((__m256i*)bPtr);

     // TODO: More efficient way to exclude having this intermediate cVal variable??
     cVal = _mm256_mul_epi32(aVal, bVal);

     //accumulator += _mm256_mullo_epi16(aVal, bVal);
     accumulator = _mm256_add_epi32(accumulator, cVal);

     // Increment pointers
     aPtr += 8;
     bPtr += 8;

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

   // Perform non SIMD leftover operations
   number = eigthPoints * 8;
   for(;number < num_points; number++){
     returnValue += (*aPtr++) * (*bPtr++);
   }
   return returnValue;
 }


 static inline void avx2_si32_x2_mul_si32(int *cVector, const int *aVector, const int *bVector, unsigned int num_points)
 {

   unsigned int number = 0;
   const unsigned int eigthPoints = num_points / 8;

   int* cPtr = cVector;
   const int* aPtr = aVector;
   const int* bPtr = bVector;

   __m256i aVal, bVal, cVal;

   for(;number < eigthPoints; number++){

     // Load 256-bits of integer data from memory into dst. mem_addr does not
     // need to be aligned on any particular boundary.
     aVal = _mm256_loadu_si256((__m256i*)aPtr);
     bVal = _mm256_loadu_si256((__m256i*)bPtr);

     // Multiply packed 16-bit integers in a and b, producing intermediate
     // signed 32-bit integers. Truncate each intermediate integer to the 18
     // most significant bits, round by adding 1, and store bits [16:1] to dst.
     cVal = _mm256_mul_epi32(aVal, bVal);

     // Store 256-bits of integer data from a into memory. mem_addr does
     // not need to be aligned on any particular boundary.
     _mm256_storeu_si256((__m256i*)cPtr,cVal);

     // Increment pointers
     aPtr += 8;
     bPtr += 8;
     cPtr += 8;

   }
}


static inline void avx2_mul_short(short *cVector, const short *aVector, const short *bVector, unsigned int num_points)
{

  unsigned int number = 0;
  const unsigned int sixteenthPoints = num_points / 16;

  short* cPtr = cVector;
  const short* aPtr = aVector;
  const short* bPtr = bVector;

  __m256i aVal, bVal, cVal;

  for(;number < sixteenthPoints; number++){

    // Load 256-bits of integer data from memory into dst. mem_addr does not
    // need to be aligned on any particular boundary.
    aVal = _mm256_loadu_si256((__m256i*)aPtr);
    bVal = _mm256_loadu_si256((__m256i*)bPtr);

    // Multiply packed 16-bit integers in a and b, producing intermediate
    // signed 32-bit integers. Truncate each intermediate integer to the 18
    // most significant bits, round by adding 1, and store bits [16:1] to dst.
    cVal = _mm256_mullo_epi16(aVal, bVal);

    // Store 256-bits of integer data from a into memory. mem_addr does
    // not need to be aligned on any particular boundary.
    _mm256_storeu_si256((__m256i*)cPtr,cVal);

    // Increment pointers
    aPtr += 16;
    bPtr += 16;
    cPtr += 16;

  }

  // Perform non SIMD leftover operations
  number = sixteenthPoints * 16;
    for(;number < num_points; number++){
      *cPtr++ = (*aPtr++) * (*bPtr++);
    }

}

// Accumulate the elements of a vector and return result as a double
static inline double avx_accumulate_short(const short* inputBuffer, unsigned int num_points)
{
  int returnValue = 0;
  unsigned int number = 0;
  const unsigned int sixteenthPoints = num_points / 16;

  const short* aPtr = inputBuffer;
  /*__VOLK_ATTR_ALIGNED(32)*/ short tempBuffer[16];

  __m256i accumulator = _mm256_setzero_si256();
  __m256i aVal = _mm256_setzero_si256();

  for(;number < sixteenthPoints; number++){
    aVal = _mm256_loadu_si256((__m256i*)aPtr);
    accumulator = _mm256_adds_epi16(accumulator, aVal);
    aPtr += 16;
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

  number = sixteenthPoints * 16;
  for(;number < num_points; number++){
    returnValue += (*aPtr++);
  }
  return returnValue;
}

// Accumulate the elements of a vector
static inline double avx_accumulate_short_unsat(const short* inputBuffer, unsigned int num_points)
{
  int returnValue = 0;
  unsigned int number = 0;
  const unsigned int sixteenthPoints = num_points / 16;

  const short* aPtr = inputBuffer;
  /*__VOLK_ATTR_ALIGNED(32)*/ short tempBuffer[16];

  __m256i accumulator = _mm256_setzero_si256();
  __m256i aVal = _mm256_setzero_si256();

  for(;number < sixteenthPoints; number++){
    aVal = _mm256_loadu_si256((__m256i*)aPtr);
    accumulator = _mm256_adds_epi16(accumulator, aVal);
    aPtr += 16;
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

  number = sixteenthPoints * 16;
  for(;number < num_points; number++){
    returnValue += (*aPtr++);
  }
  return returnValue;
}

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



// Multiply two short vectors [16_bit multiplication]
// Store results as ints to perform 32 bit addition
static inline void avx2_mul_short_store_int(short *cVector, const short *aVector, const short *bVector, unsigned int num_points)
{

  unsigned int number = 0;
  const unsigned int sixteenthPoints = num_points / 16;

  short* cPtr = cVector;
  const short* aPtr = aVector;
  const short* bPtr = bVector;

  __m256i aVal, bVal, cVal;

  for(;number < sixteenthPoints; number++){

    // Load 256-bits of integer data from memory into dst. mem_addr does not
    // need to be aligned on any particular boundary.
    aVal = _mm256_loadu_si256((__m256i*)aPtr);
    bVal = _mm256_loadu_si256((__m256i*)bPtr);


    // Multiply packed 16-bit integers in a and b, producing intermediate
    // signed 32-bit integers. Truncate each intermediate integer to the 18
    // most significant bits, round by adding 1, and store bits [16:1] to dst.
    cVal = _mm256_mullo_epi16(aVal, bVal);

    // Store 256-bits of integer data from a into memory. mem_addr does
    // not need to be aligned on any particular boundary.
    _mm256_storeu_si256((__m256i*)cPtr,cVal);

    // Increment pointers
    aPtr += 16;
    bPtr += 16;
    cPtr += 16;

  }

  // Perform non SIMD leftover operations
  number = sixteenthPoints * 16;
    for(;number < num_points; number++){
      *cPtr++ = (*aPtr++) * (*bPtr++);
    }

}

// Accumulate the elements of a vector [32 bit addition]
static inline double avx_accumulate_int(const int* inputBuffer, unsigned int num_points)
{
  int returnValue = 0;
  unsigned int number = 0;
  const unsigned int eighthPoints = num_points / 8;

  const int* aPtr = inputBuffer;
  /*__VOLK_ATTR_ALIGNED(32)*/ int tempBuffer[8];

  __m256i accumulator = _mm256_setzero_si256();
  __m256i aVal = _mm256_setzero_si256();

  // no saturation used
  for(;number < eighthPoints; number++){
    aVal = _mm256_loadu_si256((__m256i*)aPtr);
    accumulator = _mm256_add_epi32(accumulator, aVal);
    aPtr += 8;
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

  number = eighthPoints * 8;
  for(;number < num_points; number++){
    returnValue += (*aPtr++);
  }
  return returnValue;
}

/*
static inline int avx2_sum_elements_short(const short *aVector, unsigned int num_points)
{

  unsigned int number = 0;
  const unsigned int sixteenthPoints = num_points / 16;

  const short* aPtr = aVector;

  __m256i aVal, sumVal;

  for(;number < sixteenthPoints; number++){

    // Load 256-bits of integer data from memory into dst. mem_addr does not
    // need to be aligned on any particular boundary.
    aVal = _mm256_loadu_si256((__m256i*)aPtr);

    //cVal = _mm256_hadd_epi16(aVal, aVal);

    //return cVal[0] + cVal[1] + cVal[2] + cVal[3] + cVal[4] + cVal[5] + cVal[6] + cVal[7];

    // Store 256-bits of integer data from a into memory. mem_addr does
    // not need to be aligned on any particular boundary.
    //_mm256_storeu_si256((__m256i*)cPtr,cVal);

    // Increment pointers
    aPtr += 16;
    cPtr += 16;

  }

  // Perform non SIMD leftover operations
  number = sixteenthPoints * 16;
    for(;number < num_points; number++){
      *cPtr++ = (*aPtr++) * (*bPtr++);
    }

}*/


/*
// Original Agner function
// Horizontal add extended: Calculates the sum of all vector elements.
// Each element is zero-extended before addition to avoid overflow
static inline uint32_t horizontal_add_x (Vec16us const & a) {
    __m256i mask  = _mm256_set1_epi32(0x0000FFFF);                    // mask for even positions
    __m256i aeven = _mm256_and_si256(a,mask);                         // even numbered elements of a
    __m256i aodd  = _mm256_srli_epi32(a,16);                          // zero extend odd numbered elements
    __m256i sum1  = _mm256_add_epi32(aeven,aodd);                     // add even and odd elements
    __m256i sum2  = _mm256_hadd_epi32(sum1,sum1);                     // horizontally add 2x4 elements in 2 steps
    __m256i sum3  = _mm256_hadd_epi32(sum2,sum2);
#if defined (_MSC_VER) && _MSC_VER <= 1700 && ! defined(__INTEL_COMPILER)
    __m128i sum4  = _mm256_extractf128_si256(sum3,1);                 // bug in MS compiler VS 11
#else
    __m128i sum4  = _mm256_extracti128_si256(sum3,1);                 // get high part
#endif
    __m128i sum5  = _mm_add_epi32(_mm256_castsi256_si128(sum3),sum4); // add low and high parts
    return          _mm_cvtsi128_si32(sum5);
}
*/

// Modified Agner function
// Horizontal add extended: Calculates the sum of all vector elements.
// Each element is zero-extended before addition to avoid overflow
/*
static inline int horizontal_add_x (const short * a) {
    __m256i aVal  = _mm256_loadu_si256((__m256i*)a);                  // load vector
    __m256i mask  = _mm256_set1_epi32(0x0000FFFF);                    // mask for even positions
    __m256i aeven = _mm256_and_si256(aVal,mask);                      // even numbered elements of a
    __m256i aodd  = _mm256_srli_epi32(aVal,16);                       // zero extend odd numbered elements
    __m256i sum1  = _mm256_add_epi32(aeven,aodd);                     // add even and odd elements
    __m256i sum2  = _mm256_hadd_epi32(sum1,sum1);                     // horizontally add 2x4 elements in 2 steps
    __m256i sum3  = _mm256_hadd_epi32(sum2,sum2);
//#if defined (_MSC_VER) && _MSC_VER <= 1700 && ! defined(__INTEL_COMPILER)
//    __m128i sum4  = _mm256_extractf128_si256(sum3,1);                 // bug in MS compiler VS 11
//#else
    __m128i sum4  = _mm256_extracti128_si256(sum3,1);                 // get high part
//#endif
    __m128i sum5  = _mm_add_epi32(_mm256_castsi256_si128(sum3),sum4); // add low and high parts
    return          _mm_cvtsi128_si32(sum5);
}
*/

/*
// function to return sum of elements
// in an array of size n
int sum(const short* a, int n)
{
    int sum = 0; // initialize sum

    // Iterate through all elements
    // and add them to sum
    for (int i = 0; i < n; i++)
    sum += arr[i];

    return sum;
}

*/

/*
void avx512_mul(int *cVector, const int *aVector, const int *bVector, unsigned int num_points)
{

  unsigned int number = 0;
  const unsigned int eighthPoints = num_points / 8;

  int* cPtr = cVector;
  const int* aPtr = aVector;
  const int* bPtr = bVector;

  __m256i aVal, bVal, cVal;

  for(;number < eighthPoints; number++){

    // Load 16 32-bit integer values from memory
    // Intel description:
    //   Load 512-bits (composed of 16 packed 32-bit integers) from memory
    //   into dst. mem_addr must be aligned on a 64-byte boundary or a
    //   general-protection exception may be generated.
    // HOW DO I KNOW IF MY INTEGERS ARE PACKED OR NOT??
    aVal = _mm256_loadu_si256(aPtr);
    bVal = _mm256_loadu_si256(bPtr);

    // Multipliy vectors
    // Intel description:
    //   Multiply the low 32-bit integers from each packed 64-bit
    //   element in a and b, and store the signed 64-bit results in dst.
    cVal = _mm512_mul_epi32(aVal, bVal);

    // Increment pointers
    aPtr += 8;
    bPtr += 8;
    cPtr += 8;

  }

  // Perform non SIMD leftover operations
  number = eighthPoints * 8;
    for(;number < num_points; number++){
      *cPtr++ = (*aPtr++) * (*bPtr++);
    }

}
*/

/*
void avx2_mul_scalar(short *cVector, const short *aVector, short scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int sixteenthPoints = num_points / 16;

  short* cPtr = cVector;
  const short* aPtr = aVector;

  __m256i aVal, sclrVal, cVal;

  for(;number < sixteenthPoints; number++){

    // Load 256-bits of integer data from memory into dst. mem_addr does not
    // need to be aligned on any particular boundary.
    aVal = _mm256_loadu_si256((__m256i*)aPtr);
    sclrVal = _mm256_set1_epi16(scalar);

    // Multiply packed 16-bit integers in a and b, producing intermediate
    // signed 32-bit integers. Truncate each intermediate integer to the 18
    // most significant bits, round by adding 1, and store bits [16:1] to dst.
    cVal = _mm256_mullo_epi16(aVal, sclrVal);

    // Store 256-bits of integer data from a into memory. mem_addr does
    // not need to be aligned on any particular boundary.
    _mm256_storeu_si256((__m256i*)cPtr,cVal);

    // Increment pointers
    aPtr += 16;
    bPtr += 16;
    cPtr += 16;

  }

  // Perform non SIMD leftover operations
  number = sixteenthPoints * 16;
    for(;number < num_points; number++){
      *cPtr++ = (*aPtr++) * (*bPtr++);
    }
}
*/

/*
void avx2_div_floats(float *cVector, const float *aVector, const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int eighthPoints = num_points / 8;

  float* cPtr = cVector;
  const float* aPtr = aVector;
  const float* bPtr = bVector;

  __m256 aVal, bVal, cVal;

  for(;number < eighthPoints; number++){

    // Load 8 32-bit floating point values from memory
    // Intel description:
    //   Load 256-bits (composed of 8 packed single-precision (32-bit)
    //   floating-point elements) from memory into dst. mem_addr must be
    //   aligned on a 32-byte boundary or a general-protection exception
    //   may be generated.
    aVal = _mm256_loadu_ps(aPtr);
    bVal = _mm256_loadu_ps(bPtr);

    // Divide
    cVal = _mm256_div_ps(aVal, bVal);

    // Store
    _mm256_storeu_ps(cPtr,cVal); // Store the results back into the C container

    // Increment pointers
    aPtr += 8;
    bPtr += 8;
    cPtr += 8;

  }

  // Perform non SIMD leftover operations
  number = eighthPoints * 8;
    for(;number < num_points; number++){
      *cPtr++ = (*aPtr++) * (*bPtr++);
    }

}
*/

/*
void avx2_div_floats_scalar(float *cVector, const float *aVector, float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int eighthPoints = num_points / 8;

  float* cPtr = cVector;
  const float* aPtr = aVector;

  __m256i aVal, sclrVal, cVal;

  for(;number < sixteenthPoints; number++){

    // Load 256-bits of integer data from memory into dst. mem_addr does not
    // need to be aligned on any particular boundary.
    aVal = _mm256_loadu_si256((__m256i*)aPtr);
    sclrVal = _mm256_set1_epi16(scalar);

    // Multiply packed 16-bit integers in a and b, producing intermediate
    // signed 32-bit integers. Truncate each intermediate integer to the 18
    // most significant bits, round by adding 1, and store bits [16:1] to dst.
    cVal = _mm256_div_ps(aVal, sclrVal);

    // Store 256-bits of integer data from a into memory. mem_addr does
    // not need to be aligned on any particular boundary.
    _mm256_storeu_si256((__m256i*)cPtr,cVal);

    // Increment pointers
    aPtr += 8;
    bPtr += 8;
    cPtr += 8;

  }

  // Perform non SIMD leftover operations
  number = eighthPoints * 8;
    for(;number < num_points; number++){
      *cPtr++ = (*aPtr++) * (*bPtr++);
    }
}
*/

/*
void avx2_add_floats(float *cVector, const float *aVector, const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int eighthPoints = num_points / 8;

  float* cPtr = cVector;
  const float* aPtr = aVector;
  const float* bPtr = bVector;

  __m256 aVal, bVal, cVal;

  for(;number < eighthPoints; number++){

    // Load 8 32-bit floating point values from memory
    // Intel description:
    //   Load 256-bits (composed of 8 packed single-precision (32-bit)
    //   floating-point elements) from memory into dst. mem_addr must be
    //   aligned on a 32-byte boundary or a general-protection exception
    //   may be generated.
    aVal = _mm256_loadu_ps(aPtr);
    bVal = _mm256_loadu_ps(bPtr);

    // Divide
    cVal = _mm256_add_ps(aVal, bVal);

    // Store
    _mm256_storeu_ps(cPtr,cVal); // Store the results back into the C container

    // Increment pointers
    aPtr += 8;
    bPtr += 8;
    cPtr += 8;

  }

  // Perform non SIMD leftover operations
  number = eighthPoints * 8;
    for(;number < num_points; number++){
      *cPtr++ = (*aPtr++) * (*bPtr++);
    }

}
*/

/*
void avx2_add_floats_scalar(float *cVector, const float *aVector, const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int sixteenthPoints = num_points / 16;

  float* cPtr = cVector;
  const float* aPtr = aVector;

  __m256i aVal, sclrVal, cVal;

  for(;number < sixteenthPoints; number++){

    // Load 256-bits of integer data from memory into dst. mem_addr does not
    // need to be aligned on any particular boundary.
    aVal = _mm256_loadu_si256((__m256i*)aPtr);
    sclrVal = _mm256_set1_epi16(scalar);

    // Multiply packed 16-bit integers in a and b, producing intermediate
    // signed 32-bit integers. Truncate each intermediate integer to the 18
    // most significant bits, round by adding 1, and store bits [16:1] to dst.
    cVal = _mm256_add_ps(aVal, sclrVal);

    // Store 256-bits of integer data from a into memory. mem_addr does
    // not need to be aligned on any particular boundary.
    _mm256_storeu_si256((__m256i*)cPtr,cVal);

    // Increment pointers
    aPtr += 8;
    bPtr += 8;
    cPtr += 8;

  }

  // Perform non SIMD leftover operations
  number = eighthPoints * 8;
    for(;number < num_points; number++){
      *cPtr++ = (*aPtr++) * (*bPtr++);
    }

}
*/

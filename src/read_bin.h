/*!
 *  \file read_bin.h
 *  \brief      Utility functions for reading data from data files
 *  \author    Damian Miralles
 *  \author    Jake Johnson
 *  \version   4.1a
 *  \date      Jan 23, 2018
 */

#include <stdio.h>
#include <stdlib.h>

double getDoubleFromFile(char file_name[1000]) {
  enum { SIZE = 1 };
  double b[SIZE];
  FILE *fp = fopen(file_name, "rb");
  size_t ret_code = fread(b, sizeof *b, SIZE, fp); // reads an array of doubles
  if (ret_code == SIZE) {
    // puts("Array read successfully, contents: ");
    // for(int n = 0; n < SIZE; ++n) printf("%f ", b[n]); // Printed value is
    // truncated putchar('\n');
  } else { // error handling
    if (feof(fp))
      printf("Error reading test.bin: unexpected end of file\n");
    else if (ferror(fp)) {
      perror("Error reading test.bin");
    }
  }
  fclose(fp);
  return b[0];
}

int getIntFromFile(char file_name[1000]) {
  enum { SIZE = 1 };
  int b[SIZE];
  FILE *fp = fopen(file_name, "rb");
  size_t ret_code = fread(b, sizeof *b, SIZE, fp); // reads an array of doubles
  if (ret_code == SIZE) {
    // puts("Array read successfully, contents: ");
    // for(int n = 0; n < SIZE; ++n) printf("%f ", b[n]); // Printed value is
    // truncated putchar('\n');
  } else { // error handling
    if (feof(fp))
      printf("Error reading test.bin: unexpected end of file\n");
    else if (ferror(fp)) {
      perror("Error reading test.bin");
    }
  }
  fclose(fp);
  return b[0];
}

double getcaCodeFromFile(char file_name[1000], double *output_array) {
  enum { SIZE = 1025 };
  double b[SIZE];
  FILE *fp = fopen(file_name, "rb");
  size_t ret_code = fread(b, sizeof *b, SIZE, fp); // reads an array of doubles
  if (ret_code == SIZE) {
    // puts("Array read successfully, contents: ");
    for (int n = 0; n < SIZE; ++n) {
      output_array[n] = b[n];
      // printf("%f ", b[n]); // Printed value is truncated
      // putchar('\n');
    }
  } else { // error handling
    if (feof(fp))
      printf("Error reading test.bin: unexpected end of file\n");
    else if (ferror(fp)) {
      perror("Error reading test.bin");
    }
  }
  fclose(fp);
}

void getcaCodeFromFileAsInt(char file_name[1000], int *output_array) {
  enum { SIZE = 1025 };
  double b[SIZE];
  FILE *fp = fopen(file_name, "rb");
  size_t ret_code = fread(b, sizeof *b, SIZE, fp); // reads an array of doubles
  if (ret_code == SIZE) {
    // puts("Array read successfully, contents: ");
    for (int n = 0; n < SIZE; ++n) {
      output_array[n] = (int)b[n];
      // printf("%f ", b[n]); // Printed value is truncated
      // putchar('\n');
    }
  } else { // error handling
    if (feof(fp))
      printf("Error reading test.bin: unexpected end of file\n");
    else if (ferror(fp)) {
      perror("Error reading test.bin");
    }
  }
  fclose(fp);
}

void getcaCodeFromFileAsFloat(char file_name[1000], float *output_array) {
  enum { SIZE = 1025 };
  double b[SIZE];
  FILE *fp = fopen(file_name, "rb");
  size_t ret_code = fread(b, sizeof *b, SIZE, fp); // reads an array of doubles
  if (ret_code == SIZE) {
    // puts("Array read successfully, contents: ");
    for (int n = 0; n < SIZE; ++n) {
      output_array[n] = (float)b[n];
      // printf("%f ", b[n]); // Printed value is truncated
      // putchar('\n');
    }
  } else { // error handling
    if (feof(fp))
      printf("Error reading test.bin: unexpected end of file\n");
    else if (ferror(fp)) {
      perror("Error reading test.bin");
    }
  }
  fclose(fp);
}

void getDataFromFile(char file_name[1000], double *output_array, int size) {
  double b[size];
  FILE *fp = fopen(file_name, "rb");
  size_t ret_code = fread(b, sizeof *b, size, fp); // reads an array of doubles
  if (ret_code == size) {
    // puts("Array read successfully, contents: ");
    for (int n = 0; n < size; ++n) {
      output_array[n] = b[n];
      // printf("%f ", b[n]); // Printed value is truncated
      // putchar('\n');
    }
  } else { // error handling
    if (feof(fp))
      printf("Error reading test.bin: unexpected end of file\n");
    else if (ferror(fp)) {
      perror("Error reading test.bin");
    }
  }
  fclose(fp);
}

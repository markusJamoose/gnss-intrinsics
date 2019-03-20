# File path configurations
INC_DIR=../src/
SRC_DIR=../prof/
BIN_DIR=../install/
DOC_DIR=../doc/

# Compilation configurations
CC=gcc
INC=-I $(INC_DIR)
DBG=-g
OPT=-O3
LIB=-lm
AVX2_FLAGS=-mavx2
AVX512_FLAGS=-mavx512f -mavx512dq

all:reg_standalone \
	nom_si32_avx2 avx_si32_avx2 nom_fl32_avx2 avx_fl32_avx2 \
	nom_si32_avx512 avx_si32_avx512 nom_fl32_avx512 avx_fl32_avx512 \
	doc

### NOMINAL EXECUTABLES ###
reg_standalone:
	gcc $(INC) $(SRC_DIR)trackC_standalone_reg.c \
	 -o $(BIN_DIR)reg_standalone $(DBG) $(AVX2_FLAGS) $(LIB) $(OPT)

### AVX2 EXECUTABLES ###
nom_si32_avx2:
	gcc $(INC) $(SRC_DIR)trackC_standalone_avx2_32i_add_mul_nom_lut_code.c \
	 -o $(BIN_DIR)nom_si32_avx2 $(DBG) $(AVX2_FLAGS) $(LIB) $(OPT)

avx_si32_avx2:
	gcc $(INC) $(SRC_DIR)trackC_standalone_avx2_32i_add_mul_avx_lut_code.c \
	 -o $(BIN_DIR)avx_si32_avx2 $(DBG) $(AVX2_FLAGS) $(LIB) $(OPT)

avx_fl32_avx2:
	gcc $(INC) $(SRC_DIR)trackC_standalone_avx2_fl32_add_mul_avx_lut_code.c \
	 -o $(BIN_DIR)avx_fl32_avx2 $(DBG) $(AVX2_FLAGS) $(LIB) $(OPT)

nom_fl32_avx2:
	gcc $(INC) $(SRC_DIR)trackC_standalone_avx2_fl32_add_mul_nom_lut_code.c \
	 -o $(BIN_DIR)nom_fl32_avx2 $(DBG) $(AVX2_FLAGS) $(LIB) $(OPT)

### AVX512 EXECUTABLES ###
nom_si32_avx512:
	gcc $(INC) $(SRC_DIR)trackC_standalone_avx512_si32_add_mul_nom_lut_code.c \
	 -o $(BIN_DIR)nom_si32_avx512 $(DBG) $(AVX512_FLAGS) $(LIB) $(OPT)

avx_si32_avx512:
	gcc $(INC) $(SRC_DIR)trackC_standalone_avx512_si32_add_mul_avx_lut_code.c \
	 -o $(BIN_DIR)avx_si32_avx512 $(DBG) $(AVX512_FLAGS) $(LIB) $(OPT)

avx_fl32_avx512:
	gcc $(INC) $(SRC_DIR)trackC_standalone_avx512_fl32_add_mul_avx_lut_code.c \
 	 -o $(BIN_DIR)avx_fl32_avx512 $(DBG) $(AVX512_FLAGS) $(LIB) $(OPT)

nom_fl32_avx512:
	gcc $(INC) $(SRC_DIR)trackC_standalone_avx512_fl32_add_mul_nom_lut_code.c \
	 -o $(BIN_DIR)nom_fl32_avx512 $(DBG) $(AVX512_FLAGS) $(LIB) $(OPT)

doc:
	doxygen $(DOC_DIR)gnss-intrinsics-doxyfile

clean:
	rm -f $(BIN_DIR)reg_standalone
	rm -f $(BIN_DIR)nom_si32_avx2
	rm -f $(BIN_DIR)avx_si32_avx2
	rm -f $(BIN_DIR)nom_fl32_avx2
	rm -f $(BIN_DIR)avx_fl32_avx2
	rm -f $(BIN_DIR)nom_si32_avx512
	rm -f $(BIN_DIR)avx_si32_avx512
	rm -f $(BIN_DIR)nom_fl32_avx512
	rm -f $(BIN_DIR)avx_fl32_avx512

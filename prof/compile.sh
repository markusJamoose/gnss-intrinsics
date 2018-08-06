
#!/bin/bash
# declare STRING variable
STRING1="Compiling selected source files"
STRING2="Finishing compilation"
#print variable on a screen
echo $STRING1

# Standalone source file
gcc -I ../src/ trackC_standalone_reg.c -mavx512f -mavx512dq -lm -g -o reg_standalone -O3

# Compiling avx2 si32 files
gcc -I ../src/ trackC_standalone_avx2_32i_add_mul_avx_lut_code.c -g -mavx2 -lm -o avx_si32_avx2 -O3
gcc -I ../src/ trackC_standalone_avx2_32i_add_mul_nom_lut_code.c -g -mavx2 -lm -o nom_si32_avx2 -O3

# Compiling avx2 fl32 files
gcc -I ../src/ trackC_standalone_avx2_fl32_add_mul_avx_lut_code.c -g -mavx2 -lm -o avx_fl32_avx2 -O3
gcc -I ../src/ trackC_standalone_avx2_fl32_add_mul_nom_lut_code.c -g -mavx2 -lm -o nom_fl32_avx2 -O3

# Compiling avx512 si32 files
gcc -I ../src/ trackC_standalone_avx512_si32_add_mul_avx_lut_code.c -g -mavx512f -mavx512dq -lm -o avx_si32_avx512 -O3
gcc -I ../src/ trackC_standalone_avx512_si32_add_mul_nom_lut_code.c -g -mavx512f -mavx512dq -lm -o nom_si32_avx512 -O3

# Compiling avx2 fl32 files
gcc -I ../src/ trackC_standalone_avx512_fl32_add_mul_avx_lut_code.c -g -mavx512f -mavx512dq -lm -o avx_fl32_avx512 -O3
gcc -I ../src/ trackC_standalone_avx512_fl32_add_mul_nom_lut_code.c -g -mavx512f -mavx512dq -lm -o nom_fl32_avx512 -O3

echo $STRING2

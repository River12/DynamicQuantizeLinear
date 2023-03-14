#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <algorithm>    // std::fill
#include <vector>       // std::vector
#include <omp.h>
#include <math.h>
#include <immintrin.h>

using namespace std;

#define QMIN 0
#define QMAX 255

#define NUM_THREADS 32
#define TENSOR_SIZE 100000000
// #define TENSOR_SIZE 10000
int TILE_SIZE = 8;

/**
 * Test Platform
 * CPU: Intel(R) Core(TM) i9-7960X CPU @ 2.80GHz
*/

// middle tensor:  
// large tensor:
// extreme large tensor:


void dynamicQuantizeLinear_naive(float *input, size_t size, unsigned int *output, unsigned int &y_zero_point, float &y_scale)
{

    // calculate maximum and minimun value of input
    float min_val, max_val;
    min_val = INFINITY;
    max_val = -INFINITY;

    // for(size_t i=0; i<size; i++) {
    //     min_val = fminf(0, fminf(min_val, input[i]));
    //     max_val = fmaxf(0, fmaxf(max_val, input[i]));
    // }

    for(size_t i=0; i<size; i++) {
        min_val = fminf(min_val, input[i]);
        max_val = fmaxf(max_val, input[i]);
    }
    min_val = fminf(0, min_val);
    max_val = fmaxf(0, max_val);

    // calculate y_scale and y_zero_point
    y_scale = (max_val - min_val) / (QMAX - QMIN); 
    y_zero_point = fmaxf(QMIN, fminf(QMAX, round((0 - min_val) / y_scale)));

    // calculate y
    for(size_t i=0; i<size; i++) {
        output[i] = fmaxf(QMIN, fminf(QMAX, round(input[i] / y_scale) + y_zero_point));
    }

}



void dynamicQuantizeLinear_omp(float *input, size_t size, unsigned int *output, unsigned int &y_zero_point, float &y_scale)
{

    // calculate maximum and minimun value of input
    float min_val, max_val;
    min_val = INFINITY;
    max_val = -INFINITY;

    omp_set_num_threads(NUM_THREADS);

    #pragma omp parallel for simd reduction(min: min_val) reduction(max: max_val)
    for(size_t i = 0; i < size; i++) {
        float tmp = input[i];
        if(tmp < min_val) min_val = tmp;
        if(tmp > max_val) max_val = tmp;
    }
    if(0 < min_val) min_val = 0;
    if(0 > max_val) max_val = 0; 


    // calculate y_scale and y_zero_point
    y_scale = (max_val - min_val) / (QMAX - QMIN); 
    y_zero_point = fmaxf(QMIN, fminf(QMAX, round((0 - min_val) / y_scale)));


    // calculate y
    #pragma omp parallel for simd
    for(size_t i = 0; i < size; i++) {
        float tmp = round(input[i] / y_scale) + y_zero_point;
        if(QMAX < tmp) tmp = QMAX;
        if(QMIN > tmp) tmp = QMIN;
        output[i] = tmp;
    }

}


void dynamicQuantizeLinear_omp_tiling(float *input, size_t size, unsigned int *output, unsigned int &y_zero_point, float &y_scale)
{

    // calculate maximum and minimun value of input
    float min_val, max_val;
    min_val = INFINITY;
    max_val = -INFINITY;

    omp_set_num_threads(NUM_THREADS);

    #pragma omp parallel for simd reduction(min: min_val) reduction(max: max_val)
    for(size_t i = 0; i < size; i++) {
        float tmp = input[i];
        if(tmp < min_val) min_val = tmp;
        if(tmp > max_val) max_val = tmp;
    }
    if(0 < min_val) min_val = 0;
    if(0 > max_val) max_val = 0; 


    // calculate y_scale and y_zero_point
    y_scale = (max_val - min_val) / (QMAX - QMIN); 
    y_zero_point = fmaxf(QMIN, fminf(QMAX, round((0 - min_val) / y_scale)));

    size_t remain = size % TILE_SIZE;
    size_t num_iter = size - remain;

    // cout << "num_iter: " << num_iter << " remain: " << remain << endl;
    // calculate y
    #pragma omp parallel for
    for(size_t i = 0; i < num_iter; i += TILE_SIZE) {

        for(size_t j=0; j < TILE_SIZE; j++) {
            size_t idx = i + j;
            float tmp = round(input[idx] / y_scale) + y_zero_point;
            if(QMAX < tmp) tmp = QMAX;
            if(QMIN > tmp) tmp = QMIN;
            output[idx] = tmp;

        }

    }

    // process the remaining part
    for(size_t i = num_iter; i < size; i++) {
        float tmp = round(input[i] / y_scale) + y_zero_point;
        if(QMAX < tmp) tmp = QMAX;
        if(QMIN > tmp) tmp = QMIN;
        output[i] = tmp;
    }

}

/**
 * AVX instruction is added to previous openmp version
*/
void dynamicQuantizeLinear_omp_avx(float *input, size_t size, unsigned int *output, unsigned int &y_zero_point, float &y_scale)
{

    // calculate maximum and minimun value of input
    float min_val, max_val;
    min_val = INFINITY;
    max_val = -INFINITY;

    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel for simd reduction(min: min_val) reduction(max: max_val)
    for(size_t i = 0; i < size; i++) {
        float tmp = input[i];
        if(tmp < min_val) min_val = tmp;
        if(tmp > max_val) max_val = tmp;
    }
    if(0 < min_val) min_val = 0;
    if(0 > max_val) max_val = 0; 


    // calculate y_scale and y_zero_point
    y_scale = (max_val - min_val) / (QMAX - QMIN); 
    y_zero_point = fmaxf(QMIN, fminf(QMAX, round((0 - min_val) / y_scale)));


    __m256 y_scale256 = _mm256_set1_ps(1.0 / y_scale);              // inverse of y_scal, for easy multiplication
    __m256 y_zero_point256 = _mm256_set1_ps(y_zero_point * 1.0);
    __m256i mask256 = _mm256_set1_epi32((QMAX - QMIN));
    size_t remain = size % 8;
    size_t num_iter = size - remain;

    #pragma omp parallel for
    for(int i = 0; i < num_iter; i += 8) {

        __m256 input256 = _mm256_loadu_ps(&input[i]);
        __m256i tmp256 = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(input256, y_scale256), y_zero_point256));
        tmp256 = _mm256_and_si256(tmp256, mask256);
        _mm256_storeu_si256((__m256i*)&output[i], tmp256);

    }

    for(int i = size - remain; i < size; i++) {
        float tmp = round(input[i] / y_scale) + y_zero_point;
        if(QMAX < tmp) tmp = QMAX;
        if(QMIN > tmp) tmp = QMIN;
        output[i] = tmp;

    }

}


void dynamicQuantizeLinear_omp_avx512(float *input, size_t size, unsigned int *output, unsigned int &y_zero_point, float &y_scale)
{

    // calculate maximum and minimun value of input
    float min_val, max_val;
    min_val = INFINITY;
    max_val = -INFINITY;

    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel for simd reduction(min: min_val) reduction(max: max_val)
    for(size_t i = 0; i < size; i++) {
        float tmp = input[i];
        if(tmp < min_val) min_val = tmp;
        if(tmp > max_val) max_val = tmp;
    }
    if(0 < min_val) min_val = 0;
    if(0 > max_val) max_val = 0; 


    // calculate y_scale and y_zero_point
    y_scale = (max_val - min_val) / (QMAX - QMIN); 
    y_zero_point = fmaxf(QMIN, fminf(QMAX, round((0 - min_val) / y_scale)));


    __m512 y_scale512 = _mm512_set1_ps(1.0 / y_scale);        // inverse of y_scal, for easy multiplication
    __m512 y_zero_point512 = _mm512_set1_ps(y_zero_point * 1.0);
    __m512i mask512 = _mm512_set1_epi32((QMAX - QMIN));
    size_t remain = size % 16;
    size_t num_iter = size - remain;

    #pragma omp parallel for
    for(int i = 0; i < num_iter; i += 16) {

        __m512 input512 = _mm512_loadu_ps(&input[i]);
        __m512i tmp512 = _mm512_cvtps_epi32(_mm512_add_ps(_mm512_mul_ps(input512, y_scale512), y_zero_point512));
        tmp512 = _mm512_and_si512(tmp512, mask512);
        _mm512_storeu_si512((__m512i*)&output[i], tmp512);

    }

    for(int i = size - remain; i < size; i++) {
        float tmp = round(input[i] / y_scale) + y_zero_point;
        if(QMAX < tmp) tmp = QMAX;
        if(QMIN > tmp) tmp = QMIN;
        output[i] = tmp;
    }

}


void dynamicQuantizeLinear_omp_avx512_fused(float *input, size_t size, unsigned int *output, unsigned int &y_zero_point, float &y_scale)
{

    // calculate maximum and minimun value of input
    float min_val, max_val;
    min_val = INFINITY;
    max_val = -INFINITY;

    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel for simd reduction(min: min_val) reduction(max: max_val)
    for(size_t i = 0; i < size; i++) {
        float tmp = input[i];
        if(tmp < min_val) min_val = tmp;
        if(tmp > max_val) max_val = tmp;
    }
    if(0 < min_val) min_val = 0;
    if(0 > max_val) max_val = 0; 


    // calculate y_scale and y_zero_point
    y_scale = (max_val - min_val) / (QMAX - QMIN); 
    y_zero_point = fmaxf(QMIN, fminf(QMAX, round((0 - min_val) / y_scale)));


    __m512 y_scale512 = _mm512_set1_ps(1.0 / y_scale);        // inverse of y_scal, for easy multiplication
    __m512 y_zero_point512 = _mm512_set1_ps(y_zero_point * 1.0);
    __m512i mask512 = _mm512_set1_epi32((QMAX - QMIN));
    size_t remain = size % 16;
    size_t num_iter = size - remain;

    #pragma omp parallel for
    for(int i = 0; i < num_iter; i += 16) {

        __m512 input512 = _mm512_loadu_ps(&input[i]);
        // __m512i tmp512 = _mm512_cvtps_epi32(_mm512_add_ps(_mm512_mul_ps(input512, y_scale512), y_zero_point512));
        __m512i tmp512 = _mm512_cvtps_epi32(_mm512_fmadd_ps(input512, y_scale512, y_zero_point512));
        tmp512 = _mm512_and_si512(tmp512, mask512);
        _mm512_storeu_si512((__m512i*)&output[i], tmp512);

    }

    for(int i = size - remain; i < size; i++) {
        float tmp = round(input[i] / y_scale) + y_zero_point;
        if(QMAX < tmp) tmp = QMAX;
        if(QMIN > tmp) tmp = QMIN;
        output[i] = tmp;
    }

}




/******************* Test Case ***********************/
void test1_dynamicQuantizeLinear()
{
    // test case 1
    size_t size= 6;
    float A[size] = {0, 2, -3, -2.5, 1.34, 0.5};
    unsigned int B[size] = {0};
    float y_scale;
    unsigned int y_zero_point;


    // dynamicQuantizeLinear_naive(A, size, B, y_zero_point, y_scale);
    // dynamicQuantizeLinear_omp(A, size, B, y_zero_point, y_scale);
    // dynamicQuantizeLinear_omp_tiling(A, size, B, y_zero_point, y_scale);
    // dynamicQuantizeLinear_omp_avx(A, size, B, y_zero_point, y_scale);
    // dynamicQuantizeLinear_omp_avx512(A, size, B, y_zero_point, y_scale);
    dynamicQuantizeLinear_omp_avx512_fused(A, size, B, y_zero_point, y_scale);

    cout << "y_scale: " << y_scale << ", y_zero_point: " << y_zero_point << endl;
    for(int i=0; i<size; i++) {
        cout << B[i] << ", ";
    }
    cout << endl;

}

void test2_dynamicQuantizeLinear()
{
    // test case 1
    size_t size= 6;
    float A[size] = {-1.0, -2.1, -1.3, -2.5, -3.34, -4.0};
    unsigned int B[size] = {0};
    float y_scale;
    unsigned int y_zero_point;


    // dynamicQuantizeLinear_naive(A, size, B, y_zero_point, y_scale);
    // dynamicQuantizeLinear_omp(A, size, B, y_zero_point, y_scale);
    // dynamicQuantizeLinear_omp_tiling(A, size, B, y_zero_point, y_scale);
    // dynamicQuantizeLinear_omp_avx(A, size, B, y_zero_point, y_scale);
    // dynamicQuantizeLinear_omp_avx512(A, size, B, y_zero_point, y_scale);
    dynamicQuantizeLinear_omp_avx512_fused(A, size, B, y_zero_point, y_scale);

    cout << "y_scale: " << y_scale << ", y_zero_point: " << y_zero_point << endl;
    for(int i=0; i<size; i++) {
        cout << B[i] << ", ";
    }
    cout << endl;

}

void test3_dynamicQuantizeLinear()
{
    // test case 3
    size_t size = 12; 
    float A[size] = {1, 2.1, 1.3, 2.5, 3.34, 4.0, 1.5, 2.6, 3.9, 4.0, 3.0, 2.345};
    unsigned int B[size] = {0};
    float y_scale;
    unsigned int y_zero_point;


    // dynamicQuantizeLinear_naive(A, size, B, y_zero_point, y_scale);
    // dynamicQuantizeLinear_omp(A, size, B, y_zero_point, y_scale);
    // dynamicQuantizeLinear_omp_tiling(A, size, B, y_zero_point, y_scale);
    // dynamicQuantizeLinear_omp_avx(A, size, B, y_zero_point, y_scale);
    // dynamicQuantizeLinear_omp_avx512(A, size, B, y_zero_point, y_scale);
    dynamicQuantizeLinear_omp_avx512_fused(A, size, B, y_zero_point, y_scale);

    cout << "y_scale: " << y_scale << ", y_zero_point: " << y_zero_point << endl;
    for(int i=0; i<size; i++) {
        cout << B[i] << ", ";
    }
    cout << endl;

}


int main() {

    /*** Correctness Test ***/
    cout << "========== Correctness Test ==========" << endl;
    test1_dynamicQuantizeLinear();
    test2_dynamicQuantizeLinear();
    test3_dynamicQuantizeLinear();


    /*** Performance Test ***/
    cout << "========== Performance Test ==========" << endl;
    double start, end;
    double tm_naive, tm_omp, tm_omp_tiling, tm_omp_avx, tm_omp_avx512, tm_omp_avx512_fused;

    /*** init tensor ***/
    float *input, y_scale;
    unsigned int *output, y_zero_point; 

    input = new float[TENSOR_SIZE];
    output = new unsigned int[TENSOR_SIZE]();

    for(int i=0; i<TENSOR_SIZE; i++) input[i] = rand();

    cout << "input tensor size: " << sizeof(float) * TENSOR_SIZE / 1e6 << " MB" << endl;
    /*** dynamicQuantizeLinear_naive ***/
    start = omp_get_wtime();
    dynamicQuantizeLinear_naive(input, TENSOR_SIZE, output, y_zero_point, y_scale);
    end = omp_get_wtime();

    tm_naive = end - start;

    printf("dynamicQuantizeLinear_naive runtime:            %lf \n", tm_naive);
    fill_n(output, TENSOR_SIZE, 0);

    /*** dynamicQuantizeLinear_omp ***/
    start = omp_get_wtime();
    dynamicQuantizeLinear_omp(input, TENSOR_SIZE, output, y_zero_point, y_scale);
    end = omp_get_wtime();

    tm_omp = end - start;

    printf("dynamicQuantizeLinear_omp runtime:              %lf, speedup: %.2f\n", tm_omp, tm_naive/tm_omp);
    fill_n(output, TENSOR_SIZE, 0);
    

    /*** dynamicQuantizeLinear_omp_tiling ***/
    start = omp_get_wtime();
    dynamicQuantizeLinear_omp_tiling(input, TENSOR_SIZE, output, y_zero_point, y_scale);
    end = omp_get_wtime();

    tm_omp_tiling = end - start;

    printf("dynamicQuantizeLinear_omp_tiling runtime:       %lf, speedup: %.2f\n", tm_omp_tiling, tm_naive/tm_omp_tiling);
    fill_n(output, TENSOR_SIZE, 0);



    /*** dynamicQuantizeLinear_omp_avx ***/
    start = omp_get_wtime();
    dynamicQuantizeLinear_omp_avx(input, TENSOR_SIZE, output, y_zero_point, y_scale);
    end = omp_get_wtime();

    tm_omp_avx = end - start;
    printf("dynamicQuantizeLinear_omp_avx runtime:          %lf, speedup: %.2f\n", tm_omp_avx, tm_naive/tm_omp_avx);

    /*** dynamicQuantizeLinear_omp_avx512 ***/
    start = omp_get_wtime();
    dynamicQuantizeLinear_omp_avx512(input, TENSOR_SIZE, output, y_zero_point, y_scale);
    end = omp_get_wtime();

    tm_omp_avx512 = end - start;
    printf("dynamicQuantizeLinear_omp_avx512 runtime:       %lf, speedup: %.2f\n", tm_omp_avx512, tm_naive/tm_omp_avx512);


    /*** dynamicQuantizeLinear_omp_avx512_fused ***/
    start = omp_get_wtime();
    dynamicQuantizeLinear_omp_avx512_fused(input, TENSOR_SIZE, output, y_zero_point, y_scale);
    end = omp_get_wtime();

    tm_omp_avx512_fused = end - start;
    printf("dynamicQuantizeLinear_omp_avx512_fused runtime: %lf, speedup: %.2f\n", tm_omp_avx512_fused, tm_naive/tm_omp_avx512_fused);


    /*** write to .csv file for further analysis ***/
    FILE *fp;
    fp = fopen("performance.csv", "a+");
    fprintf(fp, "%lf, %lf, %lf, %lf, %lf, %lf\n", tm_naive, tm_omp, tm_omp_tiling, tm_omp_avx, tm_omp_avx512, tm_omp_avx512_fused);
    fclose(fp);

    // thread comparision

    // input size comparision

    // tile size?

    // any other method to optimize?




    delete []input;
    delete []output;

    return 0;
}
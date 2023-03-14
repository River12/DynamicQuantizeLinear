#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <math.h>
#include <immintrin.h>

using namespace std;

#define QMIN 0
#define QMAX 255
#define TENSOR_SIZE 1000000000    // extremely large tensor: 1e9
// #define TENSOR_SIZE 100000           // large tensor: 100,000 1e5

int NUM_THREADS = 32;
int TILE_SIZE = 32;

/**
 * Test Platform
 * CPU: Intel(R) Core(TM) i9-7960X CPU @ 2.80GHz
*/


void dynamicQuantizeLinear_naive(float *, size_t, unsigned int *, unsigned int &, float &);
void dynamicQuantizeLinear_omp(float *, size_t, unsigned int *, unsigned int &, float &);
void dynamicQuantizeLinear_omp_tiling(float *, size_t, unsigned int *, unsigned int &, float &);
void dynamicQuantizeLinear_omp_avx(float *, size_t, unsigned int *, unsigned int &, float &);
void dynamicQuantizeLinear_omp_avx512(float *, size_t, unsigned int *, unsigned int &, float &);
void dynamicQuantizeLinear_omp_avx512_fused(float *, size_t, unsigned int *, unsigned int &, float &);

void test1_dynamicQuantizeLinear();
void test2_dynamicQuantizeLinear();
void test3_dynamicQuantizeLinear();

void test_methods(float *, size_t, unsigned int *, unsigned int &, float &);
void test_scalability(float *, size_t, unsigned int *, unsigned int &, float &);
void test_tile_size(float *, size_t, unsigned int *, unsigned int &, float &);


int main() {


    for (size_t i = 0; i < 1000000000; i++) {};

    /*** Correctness Test ***/
    cout << "========== Correctness Test ==========" << endl;
    test1_dynamicQuantizeLinear();
    test2_dynamicQuantizeLinear();
    test3_dynamicQuantizeLinear();


    /*** Performance Test ***/
    cout << "========== Init input and output ==========" << endl;
    /*** init tensor ***/
    float *input, y_scale;
    unsigned int *output, y_zero_point; 

    input = new float[TENSOR_SIZE];
    output = new unsigned int[TENSOR_SIZE]();

    for(int i=0; i<TENSOR_SIZE; i++) input[i] = (float) rand();

    cout << "input tensor size: " << sizeof(float) * TENSOR_SIZE / 1e6 << " MB" << endl;


    // cout << "========== Methods Test ==========" << endl;
    // test_methods(input, TENSOR_SIZE, output, y_zero_point, y_scale);


    // cout << "========== Scalability Test ==========" << endl;
    // test_scalability(input, TENSOR_SIZE, output, y_zero_point, y_scale);


    cout << "========== Tile Size Test ==========" << endl;
    test_tile_size(input, TENSOR_SIZE, output, y_zero_point, y_scale);


    delete []input;
    delete []output;

    return 0;
}


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

    // double start, end, tm1, tm2;
    // start = omp_get_wtime();
    for(size_t i=0; i<size; i++) {
        min_val = fminf(min_val, input[i]);
        max_val = fmaxf(max_val, input[i]);
    }
    // end = omp_get_wtime();
    // tm1 = end - start;

    min_val = fminf(0, min_val);
    max_val = fmaxf(0, max_val);


    // calculate y_scale and y_zero_point
    y_scale = (max_val - min_val) / (QMAX - QMIN); 
    y_zero_point = fmaxf(QMIN, fminf(QMAX, round((0 - min_val) / y_scale)));

    // calculate y
    // start = omp_get_wtime();
    for(size_t i=0; i<size; i++) {
        output[i] = fmaxf(QMIN, fminf(QMAX, round(input[i] / y_scale) + y_zero_point));
    }
    // end = omp_get_wtime();
    // tm2 = end - start;

    // cout << "1st loop: " << tm1 << ", 2nd loop:" << tm2 << endl;
    // 1st loop: 4.58273, 2nd loop:9.45152    for input size 1e9
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
        float tmp = round(input[i] / y_scale) + y_zero_point;  // contain operations that cannot be vectorized
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
    size_t num_tile = (int) size / TILE_SIZE;
    // cout << "num_tile: " << num_tile << ", num_iter: " << num_iter << " remain: " << remain << endl;

    // calculate y

    // #pragma omp parallel for
    // for(size_t i = 0; i < num_iter; i += TILE_SIZE) {
    //     for(size_t j = i; j < i + TILE_SIZE; j++) {
    //         float tmp = round(input[j] / y_scale) + y_zero_point;
    //         if(QMAX < tmp) tmp = QMAX;
    //         if(QMIN > tmp) tmp = QMIN;
    //         output[j] = tmp;

    //     }

    // }

    // tiling optimized here, to improve cache hit
    #pragma omp parallel for
    for(size_t j = 0; j < num_tile; j++) {
        int tid = omp_get_thread_num();
        size_t idx = j * TILE_SIZE + tid;
        float tmp = round(input[idx] / y_scale) + y_zero_point;
        if(QMAX < tmp) tmp = QMAX;
        if(QMIN > tmp) tmp = QMIN;
        output[idx] = tmp;
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
    __m256i y_zero_point256 = _mm256_set1_epi32(y_zero_point);
    __m256i mask256 = _mm256_set1_epi32((QMAX - QMIN));

    size_t remain = size % 8;
    size_t num_iter = size - remain;
    size_t num_tile = (int) size / 8;

    // double start, end, tm256;
    // start = omp_get_wtime();

    #pragma omp parallel for
    for(size_t i = 0; i < num_iter; i += 8) {
        __m256 input256 = _mm256_loadu_ps(&input[i]);
        __m256i tmp256 = _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(input256, y_scale256)), y_zero_point256);
        tmp256 = _mm256_and_si256(tmp256, mask256);
        _mm256_storeu_si256((__m256i*)&output[i], tmp256);
    }


    // end = omp_get_wtime();
    // tm256 = end - start;
    // cout << "avx256: " << tm256 << endl;

    for(size_t i = size - remain; i < size; i++) {
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
    __m512i y_zero_point512 = _mm512_set1_epi32(y_zero_point);
    __m512i mask512 = _mm512_set1_epi32((QMAX - QMIN));
    size_t remain = size % 16;
    size_t num_iter = size - remain;


    // double start, end, tm512;
    // start = omp_get_wtime();

    #pragma omp parallel for
    for(size_t i = 0; i < num_iter; i += 16) {

        __m512 input512 = _mm512_loadu_ps(&input[i]);
        __m512i tmp512 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(input512, y_scale512)), y_zero_point512);
        tmp512 = _mm512_and_si512(tmp512, mask512);
        _mm512_storeu_si512((__m512i*)&output[i], tmp512);

    }

    // end = omp_get_wtime();
    // tm512 = end - start;
    // cout << "avx512: " << tm512 << endl;

    for(size_t i = size - remain; i < size; i++) {
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


    __m512 y_scale512 = _mm512_set1_ps(1.0 / y_scale);        // inverse of y_scale, for easy multiplication
    __m512 y_zero_point512 = _mm512_set1_ps(y_zero_point * 1.0);
    __m512i mask512 = _mm512_set1_epi32((QMAX - QMIN));
    size_t remain = size % 16;
    size_t num_iter = size - remain;

    #pragma omp parallel for
    for(size_t i = 0; i < num_iter; i += 16) {

        __m512 input512 = _mm512_loadu_ps(&input[i]);
        // __m512i tmp512 = _mm512_cvtps_epi32(_mm512_add_ps(_mm512_mul_ps(input512, y_scale512), y_zero_point512));
        __m512i tmp512 = _mm512_cvtps_epi32(_mm512_fmadd_ps(input512, y_scale512, y_zero_point512));
        tmp512 = _mm512_and_si512(tmp512, mask512);
        _mm512_storeu_si512((__m512i*)&output[i], tmp512);

    }

    for(size_t i = size - remain; i < size; i++) {
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
    dynamicQuantizeLinear_omp_avx(A, size, B, y_zero_point, y_scale);
    // dynamicQuantizeLinear_omp_avx512(A, size, B, y_zero_point, y_scale);
    // dynamicQuantizeLinear_omp_avx512_fused(A, size, B, y_zero_point, y_scale);

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
    dynamicQuantizeLinear_omp_avx(A, size, B, y_zero_point, y_scale);
    // dynamicQuantizeLinear_omp_avx512(A, size, B, y_zero_point, y_scale);
    // dynamicQuantizeLinear_omp_avx512_fused(A, size, B, y_zero_point, y_scale);

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
    dynamicQuantizeLinear_omp_avx(A, size, B, y_zero_point, y_scale);
    // dynamicQuantizeLinear_omp_avx512(A, size, B, y_zero_point, y_scale);
    // dynamicQuantizeLinear_omp_avx512_fused(A, size, B, y_zero_point, y_scale);

    cout << "y_scale: " << y_scale << ", y_zero_point: " << y_zero_point << endl;
    for(int i=0; i<size; i++) {
        cout << B[i] << ", ";
    }
    cout << endl;

}


void test_methods(float *input, size_t size, unsigned int *output, unsigned int &y_zero_point, float &y_scale)
{
    double start, end;
    double tm_naive, tm_omp, tm_omp_tiling, tm_omp_avx, tm_omp_avx512, tm_omp_avx512_fused;
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
    fill_n(output, TENSOR_SIZE, 0);


    /*** dynamicQuantizeLinear_omp_avx512 ***/
    start = omp_get_wtime();
    dynamicQuantizeLinear_omp_avx512(input, TENSOR_SIZE, output, y_zero_point, y_scale);
    end = omp_get_wtime();

    tm_omp_avx512 = end - start;
    printf("dynamicQuantizeLinear_omp_avx512 runtime:       %lf, speedup: %.2f\n", tm_omp_avx512, tm_naive/tm_omp_avx512);
    fill_n(output, TENSOR_SIZE, 0);

    /*** dynamicQuantizeLinear_omp_avx512_fused ***/
    start = omp_get_wtime();
    dynamicQuantizeLinear_omp_avx512_fused(input, TENSOR_SIZE, output, y_zero_point, y_scale);
    end = omp_get_wtime();

    tm_omp_avx512_fused = end - start;
    printf("dynamicQuantizeLinear_omp_avx512_fused runtime: %lf, speedup: %.2f\n", tm_omp_avx512_fused, tm_naive/tm_omp_avx512_fused);


    /*** write to .csv file for further analysis ***/
    FILE *fp;
    fp = fopen("opt_methods.csv", "a+");
    fprintf(fp, "%lf, %lf, %lf, %lf, %lf, %lf\n", tm_naive, tm_omp, tm_omp_tiling, tm_omp_avx, tm_omp_avx512, tm_omp_avx512_fused);
    fclose(fp);
}

void test_scalability(float *input, size_t size, unsigned int *output, unsigned int &y_zero_point, float &y_scale)
{
    // avx512
    double start, end;
    vector<double> avx512_vec; 
    for(NUM_THREADS=2; NUM_THREADS<257; NUM_THREADS *= 2) 
    {
        fill_n(output, TENSOR_SIZE, 0);
        start = omp_get_wtime();
        dynamicQuantizeLinear_omp_avx512(input, size, output, y_zero_point, y_scale);
        end = omp_get_wtime();
        avx512_vec.push_back(end - start);
    }

    FILE *fp;
    fp = fopen("avx512_scalability.csv", "a+");
    fprintf(fp, "%lf, %lf, %lf, %lf, %lf, %lf, %lf\n", avx512_vec[0], avx512_vec[1], 
            avx512_vec[2], avx512_vec[3], avx512_vec[4], avx512_vec[5], avx512_vec[6]);
    fclose(fp);


    // omp_tiling
    vector<double> omp_tiling_vec; 
    for(NUM_THREADS=2; NUM_THREADS<257; NUM_THREADS *= 2) 
    {
        fill_n(output, TENSOR_SIZE, 0);
        start = omp_get_wtime();

        dynamicQuantizeLinear_omp_tiling(input, size, output, y_zero_point, y_scale);
        end = omp_get_wtime();
        omp_tiling_vec.push_back(end - start);
    }


    fp = fopen("omp_tiling_scalability.csv", "a+");
    fprintf(fp, "%lf, %lf, %lf, %lf, %lf, %lf, %lf\n", omp_tiling_vec[0], omp_tiling_vec[1], 
            omp_tiling_vec[2], omp_tiling_vec[3], omp_tiling_vec[4], omp_tiling_vec[5], omp_tiling_vec[6]);
    fclose(fp);


}

void test_tile_size(float *input, size_t size, unsigned int *output, unsigned int &y_zero_point, float &y_scale)
{
 
    // omp_tiling
    double start, end;
    vector<double> omp_tiling_vec; 
    for(TILE_SIZE=2; TILE_SIZE<513; TILE_SIZE *= 2) 
    {
        // cout << TILE_SIZE << endl;
        fill_n(output, TENSOR_SIZE, 0);
        start = omp_get_wtime();
        dynamicQuantizeLinear_omp_tiling(input, size, output, y_zero_point, y_scale);
        end = omp_get_wtime();
        omp_tiling_vec.push_back(end - start);
    }

    FILE *fp;
    fp = fopen("omp_tiling_tile_size.csv", "a+");
    fprintf(fp, "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", omp_tiling_vec[0], 
    omp_tiling_vec[1],omp_tiling_vec[2], omp_tiling_vec[3], omp_tiling_vec[4], 
    omp_tiling_vec[5], omp_tiling_vec[6], omp_tiling_vec[7], omp_tiling_vec[8]);
    fclose(fp);


}


// TODO:
// memory alignment for SIMD
// float *input = (float *)_mm_malloc(TENSOR_SIZE * sizeof(float), 32);                        // error check if needed
// unsigned int *output = (unsigned int *)_mm_malloc(TENSOR_SIZE * sizeof(unsigned int), 32);  // error check if needed
// _mm_free(input);
// _mm_free(output);
// any other method to optimize
#pragma once

#include <CL/cl.h>

void multiply(float *a, float *b, float *c, int m, int n, int k);
void multiply_omp(float *a, float *b, float *c, int m, int n, int k);
void multiply_ocl(float *a, float *b, float *c, int m, int n, int k, cl_device_id deviceId, float *elapsed);

#pragma once

#include <CL/cl.h>

void multiply(float *a, float *b, float *c, int m, int n, int k);

namespace omp {
void multiply(float *a, float *b, float *c, int m, int n, int k);
} // namespace omp

namespace ocl {
void multiply(float *a, float *b, float *c, int m, int n, int k, cl_device_id deviceId, float *elapsed);
void multiplyBlock(float *a, float *b, float *c, int m, int n, int k, cl_device_id deviceId, float *elapsed);
void multiplyImage(float *a, float *b, float *c, int m, int n, int k, cl_device_id deviceId, float *elapsed);
} // namespace ocl

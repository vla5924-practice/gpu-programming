#pragma once

#include <CL/cl.h>

struct CompResults {
    int iter = 0;
    double kernelTime = 0;
    double fullTime = 0;
    float convNorm = 0;
};

CompResults jacobi(float *a, float *b, float *x, int n, int iter, float convThreshold, cl_device_id deviceId);
float deviation(float *a, float *b, float *x, int n);

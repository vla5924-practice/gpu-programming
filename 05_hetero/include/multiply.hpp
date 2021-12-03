#pragma once

#include <CL/cl.h>

void multiply(float *a, float *b, float *c, int n);

namespace ocl {
void multiply(float *a, float *b, float *c, int n, cl_device_id deviceId, float *elapsed);
void multiplyHetero(float *a, float *b, float *c, int n, int delim, cl_device_id cpuDeviceId, cl_device_id gpuDeviceId,
                    float *elapsed);
} // namespace ocl

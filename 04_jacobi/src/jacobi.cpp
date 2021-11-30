#include "jacobi.hpp"

#include <cmath>
#include <vector>

#include <omp.h>

#include "utils.hpp"

float norm(const std::vector<float> &x0, const std::vector<float> &x1) {
    size_t n = x0.size();
    float s = 0;
    for (size_t i = 0; i < n; i++)
        s += (x0[i] - x1[i]) * (x0[i] - x1[i]);
    return std::sqrt(s);
}

CompResults jacobi(float *a, float *b, float *x, int n, int iter, float epsilon, cl_device_id deviceId) {
    CompResults results;
    results.fullTime = omp_get_wtime();
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    std::string source = Utils::readFile(KERNELS_DIR "jacobi.cl");
    const char *strings[] = {source.c_str()};
    cl_int ret = 0;
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, &ret);
    ret = clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "jacobi", &ret);

    size_t vecSize = static_cast<size_t>(n) * sizeof(float);
    cl_mem aMem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * vecSize, nullptr, &ret);
    ret = clEnqueueWriteBuffer(queue, aMem, CL_TRUE, 0, n * vecSize, a, 0, nullptr, nullptr);
    cl_mem bMem = clCreateBuffer(context, CL_MEM_READ_ONLY, vecSize, nullptr, &ret);
    ret = clEnqueueWriteBuffer(queue, bMem, CL_TRUE, 0, vecSize, b, 0, nullptr, nullptr);

    cl_mem x0Mem = clCreateBuffer(context, CL_MEM_READ_ONLY, vecSize, nullptr, &ret);
    cl_mem x1Mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, vecSize, nullptr, &ret);
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &x0Mem);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &x1Mem);
    ret = clSetKernelArg(kernel, 4, sizeof(int), &n);

    results.iter = 0;
    results.convNorm = 0;
    std::vector<float> x0(0, n);
    std::vector<float> x1(b, b + n);
    results.kernelTime = 0;
    do {
        x0 = x1;

        ret = clEnqueueWriteBuffer(queue, x0Mem, CL_TRUE, 0, vecSize, x0.data(), 0, nullptr, nullptr);

        size_t globalWorkSize = static_cast<size_t>(n);

        float begin = omp_get_wtime();
        ret = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
        ret = clFinish(queue);
        float end = omp_get_wtime();
        results.kernelTime += end - begin;
        ret = clEnqueueReadBuffer(queue, x1Mem, CL_TRUE, 0, vecSize, x1.data(), 0, nullptr, nullptr);
        ret = clFinish(queue);

        results.convNorm = norm(x0, x1);
    } while (++results.iter < iter && results.convNorm > epsilon);
    for (int i = 0; i < n; i++)
        x[i] = x1[i];
    clReleaseMemObject(aMem);
    clReleaseMemObject(bMem);
    ret = clReleaseMemObject(x0Mem);
    ret = clReleaseMemObject(x1Mem);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    results.fullTime = omp_get_wtime() - results.fullTime;
    return results;
}

float deviation(float *a, float *b, float *x, int n) {
    float norm = 0;
    for (int i = 0; i < n; i++) {
        float s = 0;
        for (int j = 0; j < n; j++) {
            s += a[j * n + i] * x[j];
        }
        s -= b[i];
        norm += s * s;
    }
    return sqrt(norm);
}

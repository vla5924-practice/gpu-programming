#include "jacobi.hpp"

#include <cmath>
#include <vector>

#include <omp.h>

#include "utils.hpp"

static inline float vectorLength(const float *x, size_t n) {
    float s = 0;
    for (size_t i = 0; i < n; i++) {
        s += x[i] * x[i];
    }
    return std::sqrt(s);
}

static inline float normAbs(const std::vector<float> &x0, const std::vector<float> &x1) {
    size_t n = x0.size();
    float s = 0;
    for (size_t i = 0; i < n; i++) {
        s += (x0[i] - x1[i]) * (x0[i] - x1[i]);
    }
    return std::sqrt(s);
}

static inline float normRel(const std::vector<float> &x0, const std::vector<float> &x1) {
    return normAbs(x0, x1) / vectorLength(x0.data(), x0.size());
}

float norm(const std::vector<float> &x0, const std::vector<float> &x1) {
    return normRel(x0, x1);
}

CompResults jacobi(float *a, float *b, float *x, int n, int iter, float convThreshold, cl_device_id deviceId) {
    CompResults results;
    results.fullTime = omp_get_wtime();

    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    std::string source = Utils::readFile(KERNELS_DIR "jacobi.cl");
    const char *strings[] = {source.c_str()};
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
    clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "jacobi", nullptr);

    size_t vecSize = static_cast<size_t>(n) * sizeof(float);
    cl_mem aMem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * vecSize, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, aMem, CL_TRUE, 0, n * vecSize, a, 0, nullptr, nullptr);
    cl_mem bMem = clCreateBuffer(context, CL_MEM_READ_ONLY, vecSize, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, bMem, CL_TRUE, 0, vecSize, b, 0, nullptr, nullptr);
    cl_mem x0Mem = clCreateBuffer(context, CL_MEM_READ_ONLY, vecSize, nullptr, nullptr);
    cl_mem x1Mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, vecSize, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &x0Mem);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &x1Mem);
    clSetKernelArg(kernel, 4, sizeof(int), &n);

    results.iter = 0;
    results.convNorm = 0;
    results.kernelTime = 0;

    std::vector<float> x0(0, n);
    std::vector<float> x1(b, b + n);

    do {
        x0 = x1;
        clEnqueueWriteBuffer(queue, x0Mem, CL_TRUE, 0, vecSize, x0.data(), 0, nullptr, nullptr);
        size_t globalWorkSize = static_cast<size_t>(n);
        double begin = omp_get_wtime();
        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
        clFinish(queue);
        double end = omp_get_wtime();
        results.kernelTime += end - begin;
        clEnqueueReadBuffer(queue, x1Mem, CL_TRUE, 0, vecSize, x1.data(), 0, nullptr, nullptr);
        results.convNorm = norm(x0, x1);
    } while (++results.iter < iter && results.convNorm > convThreshold);

    for (int i = 0; i < n; i++)
        x[i] = x1[i];

    clReleaseMemObject(aMem);
    clReleaseMemObject(bMem);
    clReleaseMemObject(x0Mem);
    clReleaseMemObject(x1Mem);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    results.fullTime = omp_get_wtime() - results.fullTime;
    return results;
}

CompResults jacobiHetero(float *a, float *b, float *x, int n, int iter, float convThreshold, int delim,
                         cl_device_id cpuDeviceId, cl_device_id gpuDeviceId) {
    CompResults results;
    results.fullTime = omp_get_wtime();

    cl_context cpuContext = clCreateContext(nullptr, 1, &cpuDeviceId, nullptr, nullptr, nullptr);
    cl_context gpuContext = clCreateContext(nullptr, 1, &gpuDeviceId, nullptr, nullptr, nullptr);
    cl_command_queue cpuQueue = clCreateCommandQueue(cpuContext, cpuDeviceId, 0, nullptr);
    cl_command_queue gpuQueue = clCreateCommandQueue(gpuContext, gpuDeviceId, 0, nullptr);

    std::string source = Utils::readFile(KERNELS_DIR "jacobi.cl");
    const char *strings[] = {source.c_str()};
    cl_program cpuProgram = clCreateProgramWithSource(cpuContext, 1, strings, nullptr, nullptr);
    cl_program gpuProgram = clCreateProgramWithSource(gpuContext, 1, strings, nullptr, nullptr);
    clBuildProgram(cpuProgram, 1, &cpuDeviceId, nullptr, nullptr, nullptr);
    clBuildProgram(gpuProgram, 1, &gpuDeviceId, nullptr, nullptr, nullptr);
    cl_kernel cpuKernel = clCreateKernel(cpuProgram, "jacobi", nullptr);
    cl_kernel gpuKernel = clCreateKernel(gpuProgram, "jacobi", nullptr);

    size_t vecSize = static_cast<size_t>(n) * sizeof(float);
    cl_mem aMemCpu = clCreateBuffer(cpuContext, CL_MEM_READ_ONLY, n * vecSize, nullptr, nullptr);
    cl_mem bMemCpu = clCreateBuffer(cpuContext, CL_MEM_READ_ONLY, vecSize, nullptr, nullptr);
    cl_mem x0MemCpu = clCreateBuffer(cpuContext, CL_MEM_READ_ONLY, vecSize, nullptr, nullptr);
    cl_mem x1MemCpu = clCreateBuffer(cpuContext, CL_MEM_WRITE_ONLY, vecSize, nullptr, nullptr);

    cl_mem aMemGpu = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, n * vecSize, nullptr, nullptr);
    cl_mem bMemGpu = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, vecSize, nullptr, nullptr);
    cl_mem x0MemGpu = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, vecSize, nullptr, nullptr);
    cl_mem x1MemGpu = clCreateBuffer(gpuContext, CL_MEM_WRITE_ONLY, vecSize, nullptr, nullptr);

    clEnqueueWriteBuffer(cpuQueue, aMemCpu, CL_FALSE, 0, n * vecSize, a, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(cpuQueue, bMemCpu, CL_FALSE, 0, vecSize, b, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(gpuQueue, aMemGpu, CL_FALSE, 0, n * vecSize, a, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(gpuQueue, bMemGpu, CL_FALSE, 0, vecSize, b, 0, nullptr, nullptr);
    clFinish(cpuQueue);
    clFinish(gpuQueue);

    clSetKernelArg(cpuKernel, 0, sizeof(cl_mem), &aMemCpu);
    clSetKernelArg(cpuKernel, 1, sizeof(cl_mem), &bMemCpu);
    clSetKernelArg(cpuKernel, 2, sizeof(cl_mem), &x0MemCpu);
    clSetKernelArg(cpuKernel, 3, sizeof(cl_mem), &x1MemCpu);
    clSetKernelArg(cpuKernel, 4, sizeof(int), &n);

    clSetKernelArg(gpuKernel, 0, sizeof(cl_mem), &aMemGpu);
    clSetKernelArg(gpuKernel, 1, sizeof(cl_mem), &bMemGpu);
    clSetKernelArg(gpuKernel, 2, sizeof(cl_mem), &x0MemGpu);
    clSetKernelArg(gpuKernel, 3, sizeof(cl_mem), &x1MemGpu);
    clSetKernelArg(gpuKernel, 4, sizeof(int), &n);

    results.iter = 0;
    results.convNorm = 0;
    results.kernelTime = 0;

    std::vector<float> x0(0, n);
    std::vector<float> x1(b, b + n);

    do {
        x0 = x1;
        clEnqueueWriteBuffer(cpuQueue, x0MemCpu, CL_FALSE, 0, vecSize, x0.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(gpuQueue, x0MemGpu, CL_FALSE, 0, vecSize, x0.data(), 0, nullptr, nullptr);
        clFinish(cpuQueue);
        clFinish(gpuQueue);

        size_t cpuWorkSize = static_cast<size_t>(delim);
        size_t gpuWorkSize = static_cast<size_t>(n - delim);
        size_t cpuOffset = 0;
        size_t gpuOffset = static_cast<size_t>(delim);
        double begin = omp_get_wtime();
        clEnqueueNDRangeKernel(cpuQueue, cpuKernel, 1, &cpuOffset, &cpuWorkSize, nullptr, 0, nullptr, nullptr);
        clEnqueueNDRangeKernel(gpuQueue, gpuKernel, 1, &gpuOffset, &gpuWorkSize, nullptr, 0, nullptr, nullptr);
        clFinish(cpuQueue);
        clFinish(gpuQueue);
        double end = omp_get_wtime();
        results.kernelTime += end - begin;
        clEnqueueReadBuffer(cpuQueue, x1MemCpu, CL_FALSE, 0, delim * sizeof(float), x1.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(gpuQueue, x1MemGpu, CL_FALSE, delim * sizeof(float), vecSize - delim * sizeof(float),
                            x1.data() + delim, 0, nullptr, nullptr);
        clFinish(cpuQueue);
        clFinish(gpuQueue);
        results.convNorm = norm(x0, x1);
    } while (++results.iter < iter && results.convNorm > convThreshold);

    for (int i = 0; i < n; i++)
        x[i] = x1[i];

    clReleaseMemObject(aMemCpu);
    clReleaseMemObject(bMemCpu);
    clReleaseMemObject(x0MemCpu);
    clReleaseMemObject(x1MemCpu);
    clReleaseKernel(cpuKernel);
    clReleaseProgram(cpuProgram);
    clReleaseCommandQueue(cpuQueue);
    clReleaseContext(cpuContext);

    clReleaseMemObject(aMemGpu);
    clReleaseMemObject(bMemGpu);
    clReleaseMemObject(x0MemGpu);
    clReleaseMemObject(x1MemGpu);
    clReleaseKernel(gpuKernel);
    clReleaseProgram(gpuProgram);
    clReleaseCommandQueue(gpuQueue);
    clReleaseContext(gpuContext);

    results.fullTime = omp_get_wtime() - results.fullTime;
    return results;
}

static inline float deviationAbs(float *a, float *b, float *x, int n) {
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

static inline float deviationRel(float *a, float *b, float *x, int n) {
    return deviationAbs(a, b, x, n) / vectorLength(b, n);
}

float deviation(float *a, float *b, float *x, int n) {
    return deviationRel(a, b, x, n);
}

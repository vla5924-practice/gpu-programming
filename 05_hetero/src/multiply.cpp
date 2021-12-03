#include "multiply.hpp"

#include <omp.h>
#include <string>

#include "utils.hpp"

#define SAFE(X) (static_cast<size_t>(X))

void multiply(float *a, float *b, float *c, int n) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            float *s = c + n * row + col;
            *s = 0;
            for (int i = 0; i < n; i++)
                *s += a[row * n + i] * b[col + n * i];
        }
    }
}

namespace ocl {

void multiply(float *a, float *b, float *c, int n, cl_device_id deviceId, float *elapsed) {
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    std::string source = Utils::readFile(KERNELS_DIR "multiply.cl");
    const char *strings[] = {source.c_str()};
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
    clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "multiply", nullptr);

    cl_mem aMem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * n * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, aMem, CL_TRUE, 0, n * n * sizeof(float), a, 0, nullptr, nullptr);
    cl_mem bMem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * n * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, bMem, CL_TRUE, 0, n * n * sizeof(float), b, 0, nullptr, nullptr);
    cl_mem cMem = clCreateBuffer(context, CL_MEM_READ_WRITE, n * n * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, cMem, CL_TRUE, 0, n * n * sizeof(float), c, 0, nullptr, nullptr);

    int isUpper = 1;
    int delim = n;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
    clSetKernelArg(kernel, 3, sizeof(int), &n);
    clSetKernelArg(kernel, 4, sizeof(int), &isUpper);
    clSetKernelArg(kernel, 5, sizeof(int), &delim);

    size_t globalWorkSize[] = {SAFE(n), SAFE(n)};
    size_t localWorkSize[] = {16u, 16u};
    float begin = omp_get_wtime();
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    clFinish(queue);
    float end = omp_get_wtime();
    if (elapsed != nullptr)
        *elapsed = end - begin;
    clEnqueueReadBuffer(queue, cMem, CL_TRUE, 0, n * n * sizeof(float), c, 0, nullptr, nullptr);

    clReleaseMemObject(aMem);
    clReleaseMemObject(bMem);
    clReleaseMemObject(cMem);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

void multiplyHetero(float *a, float *b, float *c, int n, int delim, cl_device_id cpuDeviceId, cl_device_id gpuDeviceId,
                    float *elapsed) {
    cl_int ret = 0;
    cl_context cpuContext = clCreateContext(nullptr, 1, &cpuDeviceId, nullptr, nullptr, &ret);
    cl_context gpuContext = clCreateContext(nullptr, 1, &gpuDeviceId, nullptr, nullptr, &ret);
    cl_command_queue cpuQueue = clCreateCommandQueue(cpuContext, cpuDeviceId, 0, &ret);
    cl_command_queue gpuQueue = clCreateCommandQueue(gpuContext, gpuDeviceId, 0, &ret);

    std::string source = Utils::readFile(KERNELS_DIR "multiply.cl");
    const char *strings[] = {source.c_str()};
    cl_program cpuProgram = clCreateProgramWithSource(cpuContext, 1, strings, nullptr, &ret);
    ret = clBuildProgram(cpuProgram, 1, &cpuDeviceId, nullptr, nullptr, nullptr);
    cl_program gpuProgram = clCreateProgramWithSource(gpuContext, 1, strings, nullptr, &ret);
    ret = clBuildProgram(gpuProgram, 1, &gpuDeviceId, nullptr, nullptr, nullptr);
    cl_kernel cpuKernel = clCreateKernel(cpuProgram, "multiply", &ret);
    cl_kernel gpuKernel = clCreateKernel(gpuProgram, "multiply", &ret);

    cl_mem aMemCpu = clCreateBuffer(cpuContext, CL_MEM_READ_ONLY, n * n * sizeof(float), nullptr, &ret);
    cl_mem aMemGpu = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, n * n * sizeof(float), nullptr, &ret);
    ret = clEnqueueWriteBuffer(cpuQueue, aMemCpu, CL_FALSE, 0, n * n * sizeof(float), a, 0, nullptr, nullptr);
    ret = clEnqueueWriteBuffer(gpuQueue, aMemGpu, CL_FALSE, 0, n * n * sizeof(float), a, 0, nullptr, nullptr);
    cl_mem bMemCpu = clCreateBuffer(cpuContext, CL_MEM_READ_ONLY, n * n * sizeof(float), nullptr, &ret);
    cl_mem bMemGpu = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, n * n * sizeof(float), nullptr, &ret);
    ret = clEnqueueWriteBuffer(cpuQueue, bMemCpu, CL_FALSE, 0, n * n * sizeof(float), b, 0, nullptr, nullptr);
    ret = clEnqueueWriteBuffer(gpuQueue, bMemGpu, CL_FALSE, 0, n * n * sizeof(float), b, 0, nullptr, nullptr);
    cl_mem cMemCpu = clCreateBuffer(cpuContext, CL_MEM_READ_WRITE, n * n * sizeof(float), nullptr, &ret);
    cl_mem cMemGpu = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, n * n * sizeof(float), nullptr, &ret);
    ret = clEnqueueWriteBuffer(cpuQueue, cMemCpu, CL_FALSE, 0, n * n * sizeof(float), c, 0, nullptr, nullptr);
    ret = clEnqueueWriteBuffer(gpuQueue, cMemGpu, CL_FALSE, 0, n * n * sizeof(float), c, 0, nullptr, nullptr);
    ret = clFinish(cpuQueue);
    ret = clFinish(gpuQueue);

    int cpuIsUpper = 1;
    int gpuIsUpper = 0;

    ret = clSetKernelArg(cpuKernel, 0, sizeof(cl_mem), &aMemCpu);
    ret = clSetKernelArg(cpuKernel, 1, sizeof(cl_mem), &bMemCpu);
    ret = clSetKernelArg(cpuKernel, 2, sizeof(cl_mem), &cMemCpu);
    ret = clSetKernelArg(cpuKernel, 3, sizeof(int), &n);
    ret = clSetKernelArg(cpuKernel, 4, sizeof(int), &cpuIsUpper);
    ret = clSetKernelArg(cpuKernel, 5, sizeof(int), &delim);

    ret = clSetKernelArg(gpuKernel, 0, sizeof(cl_mem), &aMemGpu);
    ret = clSetKernelArg(gpuKernel, 1, sizeof(cl_mem), &bMemGpu);
    ret = clSetKernelArg(gpuKernel, 2, sizeof(cl_mem), &cMemGpu);
    ret = clSetKernelArg(gpuKernel, 3, sizeof(int), &n);
    ret = clSetKernelArg(gpuKernel, 4, sizeof(int), &gpuIsUpper);
    ret = clSetKernelArg(gpuKernel, 5, sizeof(int), &delim);

    size_t globalWorkSize[] = {SAFE(n), SAFE(n)};
    size_t localWorkSize[] = {16u, 16u};
    float begin = omp_get_wtime();
    ret = clEnqueueNDRangeKernel(cpuQueue, cpuKernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    ret = clEnqueueNDRangeKernel(gpuQueue, gpuKernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    ret = clFinish(cpuQueue);
    ret = clFinish(gpuQueue);
    float end = omp_get_wtime();
    if (elapsed != nullptr)
        *elapsed = end - begin;
    ret = clEnqueueReadBuffer(cpuQueue, cMemCpu, CL_FALSE, 0, n * n * sizeof(float), c, 0, nullptr, nullptr);
    ret = clEnqueueReadBuffer(gpuQueue, cMemGpu, CL_FALSE, delim, n * n * sizeof(float), c, 0, nullptr, nullptr);
    ret = clFinish(cpuQueue);
    ret = clFinish(gpuQueue);

    ret = clReleaseMemObject(aMemCpu);
    ret = clReleaseMemObject(bMemCpu);
    ret = clReleaseMemObject(cMemCpu);
    ret = clReleaseMemObject(aMemGpu);
    ret = clReleaseMemObject(bMemGpu);
    ret = clReleaseMemObject(cMemGpu);
    ret = clReleaseKernel(cpuKernel);
    ret = clReleaseKernel(gpuKernel);
    ret = clReleaseProgram(cpuProgram);
    ret = clReleaseProgram(gpuProgram);
    ret = clReleaseCommandQueue(cpuQueue);
    ret = clReleaseCommandQueue(gpuQueue);
    ret = clReleaseContext(cpuContext);
    ret = clReleaseContext(gpuContext);
}

} // namespace ocl

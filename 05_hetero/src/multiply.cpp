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

    size_t byteSize = n * n * sizeof(float);
    cl_mem aMem = clCreateBuffer(context, CL_MEM_READ_ONLY, byteSize, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, aMem, CL_TRUE, 0, byteSize, a, 0, nullptr, nullptr);
    cl_mem bMem = clCreateBuffer(context, CL_MEM_READ_ONLY, byteSize, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, bMem, CL_TRUE, 0, byteSize, b, 0, nullptr, nullptr);
    cl_mem cMem = clCreateBuffer(context, CL_MEM_READ_WRITE, byteSize, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, cMem, CL_TRUE, 0, byteSize, c, 0, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
    clSetKernelArg(kernel, 3, sizeof(int), &n);

    size_t globalWorkSize[] = {SAFE(n), SAFE(n)};
    size_t localWorkSize[] = {16u, 16u};
    float begin = omp_get_wtime();
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    clFinish(queue);
    float end = omp_get_wtime();
    if (elapsed != nullptr)
        *elapsed = end - begin;
    clEnqueueReadBuffer(queue, cMem, CL_TRUE, 0, byteSize, c, 0, nullptr, nullptr);

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
    cl_command_queue cpuQueue = clCreateCommandQueue(cpuContext, cpuDeviceId, CL_QUEUE_PROFILING_ENABLE, &ret);
    cl_command_queue gpuQueue = clCreateCommandQueue(gpuContext, gpuDeviceId, CL_QUEUE_PROFILING_ENABLE, &ret);

    std::string source = Utils::readFile(KERNELS_DIR "multiply.cl");
    const char *strings[] = {source.c_str()};
    cl_program cpuProgram = clCreateProgramWithSource(cpuContext, 1, strings, nullptr, &ret);
    ret = clBuildProgram(cpuProgram, 1, &cpuDeviceId, nullptr, nullptr, nullptr);
    cl_program gpuProgram = clCreateProgramWithSource(gpuContext, 1, strings, nullptr, &ret);
    ret = clBuildProgram(gpuProgram, 1, &gpuDeviceId, nullptr, nullptr, nullptr);
    cl_kernel cpuKernel = clCreateKernel(cpuProgram, "multiply", &ret);
    cl_kernel gpuKernel = clCreateKernel(gpuProgram, "multiply", &ret);

    size_t byteSize = n * n * sizeof(float);
    cl_mem aMemCpu = clCreateBuffer(cpuContext, CL_MEM_READ_ONLY, byteSize, nullptr, &ret);
    cl_mem aMemGpu = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, byteSize, nullptr, &ret);
    ret = clEnqueueWriteBuffer(cpuQueue, aMemCpu, CL_FALSE, 0, byteSize, a, 0, nullptr, nullptr);
    ret = clEnqueueWriteBuffer(gpuQueue, aMemGpu, CL_FALSE, 0, byteSize, a, 0, nullptr, nullptr);
    cl_mem bMemCpu = clCreateBuffer(cpuContext, CL_MEM_READ_ONLY, byteSize, nullptr, &ret);
    cl_mem bMemGpu = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, byteSize, nullptr, &ret);
    ret = clEnqueueWriteBuffer(cpuQueue, bMemCpu, CL_FALSE, 0, byteSize, b, 0, nullptr, nullptr);
    ret = clEnqueueWriteBuffer(gpuQueue, bMemGpu, CL_FALSE, 0, byteSize, b, 0, nullptr, nullptr);
    cl_mem cMemCpu = clCreateBuffer(cpuContext, CL_MEM_WRITE_ONLY, byteSize, nullptr, &ret);
    cl_mem cMemGpu = clCreateBuffer(gpuContext, CL_MEM_WRITE_ONLY, byteSize, nullptr, &ret);
    ret = clFinish(cpuQueue);
    ret = clFinish(gpuQueue);

    ret = clSetKernelArg(cpuKernel, 0, sizeof(cl_mem), &aMemCpu);
    ret = clSetKernelArg(cpuKernel, 1, sizeof(cl_mem), &bMemCpu);
    ret = clSetKernelArg(cpuKernel, 2, sizeof(cl_mem), &cMemCpu);
    ret = clSetKernelArg(cpuKernel, 3, sizeof(int), &n);

    ret = clSetKernelArg(gpuKernel, 0, sizeof(cl_mem), &aMemGpu);
    ret = clSetKernelArg(gpuKernel, 1, sizeof(cl_mem), &bMemGpu);
    ret = clSetKernelArg(gpuKernel, 2, sizeof(cl_mem), &cMemGpu);
    ret = clSetKernelArg(gpuKernel, 3, sizeof(int), &n);

    size_t cpuWorkSize[] = {SAFE(n), SAFE(delim)};
    size_t gpuWorkSize[] = {SAFE(n), SAFE(n - delim)};
    size_t cpuOffset[] = {SAFE(0), SAFE(0)};
    size_t gpuOffset[] = {SAFE(0), SAFE(delim)};
    size_t localWorkSize[] = {16u, 16u};

    cl_event events[2];

    ret = clEnqueueNDRangeKernel(cpuQueue, cpuKernel, 2, cpuOffset, cpuWorkSize, localWorkSize, 0, nullptr, events + 0);
    ret = clEnqueueNDRangeKernel(gpuQueue, gpuKernel, 2, gpuOffset, gpuWorkSize, localWorkSize, 0, nullptr, events + 1);
    clWaitForEvents(2, events);
    ret = clFinish(cpuQueue);
    ret = clFinish(gpuQueue);

    cl_ulong cpuTime[2], gpuTime[2];
    clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), cpuTime, nullptr);
    clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), cpuTime + 1, nullptr);
    clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), gpuTime, nullptr);
    clGetEventProfilingInfo(events[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), gpuTime + 1, nullptr);
    double times[2] = {0};
    times[0] = (cpuTime[1] - cpuTime[0]) / 1e9;
    times[1] = (gpuTime[1] - gpuTime[0]) / 1e9;
    if (elapsed != nullptr)
        *elapsed = times[0] > times[1] ? times[0] : times[1];

    ret = clEnqueueReadBuffer(cpuQueue, cMemCpu, CL_FALSE, 0, delim * n * sizeof(float), c, 0, nullptr, nullptr);
    ret = clEnqueueReadBuffer(gpuQueue, cMemGpu, CL_FALSE, delim * n * sizeof(float),
                              byteSize - delim * n * sizeof(float), c + delim * n, 0, nullptr, nullptr);
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

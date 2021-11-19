#include "multiply.hpp"

#include <string>

#include <omp.h>

#include "utils.hpp"

void multiply(float *a, float *b, float *c, int m, int n, int k) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            float *s = c + k * row + col;
            *s = 0;
            for (int i = 0; i < n; i++)
                *s += a[row * n + i] * b[col + k * i];
        }
    }
}

void multiply_omp(float *a, float *b, float *c, int m, int n, int k) {
#pragma omp parallel for
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            float *s = c + k * row + col;
            *s = 0;
            for (int i = 0; i < n; i++)
                *s += a[row * n + i] * b[col + k * i];
        }
    }
}

void multiply_ocl(float *a, float *b, float *c, int m, int n, int k, cl_device_id deviceId, float *elapsed) {
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    cl_mem aMem = nullptr;
    cl_mem bMem = nullptr;
    cl_mem cMem = nullptr;

    std::string source = Utils::readFile(KERNELS_DIR "multiply.cl");
    const char *strings[] = {source.c_str()};
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
    clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "multiply", nullptr);

    cl_int ret;
    aMem = clCreateBuffer(context, CL_MEM_READ_ONLY, m * n * sizeof(float), nullptr, &ret);
    ret = clEnqueueWriteBuffer(queue, aMem, CL_TRUE, 0, m * n * sizeof(float), a, 0, nullptr, nullptr);
    bMem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * k * sizeof(float), nullptr, &ret);
    ret = clEnqueueWriteBuffer(queue, bMem, CL_TRUE, 0, n * k * sizeof(float), b, 0, nullptr, nullptr);
    cMem = clCreateBuffer(context, CL_MEM_READ_WRITE, m * k * sizeof(float), nullptr, &ret);
    ret = clEnqueueWriteBuffer(queue, cMem, CL_TRUE, 0, m * k * sizeof(float), c, 0, nullptr, nullptr);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
    ret = clSetKernelArg(kernel, 3, sizeof(int), &m);
    ret = clSetKernelArg(kernel, 4, sizeof(int), &n);
    ret = clSetKernelArg(kernel, 5, sizeof(int), &k);

    size_t group;
    clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr);

    const size_t globalWorkSize[] = {static_cast<size_t>(m), static_cast<size_t>(k)};
    const size_t localWorkSize[] = {4u, 4u};
    float begin = omp_get_wtime();
    ret = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    clFinish(queue);
    float end = omp_get_wtime();
    if (elapsed != nullptr)
        *elapsed = end - begin;
    ret = clEnqueueReadBuffer(queue, cMem, CL_TRUE, 0, m * k * sizeof(float), c, 0, nullptr, nullptr);

    clReleaseMemObject(aMem);
    clReleaseMemObject(bMem);
    clReleaseMemObject(cMem);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

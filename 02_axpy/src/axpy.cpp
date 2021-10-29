#include "axpy.hpp"

#include <string>

#include <omp.h>

#include "utils.hpp"

#define AXPY_IMPL                                                                                                      \
    for (int i = 0; i < n; i++)                                                                                        \
        y[i * incy] += a * x[i * incx];

void saxpy(int n, float a, float *x, int incx, float *y, int incy) {
    AXPY_IMPL
}

void daxpy(int n, double a, double *x, int incx, double *y, int incy) {
    AXPY_IMPL
}

void saxpy_ocl(int n, float a, float *x, int incx, float *y, int incy, cl_device_id deviceId, double *elapsed) {
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    cl_mem xMem = nullptr;
    cl_mem yMem = nullptr;

    std::string source = Utils::readFile(KERNELS_DIR "axpy.cl");
    const char *strings[] = {source.c_str()};
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
    clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "saxpy", nullptr);

    xMem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * incx * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, xMem, CL_TRUE, 0, n * incx * sizeof(float), x, 0, nullptr, nullptr);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &xMem);

    yMem = clCreateBuffer(context, CL_MEM_READ_WRITE, n * incy * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, yMem, CL_TRUE, 0, n * incy * sizeof(float), y, 0, nullptr, nullptr);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &yMem);

    clSetKernelArg(kernel, 0, sizeof(int), &n);
    clSetKernelArg(kernel, 1, sizeof(float), &a);
    clSetKernelArg(kernel, 3, sizeof(int), &incx);
    clSetKernelArg(kernel, 5, sizeof(int), &incy);

    size_t group;
    clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr);

    const size_t globalWorkSize = static_cast<size_t>(n);
    const size_t localWorkSize = 1u;
    // clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, &group, 0, nullptr, nullptr);
    double begin = omp_get_wtime();
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
    clFinish(queue);
    double end = omp_get_wtime();
    if (elapsed != nullptr)
        *elapsed = end - begin;
    clEnqueueReadBuffer(queue, yMem, CL_TRUE, 0, n * incy * sizeof(float), y, 0, nullptr, nullptr);

    clReleaseMemObject(xMem);
    clReleaseMemObject(yMem);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

void daxpy_ocl(int n, double a, double *x, int incx, double *y, int incy, cl_device_id deviceId, double *elapsed) {
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    cl_mem xMem = nullptr;
    cl_mem yMem = nullptr;

    std::string source = Utils::readFile(KERNELS_DIR "axpy.cl");
    const char *strings[] = {source.c_str()};
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
    clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "daxpy", nullptr);

    xMem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * incx * sizeof(double), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, xMem, CL_TRUE, 0, n * incx * sizeof(double), x, 0, nullptr, nullptr);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &xMem);

    yMem = clCreateBuffer(context, CL_MEM_READ_WRITE, n * incy * sizeof(double), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, yMem, CL_TRUE, 0, n * incy * sizeof(double), y, 0, nullptr, nullptr);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &yMem);

    clSetKernelArg(kernel, 0, sizeof(int), &n);
    clSetKernelArg(kernel, 1, sizeof(double), &a);
    clSetKernelArg(kernel, 3, sizeof(int), &incx);
    clSetKernelArg(kernel, 5, sizeof(int), &incy);

    size_t group;
    clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr);

    const size_t globalWorkSize = static_cast<size_t>(n);
    const size_t localWorkSize = 1u;
    // clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, &group, 0, nullptr, nullptr);
    double begin = omp_get_wtime();
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
    clFinish(queue);
    double end = omp_get_wtime();
    if (elapsed != nullptr)
        *elapsed = end - begin;
    clEnqueueReadBuffer(queue, yMem, CL_TRUE, 0, n * incy * sizeof(double), y, 0, nullptr, nullptr);

    clReleaseMemObject(xMem);
    clReleaseMemObject(yMem);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

void saxpy_omp(int n, float a, float *x, int incx, float *y, int incy) {
#pragma omp parallel for
    AXPY_IMPL
}

void daxpy_omp(int n, double a, double *x, int incx, double *y, int incy) {
#pragma omp parallel for
    AXPY_IMPL
}

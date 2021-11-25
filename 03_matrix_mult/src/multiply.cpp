#include "multiply.hpp"

#include <omp.h>
#include <string>

#include "utils.hpp"

#define SAFE(X) (static_cast<size_t>(X))

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

namespace omp {

void multiply(float *a, float *b, float *c, int m, int n, int k) {
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

} // namespace omp

namespace ocl {

void transpose(float *c, int m, int k, float *cT) {
    for (int i = 0; i < k; i++)
        for (int j = 0; j < m; j++)
            cT[i * m + j] = c[j * k + i];
}

void multiply(float *a, float *b, float *c, int m, int n, int k, cl_device_id deviceId, float *elapsed) {
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    std::string source = Utils::readFile(KERNELS_DIR "multiply.cl");
    const char *strings[] = {source.c_str()};
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
    clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "multiply", nullptr);

    cl_mem aMem = clCreateBuffer(context, CL_MEM_READ_ONLY, m * n * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, aMem, CL_TRUE, 0, m * n * sizeof(float), a, 0, nullptr, nullptr);
    cl_mem bMem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * k * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, bMem, CL_TRUE, 0, n * k * sizeof(float), b, 0, nullptr, nullptr);
    cl_mem cMem = clCreateBuffer(context, CL_MEM_READ_WRITE, m * k * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, cMem, CL_TRUE, 0, m * k * sizeof(float), c, 0, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
    clSetKernelArg(kernel, 3, sizeof(int), &m);
    clSetKernelArg(kernel, 4, sizeof(int), &n);
    clSetKernelArg(kernel, 5, sizeof(int), &k);

    size_t globalWorkSize[] = {SAFE(m), SAFE(k)};
    size_t localWorkSize[] = {16u, 16u};
    float begin = omp_get_wtime();
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    clFinish(queue);
    float end = omp_get_wtime();
    if (elapsed != nullptr)
        *elapsed = end - begin;
    clEnqueueReadBuffer(queue, cMem, CL_TRUE, 0, m * k * sizeof(float), c, 0, nullptr, nullptr);

    clReleaseMemObject(aMem);
    clReleaseMemObject(bMem);
    clReleaseMemObject(cMem);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

void multiplyBlock(float *a, float *b, float *c, int m, int n, int k, cl_device_id deviceId, float *elapsed) {
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    std::string source = Utils::readFile(KERNELS_DIR "multiplyBlock.cl");
    const char *strings[] = {source.c_str()};
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
    clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "multiplyBlockOptimal", nullptr);

    cl_mem aMem = clCreateBuffer(context, CL_MEM_READ_ONLY, m * n * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, aMem, CL_TRUE, 0, m * n * sizeof(float), a, 0, nullptr, nullptr);
    cl_mem bMem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * k * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, bMem, CL_TRUE, 0, n * k * sizeof(float), b, 0, nullptr, nullptr);
    cl_mem cMem = clCreateBuffer(context, CL_MEM_READ_WRITE, m * k * sizeof(float), nullptr, nullptr);
    clEnqueueWriteBuffer(queue, cMem, CL_TRUE, 0, m * k * sizeof(float), c, 0, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
    clSetKernelArg(kernel, 3, sizeof(int), &m);
    clSetKernelArg(kernel, 4, sizeof(int), &n);
    clSetKernelArg(kernel, 5, sizeof(int), &k);

    size_t globalWorkSize[] = {SAFE(m), SAFE(k)};
    size_t localWorkSize[] = {16u, 16u};
    float begin = omp_get_wtime();
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    clFinish(queue);
    float end = omp_get_wtime();
    if (elapsed != nullptr)
        *elapsed = end - begin;
    clEnqueueReadBuffer(queue, cMem, CL_TRUE, 0, m * k * sizeof(float), c, 0, nullptr, nullptr);

    clReleaseMemObject(aMem);
    clReleaseMemObject(bMem);
    clReleaseMemObject(cMem);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

void multiplyImage(float *a, float *b, float *c, int m, int n, int k, cl_device_id deviceId, float *elapsed) {
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    std::string source = Utils::readFile(KERNELS_DIR "multiplyImage.cl");
    const char *strings[] = {source.c_str()};
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
    clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "multiplyImage", nullptr);

    cl_image_format format;
    format.image_channel_order = CL_R;
    format.image_channel_data_type = CL_FLOAT;
    size_t origin[] = {0, 0, 0};

    cl_mem aMem = clCreateImage2D(context, CL_MEM_READ_ONLY, &format, SAFE(m), SAFE(n), 0, nullptr, nullptr);
    cl_mem bMem = clCreateImage2D(context, CL_MEM_READ_ONLY, &format, SAFE(n), SAFE(k), 0, nullptr, nullptr);
    cl_mem cMem = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format, SAFE(m), SAFE(k), 0, nullptr, nullptr);
    {
        size_t region[] = {SAFE(m), SAFE(n), 1};
        clEnqueueWriteImage(queue, aMem, CL_TRUE, origin, region, 0, 0, a, 0, nullptr, nullptr);
    }
    {
        size_t region[] = {SAFE(n), SAFE(k), 1};
        clEnqueueWriteImage(queue, bMem, CL_TRUE, origin, region, 0, 0, b, 0, nullptr, nullptr);
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
    clSetKernelArg(kernel, 3, sizeof(int), &m);
    clSetKernelArg(kernel, 4, sizeof(int), &n);
    clSetKernelArg(kernel, 5, sizeof(int), &k);

    size_t globalWorkSize[] = {SAFE(m), SAFE(k)};
    size_t localWorkSize[] = {16u, 16u};
    float begin = omp_get_wtime();
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    clFinish(queue);
    float end = omp_get_wtime();
    if (elapsed != nullptr)
        *elapsed = end - begin;
    {
        size_t region[] = {SAFE(m), SAFE(k), 1};
        clEnqueueReadImage(queue, cMem, CL_TRUE, origin, region, 0, 0, c, 0, nullptr, nullptr);
    }

    clReleaseMemObject(aMem);
    clReleaseMemObject(bMem);
    clReleaseMemObject(cMem);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

} // namespace ocl

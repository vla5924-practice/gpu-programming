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

namespace ocl {

void transpose(float *c, int m, int k, float *cT) {
    for (int i = 0; i < k; i++)
        for (int j = 0; j < m; j++)
            cT[i * m + j] = c[j * k + i];
}

void multiply(float *a, float *b, float *c, int m, int n, int k, cl_device_id deviceId, float *elapsed) {
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

    const size_t globalWorkSize[] = {SAFE(m), SAFE(k)};
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

void multiplyBlock(float *a, float *b, float *c, int m, int n, int k, cl_device_id deviceId, float *elapsed) {
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    cl_mem aMem = nullptr;
    cl_mem bMem = nullptr;
    cl_mem cMem = nullptr;

    cl_int ret;
    std::string source = Utils::readFile(KERNELS_DIR "multiplyBlock.cl");
    const char *strings[] = {source.c_str()};
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, &ret);
    ret = clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "multiplyBlockOptimal", &ret);

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

    const size_t globalWorkSize[] = {SAFE(m), SAFE(k)};
    const size_t localWorkSize[] = {16u, 16u};
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

void multiplyImage(float *a, float *b, float *c, int m, int n, int k, cl_device_id deviceId, float *elapsed) {
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    cl_mem aMem = nullptr;
    cl_mem bMem = nullptr;
    cl_mem cMem = nullptr;

    cl_int ret;
    std::string source = Utils::readFile(KERNELS_DIR "multiplyImage.cl");
    const char *strings[] = {source.c_str()};
    cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, &ret);
    ret = clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    if (ret == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *)malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
    }
    cl_kernel kernel = clCreateKernel(program, "multiplyImageNaive", &ret);

    cl_image_format format;
    format.image_channel_order = CL_R;
    format.image_channel_data_type = CL_FLOAT;
    size_t origin[] = {0, 0, 0};

    {
        aMem = clCreateImage2D(context, CL_MEM_READ_ONLY, &format, SAFE(m), SAFE(n), 0, nullptr, &ret);
        size_t region[] = {SAFE(m), SAFE(n), 1};
        ret = clEnqueueWriteImage(queue, aMem, CL_TRUE, origin, region, 0, 0, a, 0, nullptr, nullptr);
    }
    {
        bMem = clCreateImage2D(context, CL_MEM_READ_ONLY, &format, SAFE(n), SAFE(k), 0, nullptr, &ret);
        size_t region[] = {SAFE(n), SAFE(k), 1};
        ret = clEnqueueWriteImage(queue, bMem, CL_TRUE, origin, region, 0, 0, b, 0, nullptr, nullptr);
    }
    cMem = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format, SAFE(m), SAFE(k), 0, nullptr, &ret);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
    ret = clSetKernelArg(kernel, 3, sizeof(int), &m);
    ret = clSetKernelArg(kernel, 4, sizeof(int), &n);
    ret = clSetKernelArg(kernel, 5, sizeof(int), &k);

    const size_t globalWorkSize[] = {SAFE(m), SAFE(k)};
    const size_t localWorkSize[] = {16u, 16u};
    float begin = omp_get_wtime();
    ret = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    ret = clFinish(queue);
    float end = omp_get_wtime();
    if (elapsed != nullptr)
        *elapsed = end - begin;
    {
        size_t region[] = {SAFE(m), SAFE(k), 1};
        ret = clEnqueueReadImage(queue, cMem, CL_TRUE, origin, region, 0, 0, c, 0, nullptr, nullptr);
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

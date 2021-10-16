#ifndef KERNELS_DIR
#define KERNELS_DIR
#endif

#include <iostream>
#include <vector>

#include <CL/cl.h>

#include "utils.hpp"

int main() {
    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);
    cl_platform_id *platform = new cl_platform_id[platformCount];
    clGetPlatformIDs(platformCount, platform, nullptr);
    for (cl_uint i = 0; i < platformCount; i++) {
        constexpr size_t maxLength = 128;
        char platformName[maxLength];
        clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, maxLength, platformName, nullptr);
        std::cout << platformName << std::endl;
    }

    cl_device_id deviceId = 0;
    cl_uint deviceCount = 0;
    clGetDeviceIDs(platform[1], CL_DEVICE_TYPE_GPU, 1, &deviceId, &deviceCount);
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, nullptr);

    {
        std::string source = Utils::readFile(KERNELS_DIR "whoAmI.cl");
        const char *strings[] = {source.c_str()};
        cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
        clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
        cl_kernel kernel = clCreateKernel(program, "whoAmI", nullptr);
        constexpr size_t globalWorkSize = 10;
        constexpr size_t localWorkSize = 5;
        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);

        clReleaseKernel(kernel);
        clReleaseProgram(program);
    }

    {
        constexpr size_t elemCount = 10;
        std::vector<cl_uint> arr(elemCount, cl_uint(100));
        cl_mem memory = nullptr;

        std::string source = Utils::readFile(KERNELS_DIR "calculateArray.cl");
        const char *strings[] = {source.c_str()};
        cl_program program = clCreateProgramWithSource(context, 1, strings, nullptr, nullptr);
        clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
        cl_kernel kernel = clCreateKernel(program, "calculateArray", nullptr);

        memory = clCreateBuffer(context, CL_MEM_READ_WRITE, elemCount * sizeof(cl_int), nullptr, nullptr);
        clEnqueueWriteBuffer(queue, memory, CL_TRUE, 0, elemCount * sizeof(cl_int), arr.data(), 0, nullptr, nullptr);
        clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memory);

        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &elemCount, nullptr, 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, memory, CL_TRUE, 0, elemCount * sizeof(cl_int), arr.data(), 0, nullptr, nullptr);

        Utils::print(arr);

        clReleaseMemObject(memory);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
    }

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    delete[] platform;
}

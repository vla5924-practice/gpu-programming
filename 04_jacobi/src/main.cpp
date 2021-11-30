#ifndef KERNELS_DIR
#define KERNELS_DIR
#endif

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <CL/cl.h>

#include "jacobi.hpp"
#include "utils.hpp"

int main() {
    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);
    cl_platform_id *platform = new cl_platform_id[platformCount];
    clGetPlatformIDs(platformCount, platform, nullptr);

    cl_device_id cpuDeviceId = 0;
    cl_uint deviceCount = 0;
    clGetDeviceIDs(platform[1], CL_DEVICE_TYPE_CPU, 1, &cpuDeviceId, &deviceCount);
    char deviceName[128] = {0};
    clGetDeviceInfo(cpuDeviceId, CL_DEVICE_NAME, 128, deviceName, nullptr);
    std::cout << "CPU: " << deviceName << std::endl;
    cl_device_id gpuDeviceId = 0;
    clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU, 1, &gpuDeviceId, &deviceCount);
    clGetDeviceInfo(gpuDeviceId, CL_DEVICE_NAME, 128, deviceName, nullptr);
    std::cout << "GPU: " << deviceName << std::endl;

    constexpr int n = 4500;
    constexpr int iter = 500;
    constexpr float epsilon = 1e-6;

    std::vector<float> a(n * n);
    std::vector<float> b(n);
    Utils::fillRandomly(a);
    Utils::fillRandomly(b);

    {
        std::random_device rd;
        std::mt19937 mersenne(rd());
        std::uniform_real_distribution<> urd(n * 2.5, n * 2.5 + 2.0);
        for (size_t i = 0; i < n; i++)
            a[i * n + i] = urd(mersenne);
    }
    std::cout << std::defaultfloat << std::setprecision(6);

    {
        std::vector<float> x(n, 0);
        CompResults results = jacobi(a.data(), b.data(), x.data(), n, iter, epsilon, cpuDeviceId);
        std::cout << "------" << std::endl;
        std::cout << "OpenCL CPU" << std::endl;
        std::cout << "Iterations: " << results.iter << std::endl;
        std::cout << "Kernel time: " << results.kernelTime << std::endl;
        std::cout << "Full time: " << results.fullTime << std::endl;
        std::cout << "Convergency norm: " << results.convNorm << std::endl;
        std::cout << "Deviation: " << deviation(a.data(), b.data(), x.data(), n) << std::endl;
    }
    {
        std::vector<float> x(n, 0);
        CompResults results = jacobi(a.data(), b.data(), x.data(), n, iter, epsilon, gpuDeviceId);
        std::cout << "------" << std::endl;
        std::cout << "OpenCL GPU" << std::endl;
        std::cout << "Iterations: " << results.iter << std::endl;
        std::cout << "Kernel time: " << results.kernelTime << std::endl;
        std::cout << "Full time: " << results.fullTime << std::endl;
        std::cout << "Convergency norm: " << results.convNorm << std::endl;
        std::cout << "Deviation: " << deviation(a.data(), b.data(), x.data(), n) << std::endl;
    }

    delete[] platform;
}

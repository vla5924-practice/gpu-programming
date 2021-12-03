#ifndef KERNELS_DIR
#define KERNELS_DIR
#endif

#include <iostream>
#include <vector>

#include <CL/cl.h>
#include <omp.h>

#include "multiply.hpp"
#include "utils.hpp"

int main() {
#pragma omp parallel
    {
#pragma omp single
        std::cout << "OpenMP: " << omp_get_num_threads() << " threads" << std::endl;
    }

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
    clGetDeviceIDs(platform[1], CL_DEVICE_TYPE_GPU, 1, &gpuDeviceId, &deviceCount);
    clGetDeviceInfo(gpuDeviceId, CL_DEVICE_NAME, 128, deviceName, nullptr);
    std::cout << "GPU: " << deviceName << std::endl;

    constexpr int m = 1600;
    constexpr int n = 1600;
    constexpr int k = 1600;

    std::vector<float> a(m * n);
    std::vector<float> b(n * k);
    std::vector<float> cTarget(m * k);
    Utils::fillRandomly(a);
    Utils::fillRandomly(b);
    std::cout << std::defaultfloat << std::setprecision(6);

    {
        float begin = omp_get_wtime();
        multiply(a.data(), b.data(), cTarget.data(), n);
        float end = omp_get_wtime();
        std::cout << "Sequential: " << (end - begin) << std::endl;
    }
    {
        std::vector<float> c(m * k, 0);
        float elapsed = 0;
        ocl::multiply(a.data(), b.data(), c.data(), n, cpuDeviceId, &elapsed);
        std::cout << "OpenCL CPU: " << elapsed << ' ';
        std::cout << Utils::status(Utils::equals(c, cTarget)) << std::endl;
    }
    {
        std::vector<float> c(m * k, 0);
        float elapsed = 0;
        ocl::multiply(a.data(), b.data(), c.data(), n, gpuDeviceId, &elapsed);
        std::cout << "OpenCL GPU: " << elapsed << ' ';
        std::cout << Utils::status(Utils::equals(c, cTarget)) << std::endl;
    }
    {
        std::vector<float> c(m * k, 0);
        float elapsed = 0;
        ocl::multiplyHetero(a.data(), b.data(), c.data(), n, 50 * 16, cpuDeviceId, gpuDeviceId, &elapsed);
        std::cout << "OpenCL CPU+GPU: " << elapsed << ' ';
        std::cout << Utils::status(Utils::equals(c, cTarget)) << std::endl;
    }

    delete[] platform;
}

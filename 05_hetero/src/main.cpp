#ifndef KERNELS_DIR
#define KERNELS_DIR
#endif

#include <iostream>
#include <vector>

#include <CL/cl.h>
#include <omp.h>

#include "jacobi.hpp"
#include "multiply.hpp"
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
    std::cout << "------" << std::endl;
    {
        constexpr int n = 3200;

        std::vector<float> a(n * n);
        std::vector<float> b(n * n);
        Utils::fillRandomly(a);
        Utils::fillRandomly(b);
        std::cout << std::defaultfloat << std::setprecision(6);
        {
            std::vector<float> c(n * n, 0);
            float elapsed = 0;
            ocl::multiply(a.data(), b.data(), c.data(), n, cpuDeviceId, &elapsed);
            std::cout << "OpenCL CPU: " << elapsed << std::endl;
        }
        {
            std::vector<float> c(n * n, 0);
            float elapsed = 0;
            ocl::multiply(a.data(), b.data(), c.data(), n, gpuDeviceId, &elapsed);
            std::cout << "OpenCL GPU: " << elapsed << std::endl;
        }
        {
            std::vector<float> c(n * n, 0);
            float elapsed = 0;
            ocl::multiplyHetero(a.data(), b.data(), c.data(), n, 16 * 16, cpuDeviceId, gpuDeviceId, &elapsed);
            std::cout << "OpenCL CPU+GPU: " << elapsed << std::endl;
        }
    }
    std::cout << "------" << std::endl;
    {
        constexpr int n = 4800;
        constexpr int iter = 500;
        constexpr float convThreshold = 1e-6;

        std::vector<float> a(n * n);
        std::vector<float> b(n);
        Utils::fillRandomly(a);
        Utils::fillRandomly(b);

        {
            std::random_device rd;
            std::mt19937 mersenne(rd());
            std::uniform_real_distribution<> urd(n * 4.0, n * 4.0 + 2.0);
            for (size_t i = 0; i < n; i++)
                a[i * n + i] = urd(mersenne);
        }
        std::cout << std::defaultfloat << std::setprecision(6);

        {
            std::vector<float> x(n, 0);
            CompResults results = jacobi(a.data(), b.data(), x.data(), n, iter, convThreshold, cpuDeviceId);
            std::cout << "OpenCL CPU:     " << results.kernelTime << ", iters: " << results.iter
                      << ", full time: " << results.fullTime << ", conv norm: " << results.convNorm
                      << ", deviation: " << deviation(a.data(), b.data(), x.data(), n) << std::endl;
        }
        {
            std::vector<float> x(n, 0);
            CompResults results = jacobi(a.data(), b.data(), x.data(), n, iter, convThreshold, gpuDeviceId);
            std::cout << "OpenCL GPU:     " << results.kernelTime << ", iters: " << results.iter
                      << ", full time: " << results.fullTime << ", conv norm: " << results.convNorm
                      << ", deviation: " << deviation(a.data(), b.data(), x.data(), n) << std::endl;
        }
        {
            std::vector<float> x(n, 0);
            CompResults results =
                jacobiHetero(a.data(), b.data(), x.data(), n, iter, convThreshold, 400, cpuDeviceId, gpuDeviceId);
            std::cout << "OpenCL CPU+GPU: " << results.kernelTime << ", iters: " << results.iter
                      << ", full time: " << results.fullTime << ", conv norm: " << results.convNorm
                      << ", deviation: " << deviation(a.data(), b.data(), x.data(), n) << std::endl;
        }
    }

    delete[] platform;
}

#ifndef KERNELS_DIR
#define KERNELS_DIR
#endif

#include <iostream>
#include <vector>

#include <CL/cl.h>
#include <omp.h>

#include "axpy.hpp"
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
    clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU, 1, &gpuDeviceId, &deviceCount);
    clGetDeviceInfo(gpuDeviceId, CL_DEVICE_NAME, 128, deviceName, nullptr);
    std::cout << "GPU: " << deviceName << std::endl;

    constexpr int n = 100'000'000;
    constexpr int incy = 2;
    constexpr int incx = 3;
    constexpr size_t ySize = static_cast<size_t>(n * incy);
    constexpr size_t xSize = static_cast<size_t>(n * incx);
    constexpr float a = 4;

    {
        std::cout << "---\nSingle-precision\n";

        std::vector<float> xInit(xSize, 0.f);
        Utils::fillWithStride(xInit, 1.f, incx);
        // Utils::print(xInit);

        std::vector<float> yInit(ySize, 0.f);
        Utils::fillWithStride(yInit, 2.f, incy);
        // Utils::print(yInit);

        std::vector<float> yTarget;

        {
            std::cout << "Sequential ";

            auto y = yInit;
            double begin = omp_get_wtime();
            saxpy(n, a, xInit.data(), incx, y.data(), incy);
            double end = omp_get_wtime();
            std::cout << (end - begin) << std::endl;
            yTarget = y;
        }

        {
            std::cout << "OpenMP ";

            auto y = yInit;
            double begin = omp_get_wtime();
            saxpy_omp(n, a, xInit.data(), incx, y.data(), incy);
            double end = omp_get_wtime();
            std::cout << (end - begin) << ' ';
            std::cout << Utils::status(y == yTarget) << std::endl;
        }

        {
            std::cout << "OpenCL CPU ";

            auto y = yInit;
            double elapsed = 0;
            saxpy_ocl(n, a, xInit.data(), incx, y.data(), incy, cpuDeviceId, &elapsed);
            std::cout << elapsed << ' ';
            std::cout << Utils::status(y == yTarget) << std::endl;
        }

        {
            std::cout << "OpenCL GPU ";

            auto y = yInit;
            double elapsed = 0;
            saxpy_ocl(n, a, xInit.data(), incx, y.data(), incy, gpuDeviceId, &elapsed);
            std::cout << elapsed << ' ';
            std::cout << Utils::status(y == yTarget) << std::endl;
        }
    }

    {
        std::cout << "---\nDouble-precision\n";

        std::vector<double> xInit(xSize, 0.);
        Utils::fillWithStride(xInit, 1., incx);
        // Utils::print(xInit);

        std::vector<double> yInit(ySize, 0.);
        Utils::fillWithStride(yInit, 2., incy);
        // Utils::print(yInit);

        std::vector<double> yTarget;

        {
            std::cout << "Sequential ";

            auto y = yInit;
            double begin = omp_get_wtime();
            daxpy(n, a, xInit.data(), incx, y.data(), incy);
            double end = omp_get_wtime();
            std::cout << (end - begin) << std::endl;
            yTarget = y;
        }

        {
            std::cout << "OpenMP ";

            auto y = yInit;
            double begin = omp_get_wtime();
            daxpy_omp(n, a, xInit.data(), incx, y.data(), incy);
            double end = omp_get_wtime();
            std::cout << (end - begin) << ' ';
            std::cout << Utils::status(y == yTarget) << std::endl;
        }

        {
            std::cout << "OpenCL CPU ";

            auto y = yInit;
            double elapsed = 0;
            daxpy_ocl(n, a, xInit.data(), incx, y.data(), incy, cpuDeviceId, &elapsed);
            std::cout << elapsed << ' ';
            std::cout << Utils::status(y == yTarget) << std::endl;
        }

        {
            std::cout << "OpenCL GPU ";

            auto y = yInit;
            double elapsed = 0;
            daxpy_ocl(n, a, xInit.data(), incx, y.data(), incy, gpuDeviceId, &elapsed);
            std::cout << elapsed << ' ';
            std::cout << Utils::status(y == yTarget) << std::endl;
        }
    }

    delete[] platform;
}

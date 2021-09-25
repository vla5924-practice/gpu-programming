# Лабораторная работа №1 “Hello, World”

**Задание:** используя интерфейс OpenCL, вывести список доступных платформ с помощью следующего кода:

```cpp
#include <CL/cl.h>
#include <iostream>

int main() {
    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);
    cl_platform_id *platform = new cl_platform_id[platformCount];
    clGetPlatformIDs(platformCount, platform, nullptr);
    for (cl_uint i = 0; i < platformCount; i++) {
        char platformName[128];
        clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 128, platformName, nullptr);
        std::cout << platformName << std::endl;
    }
}
```

**Дополнительные задания:**

1. Написать программу на GPU, которая выводит на консоль следующую фразу:
   `I am from N block, M thread (global index: K)`
1. Скопировать на GPU массив целых чисел `a[]`, каждый поток должен вычислить `а[i] = a[i] + ThreadGlobalIndex`, затем массив `a` нужно скопировать обратно на CPU и вывести на консоль.

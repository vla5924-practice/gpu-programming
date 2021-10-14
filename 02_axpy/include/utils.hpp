#pragma once

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace Utils {

std::string readFile(const std::string &path);

template <typename T>
void print(const std::vector<T> &arr) {
    for (const T &elem : arr)
        std::cout << elem << ' ';
    std::cout << std::endl;
}

template <typename T>
void fillWithStride(std::vector<T> &arr, const T &value, size_t stride) {
    size_t size = arr.size();
    for (size_t i = 0; i < size; i += stride)
        arr[i] = value;
}

std::string status(bool ok);

} // namespace Utils

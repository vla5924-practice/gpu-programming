#pragma once

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

} // namespace Utils

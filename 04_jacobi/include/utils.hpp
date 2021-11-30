#pragma once

#include <iostream>
#include <random>
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
void fillRandomly(std::vector<T> &arr) {
    std::random_device rd;
    std::mt19937 mersenne(rd());
    std::uniform_real_distribution<> urd(1.0, 3.0);
    size_t size = arr.size();
    for (T &el : arr)
        el = urd(mersenne);
}

} // namespace Utils

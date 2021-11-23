#include "utils.hpp"

#include <fstream>
#include <iostream>
#include <limits>

std::string Utils::readFile(const std::string &path) {
    std::ifstream file(path);
    std::string content, str;
    while (std::getline(file, str)) {
        content += str;
        content.push_back('\n');
    }
    return content;
}

std::string Utils::status(bool ok) {
    if (ok)
        return "OK";
    return "FAIL";
}

bool Utils::equals(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); i++)
        if (std::abs(a[i] - b[i]) >= 1e-3f)
            return false;
    return true;
}

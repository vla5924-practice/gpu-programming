#include "utils.hpp"

#include <fstream>
#include <iostream>

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

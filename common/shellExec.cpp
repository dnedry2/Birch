#include "shellExec.hpp"

#ifdef _WIN64
#include <Windows.h>
#elif __linux__
#include <stdlib.h>
#endif

#include <string>
#include <algorithm>

void OpenWebpage(const char* url) {
    using std::string;

    // Sanitize url
    auto str = string(url);

    str.erase(std::remove(str.begin(), str.end(), '"'), str.end());
    str.erase(std::remove(str.begin(), str.end(), '\''), str.end());
    str.erase(std::remove(str.begin(), str.end(), '`'), str.end());
    str.erase(std::remove(str.begin(), str.end(), '\\'), str.end());
    str.erase(std::remove(str.begin(), str.end(), ';'), str.end());
    str.erase(std::remove(str.begin(), str.end(), '|'), str.end());
    str.erase(std::remove(str.begin(), str.end(), '>'), str.end());
    str.erase(std::remove(str.begin(), str.end(), '<'), str.end());
    str.erase(std::remove(str.begin(), str.end(), '$'), str.end());
    str.erase(std::remove(str.begin(), str.end(), '('), str.end());
    str.erase(std::remove(str.begin(), str.end(), ')'), str.end());
    str.erase(std::remove(str.begin(), str.end(), '~'), str.end());
    str.erase(std::remove(str.begin(), str.end(), '{'), str.end());
    str.erase(std::remove(str.begin(), str.end(), '}'), str.end());
    str.erase(std::remove(str.begin(), str.end(), '['), str.end());
    str.erase(std::remove(str.begin(), str.end(), ']'), str.end());
    str.erase(std::remove(str.begin(), str.end(), '^'), str.end());
    str.erase(std::remove(str.begin(), str.end(), '@'), str.end());
    str.erase(std::remove(str.begin(), str.end(), '!'), str.end());
    str.erase(std::remove(str.begin(), str.end(), ','), str.end());
    str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
    str.erase(remove_if(str.begin(),str.end(), [](char c){ return !(c >= 32 && c < 123); }), str.end());  

    str = "\"" + str + "\"";

#ifdef _WIN64
    ShellExecute(NULL, NULL, url, NULL, NULL, SW_SHOW);
#elif __linux__
    if (!system(NULL))
        return;

    str = "xdg-open " + str;

    system(str.c_str());
#endif
}
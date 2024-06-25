#ifndef __TOOLTIP_H__
#define __TOOLTIP_H__

#include <map>
#include "stopwatch.hpp"

class Tooltip {
public:
    void Render(const char* text, float delay = 0.5f);
    void Render(void (*func)(void* data), void* data = nullptr, float delay = 0.5f);
    void Render(const void* id, const char* text, float delay = 0.5f);
    void Render(const void* id, void (*func)(void* data), void* data = nullptr, float delay = 0.5f);
private:
    std::map<const void*, Stopwatch> watches;
};

#endif
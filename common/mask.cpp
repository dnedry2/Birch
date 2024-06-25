#include "mask.hpp"
#include <cstring>

Mask::Mask(unsigned width, unsigned height, const Birch::Timespan& time)
    : width(width), height(height), time(time)
{
    data = new unsigned char[width * height];
    std::fill_n(data, width * height, 255);
}
Mask::~Mask()
{
    delete[] data;
}

unsigned Mask::Width() const
{
    return width;
}
unsigned Mask::Height() const
{
    return height;
}
unsigned char* Mask::Data() const
{
    return data;
}
const Birch::Timespan& Mask::Time() const
{
    return time;
}

void Mask::SetTime(const Birch::Timespan& time)
{
    this->time = time;
}
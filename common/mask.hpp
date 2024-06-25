#ifndef __MASK_H__
#define __MASK_H__

#include "include/birch.h"

class Mask {
public:
    unsigned Width() const;
    unsigned Height() const;
    unsigned char* Data() const;
    const Birch::Timespan& Time() const;

    void SetTime(const Birch::Timespan& time);

    unsigned char& operator[](unsigned index) { return data[index]; }

    Mask(unsigned width, unsigned height, const Birch::Timespan& time);
    ~Mask();

private:
    unsigned width;
    unsigned height;
    unsigned char* data;
    Birch::Timespan time;
};

#endif
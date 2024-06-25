#ifndef _FILTERSTACK_HPP_
#define _FILTERSTACK_HPP_

#include <vector>
#include "filter.hpp"
#include "defines.h"

class FilterStack {
public:
    const std::vector<Filter*>* Filters() const;
    void Add(Filter* filter);
    void Remove(Filter* filter);
    uint Count() const;

    bool  IsLinked() const;
    bool* LinkPtr();

    void GenerateMask(PlotField* field, bool linkIQ);

    FilterStack();
    ~FilterStack();
private:
    std::vector<Filter*> filters;
    bool link = false;

    void TBDIQFilter(PlotField* tbdpd);
};

#endif
#include "filterstack.hpp"
#include "plot.hpp"
#include "logger.hpp"

#include <algorithm>

FilterStack::FilterStack() {
    filters = std::vector<Filter*>();
}
FilterStack::~FilterStack() {
    for (Filter* filter : filters) {
        delete filter;
    }
}

const std::vector<Filter*>* FilterStack::Filters() const {
    return &filters;
}
void FilterStack::Add(Filter* filter) {
    // Erase any overlapping filter when adding a new one
    auto overlaps = [&](Filter* f) { return filter->Overlaps(f); };
    filters.erase(std::remove_if(filters.begin(), filters.end(), overlaps), filters.end());

    filters.push_back(filter);
}
void FilterStack::Remove(Filter* filter) {
    for (int i = 0; i < filters.size(); i++) {
        if (filters[i] == filter) {
            filters.erase(filters.begin() + i);
            return;
        }
    }
}
uint FilterStack::Count() const {
    return filters.size();
}

bool FilterStack::IsLinked() const {
    return link;
}
bool* FilterStack::LinkPtr() {
    return &link;
}

void FilterStack::GenerateMask(PlotField* field, bool linkIQ) {
    // Filters with lower precidence are applied first
    auto compareFilter = [](Filter* a, Filter* b) { return a->Get_Precedence() < b->Get_Precedence(); };
    std::sort(filters.begin(), filters.end(), compareFilter);

    bool* mask =  field->FilterMask;
    int   size = *field->ElementCount;

    bool first    = true;
    bool lastPass = true;

    // A list of which field last set a bit to true
    // Needed to combine multiple pass filters on the same field
    double *volatile **mfr = nullptr;
    try {
        mfr = new double* volatile*[size];
    } catch (...) {
        DispError("FilterStack::GenerateMask", "%s", "Failed to allocate memory for bitmask reference");
        return;
    }

    // Reset
    if (filters.size() == 0) {
        std::fill_n(mask, size, true);
    } else {
        std::fill_n(mask, size, false);
        std::fill_n(mfr,  size, nullptr);

        // Apply filters
        for (auto filter : filters)
            filter->Apply(field, (void*)&mfr, &first, &lastPass);
    }

    // TODO: associate 6k fields
    if (linkIQ && !strcmp("PD", field->Name))
        TBDIQFilter(field);

    delete[] mfr;
}

void FilterStack::TBDIQFilter(PlotField* tbdpd) {
    auto bfile = reinterpret_cast<TBDFile*>(tbdpd->File);

    if (bfile->AssocSig() == nullptr) {
        DispWarning("TBDIQFilter", "No associated signal file");
        return;
    }


    SignalCDIF* iq        = bfile->AssocSig()->Data();
    bool* const mask      = bfile->AssocSig()->TOAMask();
    const auto  elCount   = *iq->ElementCount();
    const auto  sampRate  = 1.0 / iq->SampleInterval();
    const auto  startTime = iq->TOA()[0];
    const auto  stride    = iq->Stride();
    const auto  pdCount   = *tbdpd->ElementCount;
    const uint  mult      = stride == 1 ? 1 : 2;

    std::fill_n(mask, elCount, false);

    #pragma omp parallel for
    for (ulong i = 0; i < pdCount; i++) {
        if (!tbdpd->FilterMask[i])
            continue;

        const double pd_s = tbdpd->Data[i] / 1000000.0;

        // Extend to 5% of pulse width on either side
        const double edge = pd_s * 0.05;

        ulong startEl   = floor((((tbdpd->Timing[i] - startTime - edge)) * sampRate) / stride) * mult;
        uint  pdElCount = ceil((pd_s + edge * 2) * sampRate / stride) * mult;


        // Stop if pulse is completely outside of view range
        if (startEl >= elCount || startEl + pdElCount <= 0)
            continue;


        // Must be at least 2 elements for lines to be drawn
        if (pdElCount < 2)
            pdElCount = 2;
        

        // Truncate to view range
        if (startEl + pdElCount >= elCount)
            pdElCount = (startEl + pdElCount) - elCount;
        
        if (startEl <= 0) {
            pdElCount += startEl;
            startEl = 1;
        }

        //DispDebug("El: %ld->%ld", startEl, startEl + pdElCount);

        // Subtract 1 in order to render the first line
        std::fill_n(mask + startEl - 1, pdElCount, true);
    }

    //std::fill_n(mask, elCount, true);
}
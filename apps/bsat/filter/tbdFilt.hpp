#ifndef _TBD_FILT_HPP_
#define _TBD_FILT_HPP_

#include "filter.hpp"

// Basic pass / reject filter for pdws
class TBDFilter : public Filter {
public:
    bool RenderWidget(int id, bool odd, bool range) override;
    void RenderPlotPreview(int axis) override;
    void RenderPlotPreviewSmall(int axis) override;
    void Apply(PlotField* field, void* mfr, bool* first, bool* lastPass) override;
    double Get_Min() { return lowLimit; }
    double Get_Max() { return highLimit; }
    bool Overlaps(Filter* filter) override;
    FilterType Type() override { return FilterType::TBD; }
    bool IsFieldReference(PlotField* field) override;

    int Get_Precedence() override { return !pass; }
    void* volatile* Get_Reference() override { return (void* volatile*)reference; }

    TBDFilter(double* volatile* reference, char* referenceName, double lowLimit, double highLimit, bool pass);
private:
    double* volatile* reference;

    double lowLimit;
    double highLimit;

    bool pass = true;
};

class TBDMatchFilter : public Filter {
public:
    bool RenderWidget(int id, bool odd, bool range) override;
    void RenderPlotPreview(int axis) override;
    void RenderPlotPreviewSmall(int axis) override;
    void Apply(PlotField* field, void* mfr, bool* first, bool* lastPass) override;
    bool Overlaps(Filter* filter) override;
    FilterType Type() override { return FilterType::TBD; }
    bool IsFieldReference(PlotField* field) override;

    int Get_Precedence() override { return 9; }
    void* volatile* Get_Reference() override { return (void* volatile*)peaks; }

    TBDMatchFilter(double* volatile* peaks, double* volatile* pd, volatile unsigned long* volatile* peakCount, char* referenceName);
    ~TBDMatchFilter();
private:
    double* volatile* peaks;
    double* volatile* pd;
    volatile unsigned long* volatile* peakCount;
};

#endif
#ifndef _FIR_FILTER_HPP_
#define _FIR_FILTER_HPP_

#include <vector>

#include "filter.hpp"
#include "fir.hpp"
#include "window.hpp"
#include "cdif.hpp"
#include "birch.h"

class FIRFilter : public Filter {
public:
    bool RenderWidget(int id, bool odd, bool range) override;
    void RenderPlotPreview(int axis) override;
    void RenderPlotPreviewSmall(int axis) override;
    void Apply(PlotField* field, void* mfr, bool* first, bool* lastPass) override;
    bool Overlaps(Filter* filter) override;
    FilterType Type() override { return FilterType::IQ; }
    bool IsFieldReference(PlotField* field) override;

    int Get_Precedence() override;
    void* volatile* Get_Reference() override;

    BandpassFIR* Get_FIR() { return fir; }

    FIRFilter(SignalCDIF* reference, PlotField* spectra, double sigBW, double filtBW, double filtCenter, unsigned taps, const char* name);
    ~FIRFilter();
private:
    BandpassFIR* fir = nullptr;
    WindowFunc* window = nullptr;

    SignalCDIF* ref;
    PlotField* spectra;

    float* preview     = nullptr;
    float* response    = nullptr;
    float* previewFilt = nullptr;
    float* design      = nullptr;
    float* winPrev     = nullptr;
    float* winPrev2    = nullptr;

    double sigBW;

    bool pass = true;

    bool bandMode = true;
    std::vector<Birch::Span<double>>  bands  = std::vector<Birch::Span<double>>();
    std::vector<Birch::Point<double>> points = std::vector<Birch::Point<double>>();
};

#endif
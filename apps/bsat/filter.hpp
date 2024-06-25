#include <vector>
#include <GL/gl3w.h>

#ifndef _FILTER_
#define _FILTER_

#include "defines.h"
#include "decs.hpp"
#include "fir.hpp"
#include "window.hpp"
#include "cdif.hpp"
#include <vector>

void InitFilterWidgets(GLuint show, GLuint hide);

enum class FilterType { TBD, IQ };

class Filter {
public:
    virtual bool RenderWidget(int id, bool odd, bool range) = 0;
    virtual void RenderPlotPreview(int axis) = 0;
    virtual void RenderPlotPreviewSmall(int axis) = 0;
    virtual void Apply(PlotField* field, void* mfr, bool* first, bool* lastPass) = 0;
    virtual void* volatile* Get_Reference() = 0;
    virtual bool Overlaps(Filter* filter) = 0;
    virtual FilterType Type() = 0;
    virtual bool IsFieldReference(PlotField* field) = 0;

    virtual int Get_Precedence() = 0;

    bool Get_Enabled() { return enabled; }
    void Set_Enabled(bool enabled) { this->enabled = enabled; }
    const char* Get_Name() { return filterName; }
    char* Get_FieldName() { return fieldName; }

    virtual ~Filter() { }

protected:
    char* fieldName;
    const char* filterName;
    bool enabled = true;
};

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

/*
// magical pdw filter
class PDWFilter : public Filter {
public:
    bool RenderWidget(int id, bool odd) override;
    void RenderPlotPreview(int axis) override;
    void RenderPlotPreviewSmall(int axis) override;
    void Apply(PlotField* field) override;
    bool Overlaps(Filter* filter) override;

    int Get_Precedence() override { return !pass; }
    void* volatile* Get_Reference() override { return (void* volatile*)reference; }

    PDWFilter(Signal6k* reference, char* referenceName, PlotField* pfpd, PlotField* pfdtoa, PlotField* pfrf, PlotField* pfintra, double lowx, double lowy, double highx, double highy, bool pass);
private:
    Signal6k* reference;

    struct limit {
        double low  = 0;
        double high = 0;
        PlotField* field = nullptr;

        limit(double low, double high, PlotField* field) {
            this->low   = low;
            this->high  = high;
            this->field = field;
        }
        limit() {

        }
    };

    limit pd;
    limit pri;
    limit rf;
    limit intra;

    bool pass = true;
};
*/
//applies tbd filter to IQ data associated with the plotfield
void TBDIQFilter(PlotField* tbd);

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

    FIRFilter(SignalCDIF* reference, PlotField* spectra, double sigBW, double filtBW, double filtCenter, unsigned taps, const char* name, bool pass);
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
class MatchFilter : public Filter {
public:
    bool RenderWidget(int id, bool odd, bool range) override;
    void RenderPlotPreview(int axis) override;
    void RenderPlotPreviewSmall(int axis) override;
    void Apply(PlotField* field, void* mfr, bool* first, bool* lastPass) override;
    bool Overlaps(Filter* filter) override;
    FilterType Type() override { return FilterType::IQ; }
    bool IsFieldReference(PlotField* field) override { return false; }

    int Get_Precedence() override;
    void* volatile* Get_Reference() override;

    MatchFIR* Get_FIR() { return fir; }

    MatchFilter(SignalCDIF* reference, double* I, double* Q, unsigned size, const char* name);
    ~MatchFilter();
private:
    MatchFIR* fir = nullptr;
    SignalCDIF* ref;
};

class FilterStack {
public:
    std::vector<Filter*>* Get_Filters() { return &filters; }
    void AddFilter(Filter* filter);
    Filter& operator[](int index) { return *filters[index]; }
    int Count() { return filters.size(); }
    bool Get_Link() { return link; }
    bool* Get_LinkPtr() { return &link; }

    void GenerateMask(PlotField* field, bool linkIQ);
private:
    std::vector<Filter*> filters;
    bool link = false;
};

void TBDIQFilter(PlotField* tbdpd);

#endif
#ifndef _FILTER_HPP_
#define _FILTER_HPP_

class PlotField;

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

protected:
    char* fieldName;
    const char* filterName;
    bool enabled = true;
};


#endif
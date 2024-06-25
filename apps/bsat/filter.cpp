#include <algorithm>

#include "filter.hpp"
#include "imgui.h"
#include "imgui_internal.h"
#include "implot.h"
#include "implot_internal.h"
#include "ImPlot_Extras.h"
#include "logger.hpp"
#include "cdif.hpp"
#include "fft.h"
#include "widgets.hpp"
#include "svg.hpp"

#include "defines.h"
#include "include/birch.h"

using namespace Birch;

#include "bfile.hpp"

static GLuint hidden;
static GLuint visible;
static GLuint cog;

// col = 0x AA BB GG RR

void InitFilterWidgets(GLuint show, GLuint hide) {
    hidden  = hide;
    visible = show;

    //visible = svgTextures["bicons/DecEyeShown.svg"].GetTexture(ImVec2(16, 16))->TextureID();
    //hidden  = svgTextures["bicons/DecEyeHidden.svg"].GetTexture(ImVec2(16, 16))->TextureID();
    //cog     = svgTextures["bicons/IconCog.svg"].GetTexture(ImVec2(16, 16))->TextureID();
}

TBDFilter::TBDFilter(double* volatile* reference, char* referenceName, double lowLimit, double highLimit, bool pass) {
    this->reference  = reference;
    this->fieldName  = referenceName;
    this->lowLimit   = lowLimit;
    this->highLimit  = highLimit;
    this->pass       = pass;
    this->filterName = "tbdlimiter";
}

bool TBDFilter::RenderWidget(int id, bool odd, bool range) {
    bool changed = false;

    ImGui::PushID(this);

    ImVec2 cursor = ImGui::GetCursorScreenPos();
    ImVec2 frameEnd = ImVec2(cursor.x + ImGui::GetWindowContentRegionWidth(), cursor.y + 52);

    ImGui::RenderFrame(cursor, frameEnd, odd? 0x16000000 : 0x32000000, true, 5);

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10, 2));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 6);
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2);
    ImGui::TextUnformatted(fieldName, ImGui::FindRenderedTextEnd(fieldName));
    ImGui::PopStyleVar();

    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 6);

    ImGui::SetNextItemWidth(72);

    int passInt = pass;
    if (ImGui::Combo("##prSel", &passInt, "Reject\0Pass\0")) {
        pass = passInt == 1;

        changed = true;
    }

    ImGui::SameLine();

    double width = highLimit - lowLimit;
    double value = width / 2;

    if (RangeInput(&value, &width, range, ImVec2(-24, 0))) {
        lowLimit  = value - width / 2;
        highLimit = value + width / 2;

        changed = true;
    }

    ImGui::SameLine();

    ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0, 0, 0, 0));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() - 8);

    if (ImGui::ImageButton(t_ImTexID(enabled ? visible : hidden), ImVec2(16, 16))) {
        enabled = !enabled;
        changed = true;
    }

    ImGui::PopID();

    ImGui::PopStyleColor(3);

    return changed;
}

void TBDFilter::RenderPlotPreview(int axis) {
    ImPlot::PushPlotClipRect();
    auto rect = ImPlot::GetCurrentPlot()->PlotRect;
    ImPlot::GetPlotDrawList()->AddRectFilled(ImVec2(rect.GetTL().x, ImPlot::PlotToPixels(0, highLimit, axis).y), ImVec2(rect.GetBR().x, ImPlot::PlotToPixels(0, lowLimit, axis).y), enabled? pass? 0x246cff40 : 0x244050ff : 0x2459f9ff);
    ImPlot::PopPlotClipRect();
}

void TBDFilter::RenderPlotPreviewSmall(int axis) {
    ImPlot::PushPlotClipRect();
    auto rect = ImPlot::GetCurrentPlot()->PlotRect;
    ImPlot::GetPlotDrawList()->AddRectFilled(ImVec2(rect.GetTL().x, ImPlot::PlotToPixels(0, highLimit, axis).y), ImVec2(rect.GetTL().x + 2, ImPlot::PlotToPixels(rect.GetBR().x, lowLimit, axis).y), enabled? pass? 0xFF6cff40 : 0xFF4050ff : 0xFF59f9ff);
    ImPlot::GetPlotDrawList()->AddRectFilled(ImVec2(rect.GetTR().x, ImPlot::PlotToPixels(0, highLimit, axis).y), ImVec2(rect.GetTR().x - 2, ImPlot::PlotToPixels(rect.GetBR().x, lowLimit, axis).y), enabled? pass? 0xFF6cff40 : 0xFF4050ff : 0xFF59f9ff);
    ImPlot::PopPlotClipRect();
}

bool TBDFilter::IsFieldReference(PlotField* field) {
    return field->Data == *reference;
}

static inline bool between(double x, double low, double high) {
    return x >= low && x <= high;
}

void TBDFilter::Apply(PlotField* field, void* mfr, bool* first, bool* lastPass) {
    bool* mask = field->FilterMask;
    int   size = *field->ElementCount;
    auto  maskFieldRef = *((double* volatile ***)mfr); // how did we get here

    // true = pass, false = reject
    // everything starts as false

    // reference* points to the field the filter was set on

    // print reference
    //printf("reference: %p\n", reference);
    //printf("isFirst: %d\n", *first);

    // Assume that pass filters will always be run before rejects
    if (enabled && size != 0 && *reference != nullptr) {
        for (int i = 0; i < size; i++) {
            if (pass) {
                if (between((*reference)[i], lowLimit, highLimit)) {
                    // This filter wants to pass this point

                    if (*first) {
                        mask[i] = true;
                        maskFieldRef[i] = reference;
                    }
                    else {
                        // Don't overwrite if another filter set it to false
                        if (mask[i] == false && maskFieldRef[i] == reference) {
                            mask[i] = true;
                            maskFieldRef[i] = reference;
                        }
                    }
                } else if (!*first && maskFieldRef[i] != reference) {
                    mask[i] = false;
                    maskFieldRef[i] = reference;
                } else if (*first) {
                    maskFieldRef[i] = reference;
                }
            } else {
                if (lastPass && *first)
                    mask[i] = true; //no pass filters. set all to true

                if (between((*reference)[i], lowLimit, highLimit)) //cut additional data
                    mask[i] = false;
            }
        }

        *first = false;
        *lastPass = pass;
    }
}

bool TBDFilter::Overlaps(Filter* filter) {
    if (filter->Get_Reference() == (void* volatile*)reference && !strcmp(filter->Get_Name(), filterName)) {
        TBDFilter* tf = (TBDFilter*)filter;
        return tf->Get_Min() <= highLimit && lowLimit <= tf->Get_Max();
    }

    return false;
}

/*
PDWFilter::PDWFilter(Signal6k* reference, char* referenceName, PlotField* pfpd, PlotField* pfdtoa, PlotField* pfrf, PlotField* pfintra, double lowx, double lowy, double highx, double highy, bool pass) {
    this->reference = reference;
    this->fieldName = referenceName;
    this->pass = pass;

    TBDField* ref = GetField(reference, referenceName); // The field the selection was done on
    TBDField* toa = GetField(reference, strdup("PTOA"));

    // TODO: variable number of fields
    // TODO: Need a way to associate fields
    TBDField* fields[] = { 
                          GetField(reference, strdup("PD")), 
                          GetField(reference, strdup(u8"ΔTOA")), 
                          GetField(reference, strdup("RF")), 
                          GetField(reference, strdup("Intra"))
                         };

    pd    = limit(fields[0]->MaxVal, fields[0]->MinVal, pfpd);
    pri   = limit(fields[1]->MaxVal, fields[1]->MinVal, pfdtoa);
    rf    = limit(fields[2]->MaxVal, fields[2]->MinVal, pfrf);
    intra = limit(fields[3]->MaxVal, fields[3]->MinVal, pfintra);

    limit* lims[] = { &pd, &pri, &rf, &intra };

    // calculate bands for each field
    for (int i = 0; i < toa->TotalElementCount; i++) {
        if (between(toa->Elements[i], lowx, highx)  && between(ref->Elements[i], lowy, highy)) {
			for (int j = 0; j < 4; j++) {
				double val = fields[j]->Elements[i];
				if (val < lims[j]->low)
					lims[j]->low = val;

				if (val > lims[j]->high)
					lims[j]->high = val;
			}
        }
    }
}

void PDWFilter::RenderPlotPreview(int axis) {
    // Leave me alone
    // Don't want to render anything today
}

void PDWFilter::RenderPlotPreviewSmall(int axis) {
    ImPlot::PushPlotClipRect();
    auto rect = ImPlot::GetCurrentPlot()->PlotRect;
    ImPlot::GetPlotDrawList()->AddRectFilled(ImVec2(rect.GetTL().x, rect.GetTL().y), ImVec2(rect.GetTL().x + 2, rect.GetBL().y), enabled? 0xFFff4080 : 0xFF59f9ff);
    ImPlot::GetPlotDrawList()->AddRectFilled(ImVec2(rect.GetTR().x, rect.GetTR().y), ImVec2(rect.GetTR().x - 2, rect.GetBR().y), enabled? 0xFFff4080 : 0xFF59f9ff);
    ImPlot::PopPlotClipRect();
}

bool PDWFilter::RenderWidget(int id, bool odd) {
    return false;
}

void PDWFilter::Apply(PlotField* field) {
    //double* values = field->Data;
    auto ref = &field->Data;
    bool* mask = field->FilterMask;
    int size = *field->ElementCount;

    limit* limits[] = { &rf, &pd, &intra, &pri };

    // run all filters
    // pdw filter is essentially 4 pass filters
    // we really don't care about which field it's currently running on,
    // just run all 4...
    for (int j = 0; j < 4; j++) {
        double* refData = limits[j]->field->Data;

        if (enabled && size != 0 && reference != nullptr) {
            for (int i = 0; i < size; i++) {
                if (pass) {
                    if (between(refData[i], limits[j]->low, limits[j]->high)) {
                        if (first) {
                            mask[i] = true;
                            maskFieldRef[i] = ref;
                        }
                        else {
                            mask[i] = (refData[i] || maskFieldRef[i] == ref) == true;

                            if (mask[i])
                                maskFieldRef[i] = ref;
                        }
                    } else if (!first && maskFieldRef[i] != ref) {
                        mask[i] = false;
                        maskFieldRef[i] = ref;
                    } else if (first) {
                        maskFieldRef[i] = ref;
                    }
                } else {
                    if (lastPass && first)
                        mask[i] = true; //no pass filters. set all to true

                    if (between(refData[i], limits[j]->low, limits[j]->high)) //cut additional data
                        mask[i] = false;
                }
            }

            first = false;
            lastPass = pass;
        }
    }
}

bool PDWFilter::Overlaps(Filter* filter) {
    return false;
}
*/

void TBDIQFilter(PlotField* tbdpd) {
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

bool TBDMatchFilter::RenderWidget(int id, bool odd, bool range) {
    return false;
}
void TBDMatchFilter::RenderPlotPreview(int axis) {
    // Leave me alone
    // Don't want to render anything today
}
void TBDMatchFilter::RenderPlotPreviewSmall(int axis) {
    // No
}
void TBDMatchFilter::Apply(PlotField* field, void* mfr, bool* first, bool* lastPass) {
    bool* mask = field->FilterMask;
    auto  size = *field->ElementCount;

    if (!(enabled && size != 0 && *pd != nullptr && peaks != nullptr && *peakCount != nullptr))
        return;

    const auto  pds     = *pd;
    const auto  peak    = *peaks;
    const auto  peakCnt = **peakCount;

    if (peakCnt == 0)
        return;
    
    
    for (unsigned long i = 0; i < size; i++) {
        Span<double> span(field->Timing[i], field->Timing[i] + pds[i] * 1000000);
        // TODO: Binary search
        bool keep = false;
        for (unsigned long j = 0; j < peakCnt; j++) {
            if (span.Contains(peak[j])) {
                mask[i] = true;
                keep = true;
                break;
            }
        }

        if (!keep)
            mask[i] = false;
    }

    *first = false;
}
bool TBDMatchFilter::Overlaps(Filter* filter) {
    return false;
}
bool TBDMatchFilter::IsFieldReference(PlotField* field) {
    return false;
}

TBDMatchFilter::TBDMatchFilter(double* volatile* peaks, double* volatile* pd, volatile unsigned long* volatile* peakCount, char* referenceName) {
    fieldName = strdup(referenceName);

    this->peaks     = peaks;
    this->pd        = pd;
    this->peakCount = peakCount;
}
TBDMatchFilter::~TBDMatchFilter() {
    free(fieldName);
}

bool FIRFilter::RenderWidget(int id, bool odd, bool range) {
    bool changed = false;
    static bool update = false;
    
    ImVec2 cursor = ImGui::GetCursorScreenPos();
    ImVec2 frameEnd = ImVec2(cursor.x + ImGui::GetWindowContentRegionWidth(), cursor.y + 52);

    ImGui::PushID(id);
    ImGui::BeginGroup();

    ImGui::RenderFrame(cursor, frameEnd, odd? 0x16000000 : 0x32000000, true, 5);

    char fvisid[32];
    char vid[32];
    char wid[32];
    char frpid[32];
    snprintf(vid, 32, "##flow (%d)", id);
    snprintf(wid, 32, "##fhigh (%d)", id);
    snprintf(fvisid, 32, "##fvis (%d)", id);
    snprintf(frpid, 32, "##frpid (%d)", id);

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10, 2));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 6);
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2);
    ImGui::Text("%s", fieldName);
    ImGui::PopStyleVar();

    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 6);

    if (bands.size() == 1) {
        float inputSize = (ImGui::GetWindowWidth() - (ImGui::GetCursorPosX() + ImGui::CalcTextSize("Min:").x + ImGui::CalcTextSize("Max").x + 148)) / 2;

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 5);
        ImGui::Text("Min:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(inputSize);
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 5);
        changed |= ImGui::InputDouble(vid, &bands[0].Start, 0, 0, "%.2f");

        ImGui::SameLine();
        ImGui::Text("Max:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(inputSize);
        changed |= ImGui::InputDouble(wid, &bands[0].End, 0, 0, "%.2f");

        ImGui::SameLine();
        ImGui::SetNextItemWidth(60);

        int passReject = pass;
        changed |= ImGui::Combo(frpid, &passReject, "Reject\0Pass\0");
        pass = passReject;

    } else {
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 5);
        ImGui::Text("Custom Response");
    }

    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0, 0, 0, 0));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() - 8);

    static auto  lastLen = ref->SpectraYRes();
    static int   tapSel  = log2(fir->TapCount()) - 5;
    static float winMax  = 0;

    if (preview == nullptr || lastLen != ref->SpectraYRes() || update) {
        {
            int ndLen = pow(2, tapSel + 5);
            float* newDesign = new float[ndLen];

            for (unsigned i = 0; i < ndLen; i++) {
                newDesign[i] = 0;
            }

            if (bandMode) {
                for (auto& band : bands) {
                    if (band.Start > band.End) {
                        auto t = band.End;

                        band.End = band.Start;
                        band.Start = t;
                    }

                    const float scale = 1 / (sigBW / 1000000 / ndLen);

                    unsigned startEl = band.Start * scale + ndLen / 2;
                    unsigned endEl   = band.End   * scale + ndLen / 2;

                    for (unsigned i = startEl; i < endEl; i++) {
                        newDesign[i]++;
                    }
                }

                double max = 0;

                for (unsigned i = 0; i < ndLen; i++)
                {
                    if (newDesign[i] > max)
                        max = newDesign[i];
                }
                for (unsigned i = 0; i < ndLen; i++)
                {
                    newDesign[i] /= max;
                }

                // Invert if stop filter
                if (!pass) {
                    // Design was normalized to 1 already
                    for (unsigned i = 0; i < ndLen; i++)
                    {
                        newDesign[i] = 1 - newDesign[i];
                    }
                }
            } else {
                // Sort points by x value
                std::sort(points.begin(), points.end(), [](const Birch::Point<double>& a, const Birch::Point<double>& b) -> bool {
                    return a.X < b.X;
                });

                // Interpolate between points
                for (unsigned i = 0; i < points.size() - 1; i++) {
                    const float scale = 1 / (sigBW / 1000000 / ndLen);

                    unsigned startEl = points[i].X * scale + ndLen / 2;
                    unsigned endEl   = points[i + 1].X * scale + ndLen / 2;

                    for (unsigned j = startEl; j < endEl; j++) {
                        float t = (j - startEl) / (float)(endEl - startEl);

                        newDesign[j] = points[i].Y * (1 - t) + points[i + 1].Y * t;
                    }
                }
            }

            fir->SetFilter(newDesign, ndLen, window);

            delete[] newDesign;
        }

        delete[] preview;
        delete[] response;
        delete[] previewFilt;
        delete[] design;
        delete[] winPrev;
        delete[] winPrev2;

        winMax = 0;

        lastLen = ref->SpectraYRes();

        preview = new float[lastLen];
        response = new float[fir->FFTSize()];
        previewFilt = new float[lastLen];
        design = new float[fir->TapCount()];
        winPrev = new float[fir->TapCount()];
        winPrev2 = new float[fir->TapCount()];

        for (unsigned i = 0; i < lastLen; i++) {
            preview[i] = 0;
        }

        auto win = window->AmpResponse(fir->TapCount());

        for (unsigned i = 0; i < fir->TapCount(); i++) {
            design[i] = fir->Design()[i].R;
        }

        {
            Complex<double>* fftIn = new Complex<double>[fir->FFTSize()];
            Complex<double>* fftOut = new Complex<double>[fir->FFTSize()];

            Complex<double> zero = { 0, 0 };

            for (unsigned i = 0; i < fir->TapCount(); i++) {
                const auto el = fir->ImpulseResponse()[i];

                fftIn[i] = el;

                if (fabs(el.R) > winMax)
                    winMax = fabs(el.R);
                if (fabs(el.I) > winMax)
                    winMax = fabs(el.I);
            }
            for (unsigned i = fir->TapCount(); i < fir->FFTSize(); i++) {
                fftIn[i] = zero;
            }

            fft(fftIn, fftOut, fir->FFTSize());

            for (unsigned i = 0; i < fir->FFTSize(); i++) {
                double mag = sqrt(fftOut[i].R * fftOut[i].R + fftOut[i].I * fftOut[i].I);
                response[i] = mag;
            }

            delete[] fftIn;
            delete[] fftOut;
        }

        for (unsigned i = 0; i < fir->TapCount(); i++) {
            winPrev[i]  = win[i] *  winMax;
            winPrev2[i] = win[i] * -winMax;
        }

        double maxResp = response[0];
        double maxPrev = 0;

        for (unsigned i = 0; i < fir->FFTSize(); i++)
            if (response[i] > maxResp)
                maxResp = response[i];

        for (unsigned i = 0; i < lastLen; i++)
            if (ref->SpectraTotal()[i] > maxPrev)
                maxPrev = ref->SpectraTotal()[i];

        if (bandMode) {
            maxResp = 1.0 / maxResp;

            for (unsigned i = 0; i < fir->FFTSize(); i++) {
                response[i] *= maxResp;
            }
        }

        maxPrev = 1.0 / maxPrev;

        const double scale = (double)fir->FFTSize() / (double)lastLen;

        for (unsigned i = 0; i < lastLen; i++) {
            preview[i] = ref->SpectraTotal()[i] * maxPrev;
            previewFilt[i] = preview[i] * response[(unsigned)(i * scale)];
        }

        update = false;
    }

    const double start = (-sigBW / 2) / 1000000;
    const double pStep = (sigBW / 1000000) / lastLen;
    const double rStep = (sigBW / 1000000) / fir->FFTSize();
    const double dStep = (sigBW / 1000000) / fir->TapCount();

    ImGui::PushID(this);

    if (ImGui::ImageButton(t_ImTexID(cog), ImVec2(16, 16))) {
        ImGui::OpenPopup("FIR Designer");
    }

    bool open = true;
    if (ImGui::BeginPopupModal("FIR Designer", &open)) {
        ImGui::Columns(2, "FIR cols", true);

        auto map = ImPlot::GetInputMap();
        
        map.VerticalMod     = ImGuiModFlags_Shift;
        map.HorizontalMod   = ImGuiModFlags_Alt;
        map.BoxSelectMod    = ImGuiModFlags_Alt;
        map.QueryButton     = ImGuiMouseButton_Left;
        map.QueryMod        = ImGuiModFlags_None;
        map.BoxSelectButton = ImGuiMouseButton_Right;
        map.PanButton       = ImGuiMouseButton_Middle;

        ImPlot::GetInputMap() = map;

        if (ImPlot::BeginPlot("Frequency Response", 0, 0, ImVec2(-1, -1), ImPlotFlags_Query | ImPlotFlags_Crosshairs)) {
            ImPlot::PlotShaded("rIQ", preview, lastLen, 0, pStep, start);
            ImPlot::PlotLine("rIQ", preview, lastLen, pStep, start);
            ImPlot::PlotShaded("sIQ", previewFilt, lastLen, 0, pStep, start);
            ImPlot::PlotLine("sIQ", previewFilt, lastLen, pStep, start);
            ImPlot::PlotLine("Response", response, fir->FFTSize(), rStep, start);
            //ImPlot::PlotLine("Design", design, fir->TapCount(), dStep, start);

            if (bandMode) {
                for (auto& band : bands) {
                    ImGui::PushID(&band);
                    ImPlot::PushPlotClipRect();

                    auto dl = ImPlot::GetPlotDrawList();

                    const auto limits = ImPlot::GetPlotLimits().Y;

                    auto tl = ImPlot::PlotToPixels(ImVec2(band.Start, limits.Max));
                    auto br = ImPlot::PlotToPixels(ImVec2(band.End,   limits.Min));

                    dl->AddRectFilled(tl, br, 0x08FFFFFF, 0);

                    ImPlot::PopPlotClipRect();


                    changed |= ImPlot::DragLineX("##start", &band.Start, false);
                    changed |= ImPlot::DragLineX("##end",   &band.End,   false);
                    //IMPLOT_API bool DragPoint(const char* id, double* x, double* y, bool show_label = true);
                    double center = band.Center();
                    double yVal = (limits.Max - limits.Min) / 2;
                    if (ImPlot::DragPoint("##center", &center, &yVal, false)) {
                        const double length = band.Length() * 0.5;

                        band.Start = center - length;
                        band.End   = center + length;

                        if (band.Start < start)
                            band.Start = start;

                        if (band.End > -start)
                            band.End = -start;

                        changed = true;
                    }

                    ImGui::PopID();
                }

                if (ImPlot::IsPlotQueried() && !ImGui::GetIO().MouseClicked[1])
                {
                    auto range = ImPlot::GetPlotQuery(ImPlotYAxis_1);
                    Birch::Span<double> r = {range.X.Min, range.X.Max};

                    if (r.Length() > 0.000001)
                        bands.push_back(r);

                    ImPlot::HidePlotQuery();

                    changed = true;
                }
            } else {
                //ImPlot::PlotLine("Design", design, fir->TapCount(), dStep, start);

                for (auto& point : points) {
                    ImGui::PushID(&point);

                    changed |= ImPlot::DragPoint("##point", &point.X, &point.Y, false);

                    ImGui::PopID();
                }

                // Add point on mouse click
                if (ImGui::GetIO().MouseClicked[1]) {
                    auto pos = ImPlot::GetPlotMousePos();
                    points.push_back(Point<double>(pos.x, pos.y));
                    changed = true;
                }
            }

            ImPlot::EndPlot();
        }
        ImGui::NextColumn();
        if (ImPlot::BeginPlot("Impulse Response")) {
            ImPlot::SetNextFillStyle(ImVec4(0, 0, 0, -1), 0.1f);
            ImPlot::PlotShaded("Window", winPrev, fir->TapCount());
            ImPlot::SetNextFillStyle(ImVec4(0, 0, 0, -1), 0.1f);
            ImPlot::PlotShaded("Window", winPrev2, fir->TapCount());

            ImPlot::PlotLine("R", &(fir->ImpulseResponse()[0].R), fir->TapCount(), 1, 0, 0, sizeof(fir->ImpulseResponse()[0]));
            ImPlot::PlotLine("I", &(fir->ImpulseResponse()[0].I), fir->TapCount(), 1, 0, 0, sizeof(fir->ImpulseResponse()[0]));

            ImPlot::PlotLine("Window", winPrev, fir->TapCount());
            ImPlot::PlotLine("Window", winPrev2, fir->TapCount());

            ImPlot::EndPlot();
        }

        changed |= ImGui::Combo("Taps", &tapSel, "32\00064\000128\000256\000512\0001024\0002048\0004096\0008192\00016384\000", 5);
        changed |= WindowWidget(&window, GetWindows());

        int mode = bandMode;
        changed |= ImGui::Combo("Mode", &mode, "Points\0Bands\0", 2);
        bandMode = mode;

        if (bandMode) {
            if (ImGui::BeginChildFrame(2341223432, ImVec2(0, 0))) {
                auto removeMe = std::end(bands);
                auto bandItr  = std::begin(bands);

                for (auto& band : bands) {
                    ImGui::PushID(&band);

                    float bands[2] = { (float)band.Start, (float)band.End };

                    changed |= ImGui::DragFloat2("##band", bands, 0.1, start, -start);

                    band.Start = bands[0];
                    band.End   = bands[1];

                    ImGui::SameLine();

                    if (ImGui::Button("Delete"))
                        removeMe = bandItr;
                    
                    bandItr++;

                    ImGui::PopID();
                }

                if (removeMe != std::end(bands)) {
                    bands.erase(removeMe);
                    changed = true;
                }
            }

            ImGui::EndChildFrame();
        } else {

        }

        ImGui::SetMouseCursor(ImGuiMouseCursor_Arrow);

        ImGui::EndPopup();
    }

    ImGui::PopID();

    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() - 8);
    ImGui::PushID(fvisid);
    if (ImGui::ImageButton(t_ImTexID(enabled? visible : hidden), ImVec2(16, 16))) {
        enabled = !enabled;
        changed = true;
    }
    ImGui::PopID();

    ImGui::PopStyleColor(3);
    
    ImGui::EndGroup();
    ImGui::PopID();

    if (changed)
        update = true;

    return changed;
}
void FIRFilter::RenderPlotPreview(int axis) {
    ImPlot::PushPlotClipRect();
    auto rect = ImPlot::GetCurrentPlot()->PlotRect;

    for (const auto& band : bands) {
        auto lowLimit = band.Start;
        auto highLimit = band.End;

        ImPlot::GetPlotDrawList()->AddRectFilled(ImVec2(rect.GetTL().x, ImPlot::PlotToPixels(0, highLimit, axis).y), ImVec2(rect.GetBR().x, ImPlot::PlotToPixels(0, lowLimit, axis).y), enabled? pass? 0x246cff40 : 0x244050ff : 0x2459f9ff);
    }

    ImPlot::PopPlotClipRect();
}
void FIRFilter::RenderPlotPreviewSmall(int axis) {
    ImPlot::PushPlotClipRect();
    auto rect = ImPlot::GetCurrentPlot()->PlotRect;

    for (const auto& band : bands) {
        auto lowLimit = band.Start;
        auto highLimit = band.End;

        ImPlot::GetPlotDrawList()->AddRectFilled(ImVec2(rect.GetTL().x, ImPlot::PlotToPixels(0, highLimit, axis).y), ImVec2(rect.GetTL().x + 2, ImPlot::PlotToPixels(rect.GetBR().x, lowLimit, axis).y), enabled? pass? 0xFF6cff40 : 0xFF4050ff : 0xFF59f9ff);
        ImPlot::GetPlotDrawList()->AddRectFilled(ImVec2(rect.GetTR().x, ImPlot::PlotToPixels(0, highLimit, axis).y), ImVec2(rect.GetTR().x - 2, ImPlot::PlotToPixels(rect.GetBR().x, lowLimit, axis).y), enabled? pass? 0xFF6cff40 : 0xFF4050ff : 0xFF59f9ff);
    }

    ImPlot::PopPlotClipRect();
}
void FIRFilter::Apply(PlotField* field, void* mfr, bool* first, bool* lastPass) {
    // Unused
}
bool FIRFilter::Overlaps(Filter* filter) {
    return false;
}
int FIRFilter::Get_Precedence() {
    return 3;
}
void* volatile* FIRFilter::Get_Reference() {
    return (void* volatile*)&ref;
}
bool FIRFilter::IsFieldReference(PlotField* field) {
    return field == spectra;
}

FIRFilter::FIRFilter(SignalCDIF* reference, PlotField* spectra, double sigBW, double filtBW, double filtCenter, unsigned taps, const char* name, bool pass) {
    window = new Hann();

    this->pass = pass;

    fir = new BandpassFIR(sigBW, taps);
    fir->SetBandpass(filtCenter, filtBW, taps, window, !pass);

    filterName = "FIR";
    fieldName = strdup(name);

    ref = reference;
    this->spectra = spectra;



    this->sigBW = sigBW;

    bands.push_back(Span<double>((filtCenter - filtBW / 2) / 1000000, (filtCenter + filtBW / 2) / 1000000));

    points.push_back(Point<double>((filtCenter - filtBW / 2) / 1000000, 1));
    points.push_back(Point<double>((filtCenter + filtBW / 2) / 1000000, 1));
}
FIRFilter::~FIRFilter() {
    delete window;
    delete fir;
    delete[] preview;
    delete[] response;
    delete[] previewFilt;
    delete[] design;
    delete[] winPrev;
    delete[] winPrev2;

    free(fieldName);
}

bool MatchFilter::RenderWidget(int id, bool odd, bool range) {
    return false;
}
void MatchFilter::RenderPlotPreview(int axis) {

}
void MatchFilter::RenderPlotPreviewSmall(int axis) {

}
void MatchFilter::Apply(PlotField* field, void* mfr, bool* first, bool* lastPass) {
    // Unused
}
bool MatchFilter::Overlaps(Filter* filter) {
    return false;
}
int MatchFilter::Get_Precedence() {
    return 3;
}
void* volatile* MatchFilter::Get_Reference() {
    return (void* volatile*)&ref;
}

MatchFilter::MatchFilter(SignalCDIF* reference, double* I, double* Q, unsigned size, const char* name) {
    filterName = "Matched";
    fieldName = strdup(name);

    ref = reference;

    fir = new MatchFIR(I, Q, size);
}
MatchFilter::~MatchFilter() {
    free(fieldName);
    delete fir;
}


void FilterStack::GenerateMask(PlotField* field, bool linkIQ) {
    // Filters with lower precidence are applied first
    auto compareFilter = [](Filter* a, Filter* b) { return a->Get_Precedence() < b->Get_Precedence(); };
    std::sort(filters.begin(), filters.end(), compareFilter);

    bool* mask = field->FilterMask;
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

void FilterStack::AddFilter(Filter* filter) {
    // erase any overlapping filter when adding a new one
    auto overlaps = [&](Filter* f) { return filter->Overlaps(f); };
    filters.erase(std::remove_if(filters.begin(), filters.end(), overlaps), filters.end());

    filters.push_back(filter);
}
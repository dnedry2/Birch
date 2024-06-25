#include <omp.h>
#include <algorithm>

#include "color.hpp"
#include "imgui.h"
#include "imgui_internal.h"
#include "gradient.hpp"
#include "logger.hpp"
#include "cdif.hpp"
#include "widgets.hpp"
#include "svg.hpp"

#include "defines.h"

#include "plot.hpp"



#include "stb_image_write.h"


using namespace Birch;

static GLuint hidden;
static GLuint visible;
static GLuint cog;

void InitColorWidgets(GLuint show, GLuint hide) {
    hidden  = hide;
    visible = show;

    //visible = svgTextures["bicons/DecEyeShown.svg"].GetTexture(ImVec2(16, 16))->TextureID();
    //hidden  = svgTextures["bicons/DecEyeHidden.svg"].GetTexture(ImVec2(16, 16))->TextureID();
    //cog     = svgTextures["bicons/IconCog.svg"].GetTexture(ImVec2(16, 16))->TextureID();
}

void TransitionColor(uint start, uint end, uint* output, int steps)
{
    //TODO: Transparency

    //extract individual colors
    float sRed = start & 0xFF;
    float sGreen = ((start >> IM_COL32_G_SHIFT) & 0xFF);
    float sBlue = ((start >> IM_COL32_B_SHIFT) & 0xFF);
    float eRed = end & 0xFF;
    float eGreen = ((end >> IM_COL32_G_SHIFT) & 0xFF);
    float eBlue = ((end >> IM_COL32_B_SHIFT) & 0xFF);

    //how much to move per step
    float redShift = (eRed - sRed) / steps;
    float greenShift = (eGreen - sGreen) / steps;
    float blueShift = (eBlue - sBlue) / steps;

    for (int i = 0; i < steps; i++) {
        sRed += redShift;
        sGreen += greenShift;
        sBlue += blueShift;

        output[i] = sRed;
        output[i] |= (uint)sGreen << 8;
        output[i] |= (uint)sBlue << 16;
        output[i] |= (uint)(255) << 24;
    }
}

uint* MakeColorGradient(Gradient* gradient, int size)
{
    uint* output = (uint*)malloc(sizeof(*gradient) * size);
    int runSize = floor((double)size / (gradient->StopCount - 1));

    int gradPos = 0;
    for (int i = 1; i < gradient->StopCount; i++) {
        for (int j = 0; j < runSize; j++) {
            TransitionColor(gradient->Stops[i - 1], gradient->Stops[i], output + gradPos, runSize);
        }
        gradPos += runSize;
    }

    return output;
}

ColorizerSingle::ColorizerSingle(double* volatile* reference, char* referenceName, double value, double width, uint color)
: Colorizer(referenceName) {
    this->value     = value;
    this->width     = width;
    this->reference = reference;
    this->color     = color;
}
void ColorizerSingle::Apply(uint* colormap, ulong size)
{
    if (enabled && size != 0 && *reference != nullptr)
        for (ulong i = 0; i < size; i++)
            if ((*reference)[i] > value - width && (*reference)[i] < value + width)
                colormap[i] = color;
}
bool ColorizerSingle::RenderWidget(int id, bool odd, bool range)
{
    bool changed = false;

    ImGui::PushID(this);

    ImVec4 vecCol = ImGui::ColorConvertU32ToFloat4(color);
    float fcol[] = { vecCol.x, vecCol.y, vecCol.z };

    ImVec2 cursor = ImGui::GetCursorScreenPos();
    ImVec2 frameEnd = ImVec2(cursor.x + ImGui::GetWindowContentRegionWidth(), cursor.y + 52);

    ImGui::RenderFrame(cursor, frameEnd, odd? 0x16000000 : 0x32000000, true, 5);

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10, 2));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 6);
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2);
    ImGui::TextUnformatted(fieldName, ImGui::FindRenderedTextEnd(fieldName));
    ImGui::PopStyleVar();

    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 6);

    changed = ImGui::ColorEdit3("##color", fcol, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_PickerHueWheel);
    color   = ImGui::ColorConvertFloat4ToU32(ImVec4(fcol[0], fcol[1], fcol[2], 1));

    ImGui::SameLine();

    changed |= RangeInput(&value, &width, range, ImVec2(-24, 0));

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

//renders a horizontal gradient strip
void GradientStrip(char* id, Gradient* gradient, ImVec2 size)
{
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    ImVec2 cursorPos = ImGui::GetCursorScreenPos();
    ImVec2 endRect = ImVec2(cursorPos.x + size.x, cursorPos.y + size.y);

    ImGui::BeginGroup();
    ImGui::PushID(id);

    drawList->AddRectFilled(cursorPos, endRect, ImGui::GetColorU32(ImGuiCol_Border));

    float slice = (size.x) / (gradient->StopCount - 1);

    for (int i = 0; i < gradient->StopCount - 1;) {
        ImVec2 nextStart = ImVec2((cursorPos.x) + slice * i++, cursorPos.y);
        ImVec2 nextEnd = ImVec2(nextStart.x + slice, endRect.y);

        drawList->AddRectFilledMultiColor(nextStart, nextEnd, gradient->Stops[i - 1] | ((uint)0xFF << 24), gradient->Stops[i] | ((uint)0xFF << 24), gradient->Stops[i] | ((uint)0xFF << 24), gradient->Stops[i - 1] | ((uint)0xFF << 24));
        //drawList->AddRectFilled(nextStart, nextEnd, gradient[i - 1] | (0xFF << 24));
    }

    ImGui::SetCursorScreenPos(ImVec2(endRect.x, endRect.y));

    ImGui::PopID();
    ImGui::EndGroup();
}

// Gradient picker control
bool GradientPicker(const char* name, Gradient** gradientp, ImVec2 size, std::vector<Gradient*>* gradients, Gradient** hovGradient, bool* hovered)
{
    ImGui::PushID(name);
    ImGui::BeginGroup();

    Gradient* gradient = *gradientp;

    ImDrawList* drawList  = ImGui::GetWindowDrawList();
    ImVec2      cursorPos = ImGui::GetCursorScreenPos();
    ImVec2      endRect   = ImVec2(cursorPos.x + size.x, cursorPos.y + size.y);

    bool selectionChanged = false;

    drawList->AddRectFilled(cursorPos, endRect, ImGui::GetColorU32(ImGuiCol_Border), 5);

    float slice = (size.x) / gradient->StopCount;

    for (int i = 0; i < gradient->StopCount;) {
        ImVec2 nextStart = ImVec2((cursorPos.x) + slice * i++, cursorPos.y);
        ImVec2 nextEnd   = ImVec2(nextStart.x + slice, endRect.y);

        ImDrawCornerFlags corners = ImDrawCornerFlags_None;
        if (i - 1 == 0)
            corners = ImDrawCornerFlags_TopLeft | ImDrawCornerFlags_BotLeft;
        if (i == gradient->StopCount)
            corners = ImDrawCornerFlags_TopRight | ImDrawCornerFlags_BotRight;

        drawList->AddRectFilled(nextStart, nextEnd, gradient->Stops[i - 1] | ((uint)0xFF << 24), 5, corners);
    }

    ImGui::SetCursorScreenPos(ImVec2(endRect.x, cursorPos.y));

    //ImGui::SameLine();
    if (strnlen(name, 1) == 1)
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 8);
    else
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 2);
    
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 4);

    ImGui::TextUnformatted(name, ImGui::FindRenderedTextEnd(name));

    // TODO: Rewrite this, should be able to use push/pop id
    char psid[] = "previewStrip";
    char popupID[32];
    char gslID[32];
    snprintf(gslID, 32, "gsList%s", name);
    snprintf(popupID, 32, "gsPopup%s", name);
    bool shown = false;

    if (ImGui::BeginPopup(popupID)) {
        shown = true;
        ImGui::Text("Current: %s", gradient->Name);
        GradientStrip(psid, gradient, ImVec2(200, 15));
        ImGui::Separator();

        static bool sorted = false;
        static auto sortedGradients = std::vector<std::vector<Gradient*>>();

        if (!sorted) {
            int lastCat = 0;
            for (auto grad : *gradients) {
                if (lastCat != grad->CatagoryID)
                    sortedGradients.push_back(std::vector<Gradient*>());

                sortedGradients.back().push_back(grad);
                lastCat = grad->CatagoryID;
            }

            std::sort(sortedGradients.begin(), sortedGradients.end(),
                [](const std::vector<Gradient*> & a, const std::vector<Gradient*> & b) -> bool
                { 
                    return a.front()->CatagoryID < b.front()->CatagoryID;
                });
            
            sorted = true;
        }

        if (ImGui::BeginChild("gpsellist", ImVec2(0, 200))) {
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 10);
            for (auto grads : sortedGradients) {
                if (ImGui::CollapsingHeader(grads.front()->Catagory)) {
                    for (auto grad : grads) {
                        ImVec2 cursor = ImGui::GetCursorPos();

                        char selID[32];
                        snprintf(selID, 32, "##gs_%s", grad->Name);

                        if (ImGui::Selectable(selID, &selectionChanged, ImGuiSelectableFlags_SpanAvailWidth, ImVec2(0, 15))) {
                            *gradientp = grad;
                            selectionChanged = true;
                            ImGui::CloseCurrentPopup();
                            break;
                        }
                        
                        if (ImGui::IsItemHovered() && *gradientp != grad) {
                            *hovGradient = grad;
                            *hovered = true;
                        }

                        ImGui::SetCursorPos(cursor);

                        char gname[32];
                        snprintf(gname, 32, "gp-%s", grad->Name);
                        GradientStrip(gname, grad, ImVec2(100, 15));
                        ImGui::SameLine();
                        ImGui::Text("%s", grad->Name);
                    }
                }
            }
        }
        ImGui::EndChild();

        ImGui::EndPopup();
    }

    // If the gradient picker is clicked, open the popup
    if (!shown && ImRect(cursorPos, endRect).Contains(ImGui::GetMousePos()) && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
        ImGui::OpenPopup(popupID);

    // If the gradient picker is hovered, show the preview
    if (!shown && ImRect(cursorPos, endRect).Contains(ImGui::GetMousePos())) {
        ImGui::BeginTooltip();
        {
            GradientStrip(psid, gradient, ImVec2(90, 30));
            ImGui::SameLine();
            ImGui::BeginGroup();
            ImGui::Text("%s", gradient->Name);
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 10);
            ImGui::Text("%d colors", gradient->StopCount);
            ImGui::EndGroup();
        }
        ImGui::EndTooltip();
    }

    ImGui::EndGroup();
    ImGui::PopID();

    return selectionChanged;
}

ColorizerRange::ColorizerRange(double* volatile* reference, char* referenceName, double value, double width, Gradient* gradient, std::vector<Gradient*>* gradients)
: Colorizer(referenceName) {
    this->value     = value;
    this->width     = width;
    this->gradient  = gradient;
    this->reference = reference;
    this->gradients = gradients;

    fieldName       = referenceName;

    renderedGradient = MakeColorGradient(gradient, renderSize);
}

static void applyRange(const uint* grad, uint gradLen, uint* colormap, const double* ref, ulong len, double low, double high) {
    const double range = 1 / (high - low);
    const double rSize = range * gradLen;

    for (ulong i = 0; i < len; i++) {
        const auto val = ref[i];

        if (val >= low && val <= high)
            colormap[i] = grad[static_cast<uint>((val - low) * rSize)];
    }
}
void ColorizerRange::Apply(uint* colormap, ulong size)
{
    double lowLimit  = value - width;
    double highLimit = value + width;

    if (enabled && size != 0 && *reference != nullptr)
        applyRange(renderedGradient, renderSize, colormap, *reference, size, lowLimit, highLimit);
}
bool ColorizerRange::RenderWidget(int id, bool odd, bool range)
{
    bool changed = false;

    ImVec2 cursor = ImGui::GetCursorScreenPos();
    ImVec2 frameEnd = ImVec2(cursor.x + ImGui::GetWindowContentRegionWidth(), cursor.y + 52);

    ImGui::PushID(this);
    ImGui::BeginGroup();
    ImGui::RenderFrame(cursor, frameEnd, odd? 0x16000000 : 0x32000000, true, 5);

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10, 2));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 6);
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2);
    ImGui::TextUnformatted(fieldName, ImGui::FindRenderedTextEnd(fieldName));
    ImGui::PopStyleVar();

    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 6);

    Gradient* hoveredGradient;
    bool hovered = false;
    changed |= GradientPicker("", &gradient, ImVec2(25, 25), gradients, &hoveredGradient, &hovered);

    if (changed)
        renderedGradient = MakeColorGradient(gradient, renderSize);

    if (hovered) {
        if (!lastHoverUpdate || hoveredGradient != lastHoverGradient) {
            renderedGradient = MakeColorGradient(hoveredGradient, renderSize);
            changed = true;
        }
    }
    else if (lastHoverUpdate) {
        renderedGradient = MakeColorGradient(gradient, renderSize);
        changed = true;
    }

    lastHoverUpdate = hovered;

    ImGui::SameLine();
    
    changed |= RangeInput(&value, &width, range, ImVec2(-24, 0));

    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0, 0, 0, 0));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() - 8);

    ImGui::PushID(id);
    if (ImGui::ImageButton(t_ImTexID(enabled? visible : hidden), ImVec2(16, 16))) {
        enabled = !enabled;
        changed = true;
    }
    ImGui::PopID();

    ImGui::PopStyleColor(3);
    ImGui::EndGroup();
    ImGui::PopID();

    return changed;
}

ColorizerSpectra::ColorizerSpectra(double* volatile* reference, uint* volatile* colormap, volatile unsigned long* volatile* colormapSize, const char* referenceName, volatile int* xRes, volatile int* yRes, volatile bool* ready, volatile char* version, Gradient* gradient, std::vector<Gradient*>* gradients, FilterStack* filters)
: Colorizer(referenceName) {
    this->reference    = reference;
    this->gradients    = gradients;
    this->colormap     = colormap;
    this->colormapSize = colormapSize;
    this->xRes         = xRes;
    this->yRes         = yRes;
    this->ready        = ready;
    this->version      = version;
    this->filters      = filters;

    Set_Color(gradient);
    expandBuffer(tempBufLen);
}
ColorizerSpectra::~ColorizerSpectra() {
    delete[] tempBuf;
}
void ColorizerSpectra::expandBuffer(uint size) {
    delete[] tempBuf;

    try {
        tempBuf = new double[size];
        tempBufLen = size;
    } catch (...) {
        DispError("ColorizerSpectra::expandBuffer", "Failed to allocate memory!");
        tempBufLen = 0;
    }

    
}
template <typename T>
static inline T clamp(T in, T min, T max) {
    if (in < min)
        return min;
    else if (in > max)
        return max;
    
    return in;
}
static float* pareto(int length, float alpha, float* buffer = nullptr) {
    float* buf = buffer;

    if (buf == nullptr)
        buf = new float[length];

    const float l = 1;
    const float h = 20;
    const float a = alpha;

    const float it = 20.0 / length;
    float x = 1;

    for (int i = 0; i < length; i++, x += it) {
        buf[i] = (1 - powf(l, a) * powf(x, -a)) / (1 - powf(l/h, a));
    }

    float min = buf[0];
    float max = buf[0];

    for (int i = 0; i < length; i++) {
        if (buf[i] > max)
            max = buf[i];
        if (buf[i] < min)
            min = buf[i];
    }

    const float scale = 1 / (max - min);

    for (int i = 0; i < length; i++)
        buf[i] = (buf[i] - min) * scale;

    return buf;
}

static double* resizeArray(const double* input, uint size, uint newSize) {
    double* output = new double[newSize];
    double scale = (double)size / newSize;

    for (uint i = 0; i < newSize; ++i) {
        double interpolatedIndex = i * scale;
        uint index = floor(interpolatedIndex);
        double fraction = interpolatedIndex - index;

        if (index + 1 < size) {
            output[i] = (1.0 - fraction) * input[index] + fraction * input[index + 1];
        } else {
            output[i] = input[index];
        }
    }

    return output;
}

void ColorizerSpectra::apply() {
    auto timer = Stopwatch();

    double minMag = 0;
    double maxMag = 0;

    float* pScale = nullptr;

    const int csize = **colormapSize;
    *ready = false;

    if (csize > tempBufLen)
        expandBuffer(csize);

    const auto ref = *reference;
    const int xr = *xRes;
    const int yr = *yRes;

    if (ref == nullptr) {
        DispWarning("ColorizerSpectra::Apply", "Reference was null!");
        processing = false;
        return;
    }

    if (xr <= 0 || yr <= 0) {
        DispWarning("ColorizerSpectra::Apply", "Resolution error! (%d x %d)", xr, yr);
        processing = false;
        return;
    }

    if (csize < 0) {
        DispWarning("ColorizerSpectra::Apply", "Colormap length was negative! (%d)", csize);
        processing = false;
        return;
    }

    if (csize == 0) {
        processing = false;
        return;
    }

/*
    if (filters != nullptr && filter) {
        // Get iq filters
        std::vector<FIR*> filt = std::vector<FIR*>();

        uint filterSize = 0;
        for (auto f : *(filters->Get_Filters())) {
            if (f->Type() == FilterType::IQ && f->Get_Enabled()) {
                if (!strncmp(f->Get_Name(), "Match", 5)) {
                    auto fir = static_cast<MatchFilter*>(f)->Get_FIR();

                    if (fir->FFTSize() > filterSize)
                        filterSize = fir->FFTSize();

                    filt.push_back(fir);
                } else {
                    auto fir = static_cast<FIRFilter*>(f)->Get_FIR();

                    if (fir->FFTSize() > filterSize)
                        filterSize = fir->FFTSize();

                    filt.push_back(fir);
                }
            }
        }

        // Build filter frequency response
        if (filt.size() > 0) {
            Birch::Complex<double>* filterKernel = new Birch::Complex<double>[filterSize];

            Complex<double>* curKernelIn  = new Complex<double>[filterSize];
            Complex<double>* curKernelOut = new Complex<double>[filterSize];

            fft_plan fkPlan = fft_make_plan(filterSize);

            const Complex<double> zero = { 0, 0 };
            unsigned it = 0;
            for (auto f : filt) {
                // Setup input buffer
                const auto ir = f->ImpulseResponse(); // Window will have already been applied
                for (unsigned i = 0; i < f->TapCount(); i++)
                    curKernelIn[i] = ir[i];

                // Zero out remaining buffer length
                for (unsigned i = f->TapCount(); i < filterSize; i++)
                    curKernelIn[i] = zero;

                // Send the first fft to the main buffer
                if (it == 0) {
                    fft_cpx_forward(fkPlan, curKernelIn, filterKernel);

                    if (f->Shift())
                        fftshift(filterKernel, filterSize);
                } else {
                    fft_cpx_forward(fkPlan, curKernelIn, curKernelOut);

                    if (f->Shift())
                        fftshift(curKernelOut, filterSize);

                    // Add this filter kernel to output kernel
                    for (unsigned i = 0; i < filterSize; i++)
                        filterKernel[i] = filterKernel[i] * curKernelOut[i];
                }

                it++;
            }

            delete[] curKernelIn;
            delete[] curKernelOut;

            fft_destroy_plan(fkPlan);

            fftshift(filterKernel, filterSize);

            // Get magnitude 
            double* filterMag = new double[filterSize];
            for (unsigned i = 0; i < filterSize; i++)
                filterMag[i] = filterKernel[i].Magnitude();
            

            // Scale response to xres
            double* filterMagScaled = resizeArray(filterMag, filterSize, xr);

            delete[] filterKernel;
            delete[] filterMag;

            // Normalize
            double max = filterMagScaled[0];
            for (unsigned i = 1; i < xr; i++)
                if (filterMagScaled[i] > max)
                    max = filterMagScaled[i];
            
            for (unsigned i = 0; i < xr; i++)
                filterMagScaled[i] /= max;

            // Apply filter
            #pragma omp parallel for
            for (uint y = 0; y < yr; y++) {
                for (uint x = 0; x < xr; x++) {
                    const uint idx = y * xr + x;
                    tempBuf[idx] = ref[idx] * filterMagScaled[x];
                }
            }

            delete[] filterMagScaled;
        } else {
            for (uint i = 0; i < csize; i++)
                tempBuf[i] = ref[i];
        }
    }
    */


    switch (scale) {
        case 0:
            #pragma omp parallel for 
            for (int i = 0; i < csize; i++)
                tempBuf[i] = sqrt(ref[i]);
            break;
        case 1:
            #pragma omp parallel for
            for (int i = 0; i < csize; i++)
                tempBuf[i] = log(sqrt(ref[i]));
            break;
        case 2:
            #pragma omp parallel for
            for (int i = 0; i < csize; i++)
                tempBuf[i] = exp(log10(sqrt(ref[i])));
            break;
    }

    if (cancelJob)
        goto cancel;

    sig_mag_max = 0;
    sig_mag_min = tempBuf[0];
    for (int i = 0; i < csize && !cancelJob; i++) {
        if (tempBuf[i] > sig_mag_max)
            sig_mag_max = tempBuf[i];
        if (tempBuf[i] < sig_mag_min)
            sig_mag_min = tempBuf[i];
    }

    minMag = sig_mag_max * (mag_min / 100.0);
    maxMag = sig_mag_max * (mag_max / 100.0);

    if (paretoDeg <= -1 || paretoDeg >= 1) {
        pScale = pareto(255, paretoDeg);
    }
    else {
        pScale = new float[255];
        for (int i = 0; i < 255; i++)
            pScale[i] = 1;
    }

    if (cancelJob)
        goto cancel;

    if (autoscale)
    {
        const int smOff = as_smooth / 2;

        #pragma omp parallel for
        for (int i = 0; i < yr; i++)
        {
            const int sStart = i * xr;
            const int sEnd = i * xr + xr;

            int bStart = sStart - smOff * yr;
            int bEnd = sEnd + smOff * yr;

            if (bStart < 0)
                bStart = 0;
            if (bEnd > csize)
                bEnd = csize;

            double maxMag2 = 0;

            for (int j = bStart; j < bEnd; j++)
                if (maxMag2 < tempBuf[j])
                    maxMag2 = tempBuf[j];

            const float scale = 255 / (clamp(maxMag2, minMag, maxMag) - minMag);
            for (int j = sStart; j < sEnd; j++) {
                unsigned char idx = clamp((tempBuf[j] - minMag) * scale, 0.0, 254.0);

                tempBuf[j] = idx * pScale[idx];
            }
        }
    }
    else
    {
        const double scale = 255 / (maxMag - minMag);

        #pragma omp parallel for
        for (int i = 0; i < csize; i++) {
            unsigned char idx = clamp((tempBuf[i] - minMag) * scale, 0.0, 254.0);

            tempBuf[i] = idx * pScale[idx];
        }
    }

    delete[] pScale;

    if (cancelJob)
        goto cancel;

    #pragma omp parallel for
    for (int i = 0; i < csize; i++)
        (*this->colormap)[i] = renderedGradient[(unsigned char)(tempBuf)[i]];

    cancel:

    *ready = true;
    (*version)++;

    //DispDebug("Spectra color time: %lf\n", timer.Now());

    processing = false;
}
void ColorizerSpectra::Apply(uint* colormap, ulong size) {
    if (processing) {
        DispWarning("ColorizerSpectra::Apply", "Operation already in progress");

        while (processing)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    processing = true;

    cancelJob = true;

    if (worker != nullptr && worker->joinable())
        worker->join();
    
    cancelJob = false;

    delete worker;

    worker = new std::thread(&ColorizerSpectra::apply, this);
}
bool ColorizerSpectra::RenderWidget(int id, bool odd, bool range) {
    ImGui::PushID(id);

    bool changed = false;

    //todo: remember this to avoid extra work
    char csid[32];
    char vid[32];
    char wid[32];
    char visid[32];
    snprintf(csid, 32, "Edit Color##(%d)", id);
    snprintf(vid, 32, "##val (%d)", id);
    snprintf(wid, 32, "##width (%d)", id);
    snprintf(visid, 32, "##vis (%d)", id);

    ImVec2 cursor = ImGui::GetCursorScreenPos();
    ImVec2 frameEnd = ImVec2(cursor.x + ImGui::GetWindowContentRegionWidth(), cursor.y + 160 + (autoscale ? 30 : 0));

    ImGui::RenderFrame(cursor, frameEnd, odd? 0x16000000 : 0x32000000, true, 5);

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10, 2));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 6);
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2);
    ImGui::Text("%s", fieldName);
    ImGui::PopStyleVar();

    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2);

    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 6);

    Gradient* hoveredGradient;
    bool hovered = false;
    changed |= GradientPicker(csid, &gradient, ImVec2(25, 25), gradients, &hoveredGradient, &hovered);

    if (changed)
        Set_Color(gradient);

    if (hovered) {
        if (!lastHoverUpdate || hoveredGradient != lastHoverGradient) {
            Set_Color(hoveredGradient, false);
            changed = true;
            //printf("Set to %s\n", hoveredGradient->Name);
            lastHoverGradient = hoveredGradient;
        }
    }
    else if (lastHoverUpdate) { // change it back
        Set_Color(gradient);
        changed = true;
        lastHoverGradient = nullptr;
        //printf("Set to %s\n", gradient->Name);
    }

    lastHoverUpdate = hovered;

    ImGui::SameLine();

    if (ImGui::Checkbox("Transparent", &transparent)) {
        changed = true;
        Set_Color(gradient);
    }

    ImGui::SameLine();

    if (filters == nullptr)
        ImGui::BeginDisabled();
    
    if (ImGui::Checkbox("Filtered", &filter)) {
        changed = true;
    }

    if (filters == nullptr)
        ImGui::EndDisabled();

    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 6);
    float ddWidth = (ImGui::GetContentRegionAvail().x) / 2;
    ImGui::SetNextItemWidth(ddWidth - 6);
    int sc_sel = autoscale ? 1 : 0;
    if (ImGui::Combo("##scale1", &sc_sel, "Manual\0Autoscale\0", 2)) {
        autoscale = sc_sel == 0 ? false : true;

        changed = true;
        Set_Color(gradient);
    }

    ImGui::SameLine();
    ImGui::SetNextItemWidth(ddWidth - 6);
    if (ImGui::Combo("##scale2", &scale, "Linear\0Log\0Exp(Log10)\0", 3)) {
        changed = true;
        Set_Color(gradient);
    }

    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 6);
    ImGui::SetNextItemWidth(ddWidth * 2 - 5);
        
    if (ImGui::DragFloatRange2("##mag_minmax", &mag_min, &mag_max, 0.25, 0, 100, "Min: %.2f%%", "Max: %.2f%%")) {
        changed = true;
        Set_Color(gradient);
    }

    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 6);
    ImGui::SetNextItemWidth(ddWidth * 2 - 5);
    if (ImGui::DragFloat("##pareto", &paretoDeg, 0.1, -100, 10, paretoDeg > -1 && paretoDeg < 1 ? "Pareto: Off" : "Pareto: %.2f")) {
        changed = true;
        Set_Color(gradient);
        pareto(64, paretoDeg, paretoPreview);
    }
    if (ImGui::IsItemHovered() && (paretoDeg <= -1 || paretoDeg >= 1)) {
        ImGui::BeginTooltip();

        const ImVec2 plotSize = ImVec2(128, 128);
        const auto plotFlags = ImPlotFlags_CanvasOnly | ImPlotFlags_AntiAliased;
        const auto plotAxisFlags = ImPlotAxisFlags_NoDecorations;

        ImPlot::FitNextPlotAxes();
        if (ImPlot::BeginPlot("ParetoPlot", NULL, NULL, plotSize, plotFlags, plotAxisFlags, plotAxisFlags)) {
            ImPlot::PlotLine("##val", paretoPreview, 64);
            ImPlot::EndPlot();
        }

        ImGui::EndTooltip();
    }

    if (autoscale) {
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 6);
        ImGui::SetNextItemWidth(ddWidth * 2 - 5);
        if (ImGui::DragInt("##as_smooth", &as_smooth, 2, 0, 256, as_smooth == 0 ? "Smooth: Off" : "Smooth: %d bins")) {
            changed = true;
            Set_Color(gradient);
        }
    }

    ImGui::PopID();

    return changed;
}
void ColorizerSpectra::Set_Color(Gradient* color, bool persist) {
    // For when we're just hovering over a gradient
    if (persist)
        gradient = color;

    renderedGradient = MakeColorGradient(color, renderSize);

    union color {
        uint value;
        struct {
            unsigned char r;
            unsigned char g;
            unsigned char b;
            unsigned char a;
        } rgba;
    };

    union color* cmap = (union color*)renderedGradient;

    if (transparent)
        for (int i = 0; i < renderSize / 2; i++)
            cmap[i].rgba.a = i * (255.0 / (renderSize / 2));
}
Gradient* ColorizerSpectra::Get_Color() {
    return gradient;
}

void TBDIQColor(PlotField* tbdpd) {
    auto bfile = reinterpret_cast<TBDFile*>(tbdpd->File);

    if (bfile->AssocSig() == nullptr) {
        DispWarning("TBDIQColor", "No associated signal file");
        return;
    }

    SignalCDIF* iq        = bfile->AssocSig()->Data();
    uint* const cMap      = bfile->AssocSig()->TOAColormap();
    const auto  elCount   = *iq->ElementCount();
    const auto  sampRate  = 1.0 / iq->SampleInterval();
    const auto  startTime = iq->TOA()[0];
    const auto  stride    = iq->Stride();
    const auto  pdCount   = *tbdpd->ElementCount;
    const uint  mult      = stride == 1 ? 1 : 2;

    // May not be needed
    std::fill_n(cMap, elCount, 0xFFFFFFFF);

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
        std::fill_n(cMap + startEl - 1, pdElCount, tbdpd->Colormap[i]);
    }

    //std::fill_n(cMap, elCount, 0xFFFFFFFF);
}

void ColorStack::GenerateMap(uint* colormap, ulong size, bool* mask)
{
    // Reset
    std::fill_n(colormap, size, baseColor);

    // Apply layers in order
    for (auto colorizer : colors)
        colorizer->Apply(colormap, size);


    if (mask == nullptr)
        return;

    union color {
        uint value;
        struct {
            unsigned char r;
            unsigned char g;
            unsigned char b;
            unsigned char a;
        } rgba;
    };

    color* const cmap = (color*)colormap;

    // Change opacity of filtered out elements
    // This is what causes them to be faded when shift is pressed
    for (ulong i = 0; i < size; i++)
        if (!mask[i])
            cmap[i].rgba.a = 128;
}
void ColorStack::GenerateSpectraMaps() {
    // Spectra colorizer is only meant to be used with one field, so all information is stored internally
    for (auto sc : spectraColors)
        sc->Apply();
}
void ColorStack::Set_SpectraColor(Gradient* color) {
    for (auto sc : spectraColors)
        sc->Set_Color(color);
}
void ColorStack::AddColor(Colorizer* color) {
    if (color->Get_Type() == Colorizer::Type::Point)
        colors.push_back(color);
    else
        spectraColors.push_back((ColorizerSpectra*)color);
}
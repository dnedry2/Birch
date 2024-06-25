// Checked on: 2023-03-05

#include <assert.h>
#include <cstdio>
#include <math.h>
#include <cstring>
#include <numeric>
#include <algorithm>

#include "window.hpp"

#include "fft.h"
#include "include/birch.h"

#include "imgui.h"
#include "implot.h"

using namespace Birch;
using std::accumulate;

constexpr float CALC_LEN = 1024.0;

static ImPlotContext* widgetContext = nullptr;

const char* WindowFunc::Name() {
    return name;
}
WindowFunc::WindowFunc(const char* name) {
    this->name = strdup(name);
}
WindowFunc::~WindowFunc() {
    delete[] name;
    delete[] freqResponse;
    delete[] ampResponse;
}
float WindowFunc::CohGain() {
    if (cohGainSet)
        return cohGain;

    const double* window = Build(CALC_LEN);

    double sum = accumulate(window, window + (uint)CALC_LEN, 0.0);

    delete[] window;

    cohGain = sum / CALC_LEN;
    cohGainSet = true;

    return cohGain;
}
float WindowFunc::EqNBW() {
    if (eqNBWSet)
        return eqNBW;

    const double* window = Build(CALC_LEN);

    double top = 0;
    for (int i = 0; i < CALC_LEN; i++)
        top += pow(abs(window[i]), 2);
    
    delete[] window;

    eqNBW = (top * CALC_LEN) / pow(CohGain() * CALC_LEN, 2);
    eqNBWSet = true;

    return eqNBW;
}
float WindowFunc::ProcGain() {
    if (procGainSet)
        return procGain;

    procGain = 10 * log10(CohGain() * EqNBW());
    procGainSet = true;

    return procGain;
}
float WindowFunc::ScalLoss() {
    if (scalLossSet)
        return scalLoss;

    const double* window = Build(CALC_LEN);

    double sn = 0, cs = 0;
    for (int i = 0; i < CALC_LEN; i++) {
        sn  += window[i] * cos((M_PI * i) / CALC_LEN);
        cs  += window[i] * sin((M_PI * i) / CALC_LEN);
    }

    scalLoss = 20 * log10(sqrt(pow(sn, 2) + pow(cs, 2)) / (CohGain() * CALC_LEN));
    scalLossSet = true;

    delete[] window;

    return scalLoss;
}
float WindowFunc::WorstProcLoss() {
    if (worstProcLossSet)
        return worstProcLoss;

    worstProcLoss = ProcGain() + ScalLoss();
    worstProcLossSet = true;

    return worstProcLoss;
}
float WindowFunc::OvrCor25() {
    if (ovrCor25Set)
        return ovrCor25;

    ovrCor25 = ovrCor(0.25);
    ovrCor25Set = true;

    return ovrCor25;
}
float WindowFunc::OvrCor50() {
    if (ovrCor50Set)
        return ovrCor50;

    ovrCor50 = ovrCor(0.5);
    ovrCor50Set = true;

    return ovrCor50;
}
float WindowFunc::OvrCor75() {
    if (ovrCor75Set)
        return ovrCor75;

    ovrCor75 = ovrCor(0.75);
    ovrCor75Set = true;

    return ovrCor75;
}
float WindowFunc::AmpFlat25() {
    if (ampFlat25 == 0)
        ampFlat25 = ampFlat(0.25);
    
    return ampFlat25;
}
float WindowFunc::AmpFlat50() {
    if (ampFlat50 == 0)
        ampFlat50 = ampFlat(0.5);
    
    return ampFlat50;
}
float WindowFunc::AmpFlat75() {
    if (ampFlat75 == 0)
        ampFlat75 = ampFlat(0.75);
    
    return ampFlat75;
}
float WindowFunc::MainLobeWidth3() {
    return 0;
}
float WindowFunc::MainLobeWidth6() {
    return 0;
}
float WindowFunc::PeakSideWidth3() {
    return 0;
}
float WindowFunc::PeakSideWidth6() {
    return 0;
}
float WindowFunc::PeakSideLevel() {
    return 0;
}
/*
float WindowFunc::OptimalOverlap() {
    if (optOvrSet)
        return optOvr;

    float step = 1.0 / 128.0;
    float cur = step;
    float best = 0;
    optOvr = 0;

    for (int i = 0; i < 128; i++) {
        if (cur >= 1)
            break;

        float can = abs(ampFlat(cur));
        if (can > best) {
            best = can;
            optOvr = cur;
        }

        cur += step;
    }

    ampFlat(optOvr);

    optOvrSet = true;
    return optOvr * 100;
}
*/
void WindowFunc::RenderWidget() {
    if (ImGui::BeginChild("winStats", ImVec2(-1, -1))) {
        if (widgetContext == nullptr)
            widgetContext = ImPlot::CreateContext();

        auto lastCtx = ImPlot::GetCurrentContext();
        ImPlot::SetCurrentContext(widgetContext);

        ImGui::SetWindowFontScale(1.15);
        ImGui::Text("%s Window", Name());
        ImGui::SetWindowFontScale(1);

        ImGui::Separator();

        if (ImGui::BeginChild("winStats2", ImVec2(-1, -1))) {
            float width = ImGui::GetContentRegionAvail().x - 16;
            const ImVec2 plotSize = ImVec2(width / 2, 128);
            const auto plotFlags = ImPlotFlags_CanvasOnly | ImPlotFlags_AntiAliased;
            const auto plotAxisFlags = ImPlotAxisFlags_NoDecorations;

            ImPlot::FitNextPlotAxes();
            if (ImPlot::BeginPlot("Amp Response", NULL, NULL, plotSize, plotFlags, plotAxisFlags, plotAxisFlags)) {
                ImPlot::PlotLine("##mag", AmpResponse(512), 512);
                ImPlot::EndPlot();
            }

            ImGui::SameLine();

            ImPlot::FitNextPlotAxes();
            if (ImPlot::BeginPlot("Freq Response", NULL, NULL, plotSize, plotFlags, plotAxisFlags, plotAxisFlags)) {
                ImPlot::PlotLine("##freq", FreqResponse(64), freqResponseLen / 2);
                ImPlot::EndPlot();
            }


            ImVec2 curPos = ImGui::GetCursorPos();

            ImGui::Text("Coherent Gain: %.3f", CohGain());
            ImGui::Text("Processing Gain: %.3f", ProcGain());
            ImGui::Text("EqNBW: %.3f", EqNBW());
            ImGui::Text("Scalloping Loss: %.3f", ScalLoss());
            ImGui::Text("Worst Processing Loss: %.3f", WorstProcLoss());
            //ImGui::Text("Optimal Overlap: %.3f%%", OptimalOverlap());


            const float xPos = curPos.x + width / 2 + 10;

            ImGui::SetCursorPos(curPos);
            ImGui::SetCursorPosX(xPos);
            ImGui::Text("Overlap Correlation:");
            ImGui::SetCursorPosX(xPos);
            ImGui::Text("\t@ 25: %.3f", OvrCor25());
            ImGui::SetCursorPosX(xPos);
            ImGui::Text("\t@ 50: %.3f", OvrCor50());
            ImGui::SetCursorPosX(xPos);
            ImGui::Text("\t@ 75: %.3f", OvrCor75());
            
            ImGui::SetCursorPosX(xPos);
            ImGui::Text("Amplitude Flatness:");
            ImGui::SetCursorPosX(xPos);
            ImGui::Text("\t@ 25: %.3f", AmpFlat25());
            ImGui::SetCursorPosX(xPos);
            ImGui::Text("\t@ 50: %.3f", AmpFlat50());
            ImGui::SetCursorPosX(xPos);
            ImGui::Text("\t@ 75: %.3f", AmpFlat75());

/*
            if (overlap != nullptr) {
                ImPlot::FitNextPlotAxes();
                if (ImPlot::BeginPlot("Optimal Overlap", NULL, NULL, plotSize, plotFlags, plotAxisFlags, plotAxisFlags)) {
                    ImPlot::PlotLine("##ovr", overlap, overlapLen);
                    ImPlot::EndPlot();
                }
            }
*/
        }
        ImGui::EndChild();


        ImPlot::SetCurrentContext(lastCtx);
    }
    ImGui::EndChild();
}
/*
double* WindowFunc::overlapAmpResp(float ovr, uint* len) {
    uint ovrLen = (uint)(CALC_LEN * (1.0 - ovr));
    uint its = ceil(1.0 / (1.0 - ovr)) + 1;

    const double* window = Build(CALC_LEN);
    double* sumWin = new double[CALC_LEN * its];

    std::fill_n(sumWin, CALC_LEN * its, 0.0);

    for (uint i = 0; i < its; i++)
        for (uint j = 0; j < CALC_LEN; j++)
            sumWin[i * ovrLen + j] += window[j];

    *len = ((its - 1) * ovrLen + CALC_LEN);

    return sumWin;
}
*/
float WindowFunc::ovrCor(float ovr) {
    assert(0 < ovr && ovr <= 1); // Must be between 0 and 1

    const double* window = Build(CALC_LEN);

    float sumSq = 0, tp = 0;

    for (int i = 0; i < CALC_LEN - 1; i++) {
        tp    += window[i] * window[i + (int)((1 - ovr) * CALC_LEN)];
        sumSq += window[i] * window[i];
    }

    delete[] window;

    return tp / sumSq;
}
float WindowFunc::ampFlat(float ovr) {
    /*
    assert(0 < ovr && ovr <= 1); // Must be between 0 and 1

    uint len = 0;
    const double* 

    const uint start = ovrLen;
    const uint end   = start + ovrLen;
    float low  = *std::min_element(sumWin + start, sumWin + end);
    float high = *std::max_element(sumWin + start, sumWin + end);



    delete[] window;

    return low / high * 100;
    */
   return 0;
}

float lobeMeas(double* lobe, int len) {
    return 0;
}


const double* WindowFunc::FreqResponse(int len) {
    const int mult = 8;

    if (freqResponse == nullptr || freqResponseLen != len * mult) {
        delete[] freqResponse;

        freqResponseLen = len * (mult / 2);
        freqResponse = new double[len * (mult)];

        auto fftIn  = new Complex<double>[len * mult];
        auto fftOut = new Complex<double>[len * mult];

        auto amp = Build(len);

        Complex<double> zero = { 0, 0 };
        std::fill_n(fftIn, len * mult, zero);

        for (int i = 0; i < len; i++)
            fftIn[i].R = amp[i];
        
        delete[] amp;

        fft(fftIn, fftOut, len * mult);

        delete[] fftIn;

        const int halfLen = (len * mult) / 2;

        for (int i = 0; i < halfLen; i++)
            freqResponse[i] = 10 * log10(sqrt(fftOut[i].R * fftOut[i].R + fftOut[i].I * fftOut[i].I));

        delete[] fftOut;
    }

    return freqResponse;
}
const double* WindowFunc::AmpResponse(int len) {
    if (ampResponse == nullptr || ampResponseLen != len) {
        delete[] ampResponse;

        ampResponse = Build(len);
        ampResponseLen = len;
    }

    return ampResponse;
}

void WindowFunc::Init() {
    CohGain();
    EqNBW();
    ProcGain();
    ScalLoss();
    WorstProcLoss();
    OvrCor25();
    OvrCor50();
    OvrCor75();
    AmpFlat25();
    AmpFlat50();
    AmpFlat75();
    MainLobeWidth3();
    MainLobeWidth6();
    PeakSideWidth3();
    PeakSideWidth6();
    PeakSideLevel();
}
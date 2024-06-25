#include "bfile.hpp"
#include "logger.hpp"

#include "plotRenderer.hpp"
#include "color.hpp"
#include "filter.hpp"

#include "cdif.hpp"
#include "fir.hpp"

#include "defines.h"

#include <string>
#include <algorithm>

using namespace Birch;
using std::vector;
using std::string;

constexpr unsigned spectraIdx = 11;
constexpr unsigned peaksIdx   = 10;

static inline unsigned getStride(unsigned maxElements, unsigned long long totalElements)
{
    const unsigned stride = ceil(totalElements / (double)maxElements);
    return stride == 0 ? 1 : stride;
}

FileType SignalFile::Type() {
    return FileType::Signal;
}
const vector<PlotField*>* SignalFile::Fields() {
    return &fields;
}
PlotField* SignalFile::Field(const char* name) {
    auto len  = strnlen(Filename(), 256) + 2;
    
    for (auto& field : fields) {
        auto len2 = strnlen(field->Name, 128);
        if (!strncmp(field->Name, name, len2 - len)) {
            return field;
        }
    }

    return nullptr;
}
void SignalFile::RenderSidebar() {
    plugin->RenderInfo(reinterpret_cast<void*>(ImGui::GetCurrentContext()));
}
Timespan SignalFile::FileTime() {
    return plugin->Time();
}
bool SignalFile::FetchYield() {
    return false;
}
bool SignalFile::Fetching() {
    return fetching;
}
Plugin* SignalFile::IOPlugin()  {
    return plugin;
}
const char* SignalFile::Filename() {
    return plugin->SafeFileName();
}
SignalCDIF* SignalFile::Data() {
    return *currentBuffer;
}

SignalFile::SignalFile(PluginIQGetter* plugin, ColorStack* colormap, FilterStack* filterstack) {
    this->plugin = plugin;

    bufferA = new SignalCDIF(signalMaxSize, 1 / plugin->SampleRate());
    bufferB = new SignalCDIF(signalMaxSize, 1 / plugin->SampleRate());
    currentBuffer = &bufferA;

    // TODO: Ensure data was allocated

    colors  = colormap;
    filters = filterstack;

    toaMask       = new bool[signalMaxSize];
    toaColormap   = new unsigned[signalMaxSize];
    shadeColormap = new unsigned[signalMaxSize];

    for (unsigned i = 0; i < signalMaxSize; i++) {
        toaMask[i]       = true;
        toaColormap[i]   = colormap->Get_BaseColor();
        shadeColormap[i] = colormap->Get_BaseColor();
    }

    // TODO: I shouldn't be strduping here. need to find out why ... plotfield needs to be rewritten
    fields.push_back(new PlotField(strdup(string("AM##" + string(Filename())).c_str()),           strdup("IQ"),      &fetching, nullptr, nullptr, nullptr, nullptr, toaColormap, toaMask, this, shadeColormap));
    fields.push_back(new PlotField(strdup(string("AM - Mean##" + string(Filename())).c_str()),    strdup("Stats"),   &fetching, nullptr, nullptr, nullptr, nullptr, toaColormap, toaMask, this, shadeColormap));
    fields.push_back(new PlotField(strdup(string("PM##" + string(Filename())).c_str()),           strdup("IQ"),      &fetching, nullptr, nullptr, nullptr, nullptr, toaColormap, toaMask, this, shadeColormap));
    fields.push_back(new PlotField(strdup(string("PM (Wrapped)##" + string(Filename())).c_str()), strdup("IQ"),      &fetching, nullptr, nullptr, nullptr, nullptr, toaColormap, toaMask, this, shadeColormap));
    fields.push_back(new PlotField(strdup(string("FM##" + string(Filename())).c_str()),           strdup("IQ"),      &fetching, nullptr, nullptr, nullptr, nullptr, toaColormap, toaMask, this, shadeColormap));
    fields.push_back(new PlotField(strdup(string("FM - Mean##" + string(Filename())).c_str()),    strdup("Stats"),   &fetching, nullptr, nullptr, nullptr, nullptr, toaColormap, toaMask, this, shadeColormap));
    fields.push_back(new PlotField(strdup(string("SI##" + string(Filename())).c_str()),           strdup("IQ"),      &fetching, nullptr, nullptr, nullptr, nullptr, toaColormap, toaMask, this, shadeColormap));
    fields.push_back(new PlotField(strdup(string("SQ##" + string(Filename())).c_str()),           strdup("IQ"),      &fetching, nullptr, nullptr, nullptr, nullptr, toaColormap, toaMask, this, shadeColormap));
    fields.push_back(new PlotField(strdup(string("RI##" + string(Filename())).c_str()),           strdup("IQ"),      &fetching, nullptr, nullptr, nullptr, nullptr, toaColormap, toaMask, this, shadeColormap));
    fields.push_back(new PlotField(strdup(string("RQ##" + string(Filename())).c_str()),           strdup("IQ"),      &fetching, nullptr, nullptr, nullptr, nullptr, toaColormap, toaMask, this, shadeColormap));
    fields.push_back(new PlotField(strdup(string("Peaks##" + string(Filename())).c_str()),        strdup("Peak"),    &fetching, nullptr, nullptr, nullptr, nullptr, toaColormap, toaMask, this, shadeColormap));

    fields.push_back(new PlotField(strdup(string("Spectra##" + string(Filename())).c_str()),      strdup("Spectra"), &fetching, nullptr, nullptr, nullptr, nullptr, nullptr,     toaMask, this, shadeColormap));
    fields.back()->YLimits = { (plugin->SampleRate() / 2) / -1000000, (plugin->SampleRate() / 2) / 1000000 };

    fields[0]->Renderer = new RendererShaded(fields[0]);
    fields[1]->Renderer = new RendererLine(fields[1]);

    fields[2]->Renderer = new RendererLine(fields[2]);
    

    fields[3]->Renderer  = new RendererLine(fields[3]);
    fields[4]->Renderer  = new RendererLine(fields[4]);
    fields[5]->Renderer  = new RendererLine(fields[5]);
    fields[6]->Renderer  = new RendererLine(fields[6]);
    fields[7]->Renderer  = new RendererLine(fields[7]);
    fields[8]->Renderer  = new RendererLine(fields[8]);
    fields[9]->Renderer  = new RendererLine(fields[9]);

    fields[2]->Renderer->SetAutoFit(true);
    fields[spectraIdx]->Renderer = new RendererSpectra(fields[spectraIdx]);

    fields[peaksIdx]->Renderer = new RendererScatter(fields[peaksIdx]);

    for (auto field : fields)
        field->UseFilter = filters->Get_LinkPtr();
}
SignalFile::~SignalFile() {
    delete bufferA;
    delete bufferB;

    delete[] toaMask;
    delete[] toaColormap;
    delete[] shadeColormap;

    for (auto field : fields)
        delete field;

    delete[] mmTiming;
    delete[] mmAM->Data;
    delete[] mmPM->Data;
    delete[] mmFM->Data;
    delete[] mmRI->Data;
    delete[] mmRQ->Data;
    delete mmAM;
    delete mmPM;
    delete mmFM;
    delete mmRI;
    delete mmRQ;
}

void SignalFile::updateFields() {
    SignalCDIF* cBuf = *currentBuffer;

    // Set data to new buffer
    fields[0]->Data = cBuf->Amplitude();
    //fields[1]->Data = cBuf->Statistics.AmpMean();
    fields[1]->Data = cBuf->Amplitude();
    fields[2]->Data = cBuf->Phase();
    fields[3]->Data = cBuf->PhaseWrapped();
    fields[4]->Data = cBuf->Freq();
    //fields[4]->Data = cBuf->Statistics.FreqMean();
    fields[5]->Data = cBuf->Freq();
    fields[6]->Data = cBuf->SI();
    fields[7]->Data = cBuf->SQ();
    fields[8]->Data = cBuf->RI();
    fields[9]->Data = cBuf->RQ();
    fields[spectraIdx]->Data = cBuf->Spectra();

    bool tuned = tune != 0;
    fields[2]->Tuned = tuned;
    fields[3]->Tuned = tuned;
    fields[4]->Tuned = tuned;
    fields[6]->Tuned = tuned;
    fields[7]->Tuned = tuned;

    for (int i = 0; i <= 7; i++)
        fields[i]->Filtered = filtered;


    PlotField* spectra = fields[spectraIdx];
    
    spectra->ColormapReady = false;

    // Detect clipping regions
    auto clipping = std::vector<Birch::Span<double>>();
    
    if (plugin->ClipThreshold() != 0) {
        uint   window       = 256 / currentStride;
        bool   regionStart  = false;
        uint   regionWindow = 0;
        double lastEnd      = 0;

        Birch::Span<double> region;

        const auto clipLimits = Birch::Span<double>(-plugin->ClipThreshold(), plugin->ClipThreshold());

        for (ulong i = 0; i < *cBuf->ElementCount(); i++) {
            if (!clipLimits.Contains(cBuf->SI()[i]) || !clipLimits.Contains(cBuf->SQ()[i])) {
                regionWindow = 0;
                lastEnd      = cBuf->TOA()[i];

                if (!regionStart) {
                    region.Start = lastEnd;
                    regionStart  = true;
                }
            } else if (regionStart && regionWindow++ > window) {
                region.End = lastEnd;
                clipping.push_back(region);

                regionStart = false;
            }
        }

        // Close off last region
        if (regionStart) {
            region.End = cBuf->TOA()[*cBuf->ElementCount() - 1];
            clipping.push_back(region);
        }
    }


    unsigned it = 0;
    for (auto field : fields) {
        if (it++ != peaksIdx) {
            field->Timing = cBuf->TOA();
            field->ElementCount = cBuf->ElementCount();
        }

        field->LoadStatus = PlotField::Status::Full; // (*currentBuffer)->Stride() == 1 ? PlotField::Status::Full : PlotField::Status::Preview;

        field->ViewElementCount = *(cBuf->ElementCount()); //plugin->SpanElementCount(Timespan((*currentBuffer)->Time().Start, (*currentBuffer)->Time().End));

        if (field != spectra)
            field->Colormap = toaColormap;

        field->XRes = cBuf->SpectraXRes();
        field->YRes = cBuf->SpectraYRes();

        field->XLimits = cBuf->Time();
        field->YLimits = cBuf->SpectraFreqSpan();

        field->XOffset = cBuf->SpectraXOffset();
        field->YOffset = cBuf->SpectraYOffset();


        // Set clipping regions
        field->ErrorRegions = clipping;


        field->NeedsUpdate = false;

        field->IncVersion();
    }

    spectra->ElementCount = cBuf->SpectraCount();

    if (cBuf->PeaksTOA()->size() > 0) {
        fields[peaksIdx]->Data         = &cBuf->PeaksAmp()->front();
        fields[peaksIdx]->Timing       = &cBuf->PeaksTOA()->front();
        fields[peaksIdx]->ElementCount = &peaksCount;

        peaksCount = cBuf->PeaksTOA()->size();
    }

    try {
        if (spectra->Colormap == nullptr)
            delete[] spectra->Colormap;

        spectra->Colormap = new unsigned[(unsigned long long)*spectra->ElementCount];

        memset(spectra->Colormap, 254, *spectra->ElementCount);
    }
    catch(...) {
        DispError("SignalFile::updateFields", "Failed to allocate spectra colormap buffer!");
    }

    colors->GenerateSpectraMaps();

    // Determine max power in spectra
    spectra->MaxValue = *std::max_element(spectra->Data, spectra->Data + *spectra->ElementCount);

    // tbd filter / color
    if (filters->Get_Link() && assoc != nullptr) {
        auto pd = assoc->Field("PD");

        if (pd == nullptr) {
            DispError("SignalFile::updateFields", "Failed to find PD field!");
        } else {
            TBDIQFilter(pd);
            TBDIQColor(pd);
        }
    } else {
        const auto elCount = *cBuf->ElementCount();

        for (unsigned long i = 0; i < elCount; i++)
            toaColormap[i] = colors->Get_BaseColor();
    }

    // This should probably be in cdif reader
    #pragma omp parallel for
    for (uint i = 0; i < fields.size(); i++) {
        if (i == spectraIdx)
            continue;

        auto field = fields[i];

        if (field == nullptr || field->ElementCount == nullptr)
            continue;

        const auto elCount = *field->ElementCount;

        if (elCount == 0)
            continue;

        double min = field->Data[0];
        double max = field->Data[0];

        for (ulong j = 0; j < elCount; j++) {
            const double val = field->Data[j];

            if (val < min)
                min = val;
            if (val > max)
                max = val;
        }

        field->YLimits = { min, max };
    }

    buildShadedColors();
}
TBDFile*& SignalFile::AssocTBD() {
    return assoc;
}
void SignalFile::RefreshFields() {
    colors->GenerateSpectraMaps();

    // tbd filter / color
    if (filters->Get_Link() && assoc != nullptr) {
        auto pd = assoc->Field("PD"); // TODO: Need a way for the user to set this

        if (pd == nullptr) {
            DispError("SignalFile::RefreshFields", "Failed to find field 'PD' in associated TBD file!");
            return;
        } else {
            TBDIQFilter(pd);
            TBDIQColor(pd);
        }
    } else {
        const auto elCount = *(*currentBuffer)->ElementCount();

        for (unsigned long i = 0; i < elCount; i++)
            toaColormap[i] = colors->Get_BaseColor();
    }

    buildShadedColors();
}
void SignalFile::buildShadedColors() {
    for (int i = 0; i < signalMaxSize; i++) {
        shadeColormap[i] = toaColormap[i];
    }

    /*
    if ((*currentBuffer)->Stride() == 1) {
        for (int i = 0; i < signalMaxSize; i++) {
            unsigned color = toaColormap[i];
            ((char*)&color)[3] = 0x40;

            shadeColormap[i] = color;
        }
    } else {
        for (int i = 0; i < signalMaxSize; i++) {
            shadeColormap[i] = toaColormap[i];
        }
    }
    */
}
bool* SignalFile::TOAMask() {
    return toaMask;
}
unsigned* SignalFile::TOAColormap() {
    return toaColormap;
}
const PlotField* SignalFile::Minimap(const PlotField* field) {
    if (!mmBuilt)
        return nullptr;

    return mmMap[field];
}
double& SignalFile::Tune() {
    return tune;
}
double SignalFile::SampleRate() {
    return plugin->SampleRate();
}
void SignalFile::UpdateFields() {
    needsUpdate = true;
}
bool SignalFile::NeedsUpdate() {
    for (auto field : fields)
        if (field->NeedsUpdate)
            return true;
    
    return false;
}

void SignalFile::Fetch(Timespan time, bool live, volatile bool* cancelLoad, volatile float* progress) {
    fetching = true;
    stopLoad = true;

    for (auto field : fields)
        field->NeedsUpdate = false;

    auto elementCount = plugin->SpanElementCount(time);

    unsigned stride = getStride(signalMaxSize, elementCount);

    //set currently displayed fields to unloaded
    for (auto field : fields)
        field->LoadStatus = field->Shit;

    if (!live) {
        if (!time.Overlaps(loadedTime))
            live = true;
        else if ((time.Start < loadedTime.Start || time.End > loadedTime.End) && loadedTime.Length() / time.Length() < 0.1)
            live = true;
    }

    //if not live, select the buffer that is not being viewed to load data into
    SignalCDIF** buffer;

    if (!live && !firstLoad) {
        if (useBufferA) {
            useBufferA = false;
            buffer = &bufferA;
        } else {
            useBufferA = true;
            buffer = &bufferB;
        }
    } else {
        if (useBufferA) {
            buffer        = &bufferA;
            currentBuffer = &bufferA;
            useBufferA    = false;
        } else {
            buffer        = &bufferB;
            currentBuffer = &bufferB;
            useBufferA    = true;
        }
        updateFields();
    }

    firstLoad = false;

    RendererSpectra* spectraRender = static_cast<RendererSpectra*>(fields.back()->Renderer);


    vector<FIR*> filt = vector<FIR*>();

    filtered = false;
    for (auto f : *(filters->Get_Filters())) {
        if (f->Type() == FilterType::IQ && f->Get_Enabled()) {
            filt.push_back(static_cast<FIRFilter*>(f)->Get_FIR());
            filtered = true;
        }
    }

    //FIR* smooth = new SmoothFIR(2);
    //filt.push_back(smooth);
    //filtered = true;

    if (tempBuf != nullptr)
        filt.push_back(tempBuf);

    double bw = plugin->SampleRate() / 1000000;
   
    reader.Process(plugin, *buffer, time, stride, tune, &filt, spectraRender->Window(), spectraRender->DatSize(), spectraRender->Overlap(), spectraRender->FFTSize(), spectraRender->TexSize(), progress, cancelLoad);

/*
    bool haltLiveSpectra = false;
    auto spectraColor = [&]() {
        while (!haltLiveSpectra) {
            *
            auto spectra = fields[spectraIdx];

            spectra->ColormapReady = false;
            spectra->ElementCount  = (*buffer)->SpectraCount();
            spectra->Data          = (*buffer)->Spectra();
            spectra->IncVersion();
            spectra->XOffset = (*buffer)->SpectraXOffset();
            spectra->YOffset = (*buffer)->SpectraYOffset();

            printf("Calling all prayer warriors, we're about to generate spectra maps!\n");
            colors->GenerateSpectraMaps();
            *
printf("Calling all prayer warriors, we're about to generate spectra maps!\n");
            auto spectra = fields[spectraIdx];

            spectra->ColormapReady = false;
            spectra->ElementCount  = (*buffer)->SpectraCount();
            spectra->Data          = (*buffer)->Spectra();
            spectra->IncVersion();
            spectra->XOffset = (*buffer)->SpectraXOffset();
            spectra->YOffset = (*buffer)->SpectraYOffset();

           updateFields();

            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    };

    std::thread spectraColorThread;

    if (live)
        spectraColorThread = std::thread(spectraColor);
*/
    reader.Spectra(plugin, *buffer, time, Timespan(bw / -2, bw / 2), tune, &filt, spectraRender->Window(), spectraRender->DatSize(), spectraRender->Overlap(), spectraRender->FFTSize(), spectraRender->TexSize(), progress, cancelLoad);

    
    //delete smooth;


/*
    if (live) {
        haltLiveSpectra = true;

        if (spectraColorThread.joinable())
            spectraColorThread.join();
    }
*/

    //(*buffer)->Time() = time;

    delete[] (*buffer)->SpectraTotal();
    unsigned xr = (*buffer)->SpectraXRes();
    unsigned yr = (*buffer)->SpectraYRes();
    (*buffer)->SpectraTotal() = new double[xr];

    auto sb = (*buffer)->Spectra();

    for (unsigned y = 0; y < yr; y++) {
        for (unsigned x = 0; x < xr; x++) {
            if (y == 0) {
                (*buffer)->SpectraTotal()[x] = sb[x];
            }
            else if ((*buffer)->SpectraTotal()[x] < sb[x + y * xr]) {
                (*buffer)->SpectraTotal()[x] = sb[x + y * xr];
            }
        }
    }

    if (*cancelLoad) {
        useBufferA = !useBufferA;
        *progress = 0;
        fetching = false;
        return; 
    }

    currentStride = stride;
    loadedTime    = time;
    currentBuffer = buffer;

    updateFields();


    // Build minimaps on first load of file
    if (!mmBuilt && !*cancelLoad) {
        auto cBuf = *currentBuffer;

        mmCount = *cBuf->ElementCount();

        // Setup minimap plotfields
        mmTiming = new double[mmCount];

        mmAM = new PlotField(0, 0, 0, new double[mmCount], mmTiming, &mmCount, 0, 0, 0, 0);
        mmPM = new PlotField(0, 0, 0, new double[mmCount], mmTiming, &mmCount, 0, 0, 0, 0);
        mmWP = new PlotField(0, 0, 0, new double[mmCount], mmTiming, &mmCount, 0, 0, 0, 0);
        mmFM = new PlotField(0, 0, 0, new double[mmCount], mmTiming, &mmCount, 0, 0, 0, 0);
        mmRI = new PlotField(0, 0, 0, new double[mmCount], mmTiming, &mmCount, 0, 0, 0, 0);
        mmRQ = new PlotField(0, 0, 0, new double[mmCount], mmTiming, &mmCount, 0, 0, 0, 0);

        mmAM->Renderer = new RendererShaded(mmAM);

        // Copy data
        memcpy(mmTiming, cBuf->TOA(), mmCount * sizeof(double));

        memcpy(mmAM->Data, cBuf->Amplitude(),    mmCount * sizeof(double));
        memcpy(mmPM->Data, cBuf->Phase(),        mmCount * sizeof(double));
        memcpy(mmWP->Data, cBuf->PhaseWrapped(), mmCount * sizeof(double));
        memcpy(mmFM->Data, cBuf->Freq(),         mmCount * sizeof(double));
        memcpy(mmRI->Data, cBuf->RI(),           mmCount * sizeof(double));
        memcpy(mmRQ->Data, cBuf->RQ(),           mmCount * sizeof(double));

        // Build dict
        for (auto f : fields) {
            if (!strncmp(f->Name,      "AM", 2))
                mmMap[f] = mmAM;
            else if (!strncmp(f->Name, "PM ", 3)) // PM (Wrapped)
                mmMap[f] = mmWP;
            else if (!strncmp(f->Name, "PM", 2))  // PM
                mmMap[f] = mmPM;
            else if (!strncmp(f->Name, "FM", 2))
                mmMap[f] = mmFM;
            else if (!strncmp(f->Name, "RI", 2))
                mmMap[f] = mmRI;
            else if (!strncmp(f->Name, "RQ", 2))
                mmMap[f] = mmRQ;
            else if (!strncmp(f->Name, "SI", 2))
                mmMap[f] = mmRI;
            else if (!strncmp(f->Name, "SQ", 2))
                mmMap[f] = mmRQ;
            else
                mmMap[f] = nullptr;
        }

        mmBuilt = true;
    }

    if (assoc != nullptr && !*cancelLoad) {
        assoc->RefreshFields();
    }

    fetching = false;
}
SignalCDIF* SignalFile::FetchPortion(Timespan time, Birch::Span<double>* filter, double tune) {
    auto spanElCount = plugin->SpanElementCount(time);
    SignalCDIF* out = new SignalCDIF(spanElCount, 1 / plugin->SampleRate());

    bool dummy = false;
    //float progress = 0;

    RendererSpectra* spectraRender = static_cast<RendererSpectra*>(fields.back()->Renderer);

    vector<FIR*> filt = vector<FIR*>();

    auto win = new Hann();

    if (filter != nullptr) {
        auto f = new BandpassFIR(SampleRate(), 4096);

        f->SetBandpass(filter->Center(), filter->Length(), 4096, win, false);

        filt.push_back(f);
    }

    reader.Process(plugin, out, time, 1, tune, &filt, spectraRender->Window(), spectraRender->DatSize(), spectraRender->Overlap(), spectraRender->FFTSize(), spectraRender->TexSize(), gProgressBar, &dummy);

    for (auto f : filt)
        delete f;

    delete win;

    return out;
}
const char* SignalFile::Filepath() {
    return plugin->FilePath();
}
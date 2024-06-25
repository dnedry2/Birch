#include "bfile.hpp"
#include "logger.hpp"

#include "plotRenderer.hpp"
#include "color.hpp"
#include "filter.hpp"

#include <string>
#include <iostream>
#include <algorithm>

#ifdef FFT_IMPL_CUDA
#include "bfileTBD.cuh"
#endif

using namespace Birch;
using std::vector;
using std::string;

void calcDTOA(double* dtoa, const double* toa, const bool* mask, unsigned long elCount, bool refilter) {
    if (mask == nullptr) {
        #pragma omp parallel for
        for (unsigned long i = 1; i < elCount; i++) {
            dtoa[i] = toa[i] - toa[i - 1];
        }
    } else {
        if (!refilter) {
            #pragma omp parallel for
            for (unsigned long i = 1; i < elCount; i++) {
                if (mask[i]) {
                    dtoa[i] = toa[i] - toa[i - 1];
                }
            }
        } else {
            uint lastGood = 0;

            for (unsigned long i = 1; i < elCount; i++) {
                if (mask[i]) {
                    dtoa[i] = toa[i] - toa[lastGood];
                    lastGood = i;
                }
            }
        }
    }
}

static PlotField minimizeTBD(const double* values, const double* timing, ulong count) {
    const uint xRes = 4096; // TODO: Should be dependent on screen resolution
    const uint yRes = 128;
    const uint pCount = xRes * yRes;

    float* pValues = new float[pCount];
    float* pTiming = new float[pCount];
    bool*   pMask  = new bool[pCount];

    std::fill_n(pMask, pCount, false);

    auto vl = std::minmax_element(values, values + count);
    Birch::Span<double> valueLimits(*vl.first, *vl.second);
    Birch::Span<double> timingLimits(timing[0], timing[count - 1]);

    const float xScale = (xRes - 1) / timingLimits.Length();
    const float yScale = (yRes - 1) / valueLimits.Length();

    uint pSize = 0;

    for (uint i = 0; i < count; i++) {
        uint xPos = (timing[i] - timingLimits.Start) * xScale;
        uint yPos = (values[i] - valueLimits.Start)  * yScale;

        uint idx = yPos * xRes + xPos;

        if (pMask[idx])
            continue;

        pMask[idx] = true;

        pTiming[pSize] = timing[i];
        pValues[pSize] = values[i];

        pSize++;
    }

    delete[] pMask;

    double* pValuesOut = new double[pSize];
    double* pTimingOut = new double[pSize];

    std::copy_n(pValues, pSize, pValuesOut);
    std::copy_n(pTiming, pSize, pTimingOut);

    delete[] pValues;
    delete[] pTiming;

    //printf("%d -> %d \n", (int)count, (int)pSize);

    ulong* pSizeOut = new ulong[1];
    *pSizeOut = pSize;
    PlotField output = PlotField(0, 0, 0, pValuesOut, pTimingOut, pSizeOut, 0, 0, 0, 0);

    return output;
}

FileType TBDFile::Type() {
    return FileType::TBD;
}
const std::vector<PlotField*>* TBDFile::Fields() {
    return &fields;
}
Timespan TBDFile::FileTime() {
    return plugin->Time();
}
PlotField* TBDFile::Field(const char* name) {
    auto len  = strnlen(Filename(), 256) + 2;
    
    for (auto& field : fields) {
        auto len2 = strnlen(field->Name, 128);
        if (!strncmp(field->Name, name, len2 - len)) {
            return field;
        }
    }

    return nullptr;
}
void TBDFile::RefreshFields() {
    auto timer = Stopwatch();

    int fCnt = fields.size();

    #pragma omp parallel for
    for (int i = 0; i < fCnt; i++) {
        filters->GenerateMask(fields[i], false);
        colors->GenerateMap(fields[i]->Colormap, *(fields[i]->ElementCount), fields[i]->FilterMask);

        double yMin = fields[i]->Data[0];
        double yMax = fields[i]->Data[0];

        const unsigned long elCount = *(fields[i]->ElementCount);

        //auto dtoaTimer = Stopwatch();
        //#ifdef FFT_IMPL_CUDA
        //    calcDTOACUDA(dtoaPF->Data, dtoaPF->Timing, dtoaPF->FilterMask, *dtoaPF->ElementCount);
        //#else
            calcDTOA(dtoaPF->Data, dtoaPF->Timing, dtoaPF->FilterMask, *dtoaPF->ElementCount, true);
        //#endif
        //DispDebug("DTOA Time: %lfs\n", dtoaTimer.Now());

        for (unsigned long j = 0; j < elCount; j++) {
            if (fields[i]->FilterMask[j]) {
                if (fields[i]->Data[j] < yMin)
                    yMin = fields[i]->Data[j];
                if (fields[i]->Data[j] > yMax)
                    yMax = fields[i]->Data[j];
            }
        }

        fields[i]->XLimits = { fields[i]->Timing[0], fields[i]->Timing[elCount - 1] };
        fields[i]->YLimits = { yMin, yMax };

        fields[i]->LoadStatus = PlotField::Status::Full;
        fields[i]->DataVersion++;
    }

    //DispDebug("Refresh Time: %lfs", timer.Now());
}
bool TBDFile::FetchYield() {
    return false;
}
void TBDFile::Fetch(Birch::Timespan time, bool live, volatile bool* cancelLoad, volatile float* progress) {
    fetching = true;

    for (auto field : fields)
        field->NeedsUpdate = false;

    auto timer = Stopwatch();

    auto elCount = plugin->SpanElementCount(time);

    unsigned bufferOffset = 0;
    double* toaData = nullptr;

    auto updateFields = [&](unsigned long cnt) {
        plotElCount = cnt;

        //#pragma omp parallel for
        for (unsigned i = 0; i < fields.size(); i++) {
            fields[i]->Timing       = toaData + bufferOffset;
            fields[i]->Data         = buffer[i]->Data() + bufferOffset;
            fields[i]->ElementCount = &plotElCount;
            fields[i]->Ticks        = buffer[i]->Labels()->size();

            delete[] fields[i]->TickVals;
            delete[] fields[i]->TickText; // These are pointers to the tbdfield's fieldlabel copy

            if (buffer[i]->Labels()->size() > 0) {
                fields[i]->TickText = new const char*[fields[i]->Ticks];
                fields[i]->TickVals = new double[fields[i]->Ticks];

                int j = 0;
                for (const auto& fl : *buffer[i]->Labels()) {
                    fields[i]->TickVals[j] = fl->Value();
                    fields[i]->TickText[j] = fl->Label();

                    j++;
                }
            }

            if (!mmBuilt) {
                mmStorage.push_back(minimizeTBD(fields[i]->Data, fields[i]->Timing, plotElCount));

                mmStorage.back().Name = fields[i]->Name;

                mmMap[fields[i]->Name] = i;
            }
        }
        
        mmBuilt = true;

        RefreshFields();
    };

    if (!loadedTime.Contains(time)) {
        auto startEl = plugin->TimeIdx(time.Start);
        auto recCnt  = plugin->SubrecordCount();
        auto recLen  = plugin->RecordSize();

        const unsigned bufferSize = 1024 * 1024 * 512;
        const unsigned elsPerBuffer = floor(bufferSize / recLen);


        char* bufferA = new char[bufferSize];
        char* bufferB = new char[bufferSize];

        char* dataBuffer = bufferA;
        char* readBuffer = bufferB;

        plugin->Seek(startEl);


        std::thread procThread;

        unsigned long elsRead = 0;

        while (elsRead < elCount) {
            if (cancelLoad != nullptr && *cancelLoad) {
                if (procThread.joinable())
                    procThread.join();

                delete[] bufferA;
                delete[] bufferB;
                return;
            }

            // Handle the last buffer
            unsigned long elsToRead = elCount - elsRead;
            if (elsToRead > elsPerBuffer)
                elsToRead = elsPerBuffer;

            plugin->Read(readBuffer, elsToRead);

            if (procThread.joinable())
                procThread.join();

            std::swap(dataBuffer, readBuffer);

            procThread = std::thread([&](unsigned long offset) {                
                #pragma omp parallel for
                for (unsigned i = 0; i < recCnt; i++) {
                    const auto sbr = plugin->Format() + i;
                    ParseTBD(buffer[i]->Data() + offset, dataBuffer, elsToRead, recLen, sbr, buffer[i]);


                    // TODO: TEMP, NEED TO ADD SUPPORT FOR UNITS
                    if (!strncmp(sbr->Name, "PD", 2)) {
                        for (unsigned j = 0; j < elsToRead; j++)
                            buffer[i]->Data()[j] *= 1000000;
                    }
                    if (!strncmp(sbr->Name, "RF", 2)) {
                        for (unsigned j = 0; j < elsToRead; j++)
                            buffer[i]->Data()[j] *= 1e-6;
                    }
                }

                if (toaIdx != -1) {
                    int toa = -1;

                    const char* toaName = plugin->TOAName();
                    const auto  toaLen  = strnlen(toaName, 128);

                    for (unsigned i = 0; i < recCnt; i++) {
                        if (!strncmp(plugin->Format()[i].Name, toaName, toaLen)) {
                            toa = i;
                            break;
                        }
                    }

                    // TODO: Handle gracefully
                    if (toa == -1)
                        DispError("TBDFile::Fetch", "No TOA field found in file");

                    toaData = buffer[toa]->Data();
                    toaIdx  = toa;
                }

                updateFields(elsRead);
            }, elsRead);

            elsRead += elsToRead;

            if (progress != nullptr)
                *progress = (float)elsRead / (float)elCount;
        }

        if (procThread.joinable())
            procThread.join();

        loadedTime = time;

        delete[] bufferA;
        delete[] bufferB;
    } else {
        bufferOffset = plugin->TimeIdx(time.Start) - plugin->TimeIdx(loadedTime.Start);
        toaData = buffer[toaIdx]->Data();

        updateFields(elCount);
    }

    DispInfo("ReadTBD", "Read Time: %lfs", timer.Now());

    fetching = false;
}
void TBDFile::FetchPortion(Birch::Timespan time, std::vector<std::string>* fields, std::vector<double*>* data, uint* len) {
    uint fieldCount   = this->fields.size();
    uint fieldElCount = *this->fields[0]->ElementCount;

    fields->resize(fieldCount);
    data->resize(fieldCount);

    uint start = 0;
    uint end   = 0;

    for (int i = 0; i < fieldElCount; i++) {
        if (this->fields[0]->Timing[i] >= time.Start) {
            start = i;
            break;
        }
    }
    for (int i = 0; i < fieldElCount; i++) {
        if (this->fields[0]->Timing[i] >= time.End) {
            end = i;
            break;
        }
    }

    // Remove file names from fields
    for (uint i = 0; i < fieldCount; i++) {
        uint elCount = 0;

        (*fields)[i] = this->fields[i]->Name;
        (*data)[i]   = new double[end - start];

        const auto mask   = this->fields[i]->FilterMask;
        const auto values = this->fields[i]->Data;

        // Filter data
        for (uint j = start; j < end; j++) {
            if (mask[j]) {
                (*data)[i][elCount] = values[j];
                elCount++;
            }
        }

        (*fields)[i] = (*fields)[i].substr(0, (*fields)[i].find("#"));

        *len = elCount;
    }

    //*len = end - start;
}
bool TBDFile::Fetching() {
    return fetching;
}
Plugin* TBDFile::IOPlugin()  {
    return plugin;
}
const char* TBDFile::Filename() {
    return plugin->SafeFileName();
}

SignalFile*& TBDFile::AssocSig() {
    return assoc;
}

const PlotField* TBDFile::Minimap(const PlotField* field) {
    if (mmBuilt)
        return &mmStorage[mmMap[field->Name]];
    
    return nullptr;
}
TBDFile::TBDFile(Birch::PluginTBDGetter* plugin, ColorStack* colormap, FilterStack* filterstack) {
    this->plugin = plugin;

    colors  = colormap;
    filters = filterstack;

    unsigned sbrCnt = plugin->SubrecordCount();
    auto sbrs = plugin->Format();
    long long unsigned elCount = plugin->SpanElementCount(plugin->Time());
    buffer = std::vector<TBDField*>();

    for (unsigned i = 0; i < sbrCnt; i++) {
        // TODO: PlotField should be able to init itself. No idea why it's like this...
        auto pf = new PlotField(strdup(string(std::string(sbrs[i].Name) + "##" + string(Filename())).c_str()), strdup("PDW"), &fetching, nullptr, nullptr, nullptr, nullptr, new unsigned[elCount], new bool[elCount], this);
        pf->Renderer = new RendererScatter(pf);

        fields.push_back(pf);
        buffer.push_back(new TBDField(sbrs[i].Name, elCount));
    }

    dtoaPF = new PlotField(strdup(string(string("ΔTOA##") + string(Filename())).c_str()), strdup("PDW"), &fetching, nullptr, nullptr, nullptr, nullptr, new unsigned[elCount], new bool[elCount], this);
    dtoaPF->Renderer = new RendererScatter(dtoaPF);

    dtoaTF = new TBDField(u8"ΔTOA", elCount);

    fields.push_back(dtoaPF);
    buffer.push_back(dtoaTF);
}
TBDFile::~TBDFile() {
    for (auto& f : fields)
        delete f;
    for (auto& f : buffer)
        delete f;
    
    for (auto& f : mmStorage) {
        delete[] f.Data;
        delete[] f.Timing;
        delete   f.ElementCount;
    }
}

void TBDFile::RenderSidebar() {
    plugin->RenderInfo(reinterpret_cast<void*>(ImGui::GetCurrentContext()));
}

void TBDFile::UpdateFields() {
    needsUpdate = true;
}
bool TBDFile::NeedsUpdate() {
    return needsUpdate;
}
const char* TBDFile::Filepath() {
    return plugin->FilePath();
}
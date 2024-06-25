#ifndef __BFILE_H__
#define __BFILE_H__

#include <vector>

#include "include/birch.h"
#include "plot.hpp"
#include "color.hpp"
#include "filter.hpp"
#include "tbd.hpp"
#include "cdif.hpp"
#include "fir.hpp"

enum class FileType { Signal, TBD };

class BFile {
public:
    virtual FileType Type() = 0;
    virtual const std::vector<PlotField*>* Fields() = 0;
    virtual PlotField* Field(const char* name) = 0;
    virtual void RenderSidebar() = 0;
    virtual Birch::Timespan FileTime() = 0;
    virtual void RefreshFields() = 0;
    virtual bool Fetching() = 0;
    virtual Plugin* IOPlugin() = 0;
    virtual const char* Filename() = 0;
    virtual const char* Filepath() = 0;
    virtual void UpdateFields() = 0; // Reload any fields with their update flags set to true
    virtual bool NeedsUpdate() = 0; // Returns true if any fields have their update flags set to true

    // Return minimap data. Should be entire file, unfiltered
    virtual const PlotField* Minimap(const PlotField* field) = 0;

    //TODO: Implelemt these
    // Returns true if fetch needs to be called again to get the next bit of data
    virtual bool FetchYield() = 0;

    virtual void Fetch(Birch::Timespan time, bool live, volatile bool* cancelLoad, volatile float* progress = NULL) = 0;
    //virtual Signal1k* FetchPortion(Birch::Timespan time, Signal1k* buf = nullptr) = 0;

    BFile() { }
    BFile (const BFile&) = delete;
    BFile& operator= (const BFile&) = delete;
};

class TBDFile;

class SignalFile : public BFile {
public:
    FileType Type() override;
    const std::vector<PlotField*>* Fields() override;
    PlotField* Field(const char* name) override;
    void RenderSidebar() override;
    Birch::Timespan FileTime() override;
    void RefreshFields() override;
    bool FetchYield() override;
    bool Fetching() override;
    Plugin* IOPlugin() override;
    const char* Filename() override;
    void UpdateFields() override;
    bool NeedsUpdate() override;
    const char* Filepath() override;

    void Fetch(Birch::Timespan time, bool live, volatile bool* cancelLoad, volatile float* progress = NULL) override;
    SignalCDIF* FetchPortion(Birch::Timespan time, Birch::Span<double>* filter = nullptr, double tune = 0);

    TBDFile*& AssocTBD();
    bool* TOAMask();
    unsigned* TOAColormap();
    SignalCDIF* Data();
    double& Tune();
    double SampleRate();

    const PlotField* Minimap(const PlotField* field) override;

    explicit SignalFile(Birch::PluginIQGetter* plugin, ColorStack* colormap, FilterStack* filterstack);
    ~SignalFile();
private:
    void updateFields();
    void buildShadedColors();

    Birch::PluginIQGetter* plugin = nullptr;
    std::vector<PlotField*> fields = std::vector<PlotField*>();

    SignalCDIF* bufferA = nullptr;
    SignalCDIF* bufferB = nullptr;
    SignalCDIF** volatile currentBuffer = nullptr;
    bool useBufferA = true;

    // Minimap vectors
    // For IQ, just store a minimized version of each field
    PlotField* mmAM;
    PlotField* mmPM;
    PlotField* mmWP;
    PlotField* mmFM;
    PlotField* mmRI;
    PlotField* mmRQ;
    PlotField* mmSpec;

    ulong mmCount = 0;
    double* mmTiming = nullptr;

    std::map<const PlotField*, PlotField*> mmMap;

    bool mmBuilt = false;

    // TODO: Spectra minimap
    // std::vector<float> mmSpec;

    CDIFReader reader = CDIFReader(1073741824 / 2); // TODO: Set this to user selected memory

    Birch::Timespan loadedTime;
    unsigned currentStride = 1;
    bool firstLoad = true;

    bool*     toaMask       = nullptr;
    unsigned* toaColormap   = nullptr;
    unsigned* shadeColormap = nullptr;

    ColorStack*  colors;
    FilterStack* filters;

    volatile bool fetching = false;
    volatile bool stopLoad = false;

    TBDFile* assoc = nullptr;

    double tune = 0;

    bool needsUpdate = false;

    unsigned long peaksCount = 0;

    bool filtered = false;
 
    // TODO: Temporary workaround
    BandpassFIR* tempBuf = nullptr;
};

class TBDFile : public BFile {
public:
    FileType Type() override;
    const std::vector<PlotField*>* Fields() override;
    PlotField* Field(const char* name) override;
    void RenderSidebar() override;
    Birch::Timespan FileTime() override;
    void RefreshFields() override;
    bool Fetching() override;
    Plugin* IOPlugin() override;
    const char* Filename() override;
    void UpdateFields() override;
    bool NeedsUpdate() override;
    const char* Filepath() override;

    const PlotField* Minimap(const PlotField* field) override;

    bool FetchYield() override;
    void Fetch(Birch::Timespan time, bool live, volatile bool* cancelLoad, volatile float* progress = NULL) override;
    void FetchPortion(Birch::Timespan time, std::vector<std::string>* fields, std::vector<double*>* data, uint* len);

    SignalFile*& AssocSig();

    explicit TBDFile(Birch::PluginTBDGetter* plugin, ColorStack* colormap, FilterStack* filterstack);
    ~TBDFile();

private:
    Birch::PluginTBDGetter* plugin = nullptr;
    std::vector<PlotField*> fields = std::vector<PlotField*>();

    std::vector<TBDField*> buffer;
    unsigned toaIdx = 0;

    std::vector<float>* previewData = nullptr;

    Birch::Timespan loadedTime;

    ColorStack*  colors;
    FilterStack* filters;

    bool fetching = false;

    TBDField* dtoaTF = nullptr;
    PlotField* dtoaPF = nullptr;

    SignalFile* assoc = nullptr;

    bool needsUpdate = false;

    bool mmBuilt = false;
    std::map<const char*, int> mmMap;
    std::vector<PlotField> mmStorage;

    //TODO: Update plotfield to remove this
    volatile unsigned long plotElCount = 0;
};

BFile* BFileFactory(const char* path, Plugin* plugin, ColorStack* colormap, FilterStack* filterstack);

#endif
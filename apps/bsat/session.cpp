#include "session.hpp"
#include <thread>
#include <algorithm>

#include "logger.hpp"

using namespace Birch;

Session::Session(volatile float* progressBar)
{
    this->progressBar = progressBar;
    this->colormap = new ColorStack();
    this->filters  = new FilterStack();
}
Session::~Session()
{
}

const Birch::Timespan& Session::CurrentZoom()
{
    return currentView;
}
bool Session::IsFetching() {
    for (BFile* file : files)
        if (file->Fetching())
            return true;

    return false;
}
void Session::fetchData(BFile* file, Birch::Timespan time, bool live) {
    cancelFetch = false;
    file->Fetch(time, live, &cancelFetch, progressBar);
}
void Session::SetCurrentZoom(const Birch::Timespan& time, bool live, bool force)
{
    // Don't do anything if the time is the same
    if (time == currentView && !force)
        return;

    cancelFetch = true;

    for (auto thread : fetchThreads) {
        if (thread->joinable())
            thread->join();

        delete thread;
    }

    fetchThreads.clear();

    for (int i = 0; i < files.size(); i++)
        fetchThreads.push_back(new std::thread(&Session::fetchData, this, files[i], time, live));

    currentView = time;
}
bool Session::AddFile(char* path, BFile** outfile, Plugin* loader)
{
    BFile* file = BFileFactory(path, loader, colormap, filters);

    if (file == nullptr) {
        DispWarning("Session::AddFile", "%s", "Failed to add file to session!");
        return false;
    }

    bool first = files.size() == 0;

    auto ft = file->FileTime();
    if (files.size() == 0)
        fullTime = ft;
    if (ft.Start < fullTime.Start)
        fullTime.Start = ft.Start;
    if (ft.End > fullTime.End)
        fullTime.End = ft.End;

    files.push_back(file);

    if (outfile != nullptr)
        *outfile = file;

    // TEMP
    if (file->Type() == FileType::TBD)
        for (auto f : files)
            if (f->Type() == FileType::Signal)
                ((TBDFile*)file)->AssocSig() = (SignalFile*)f;
    if (file->Type() == FileType::Signal)
        for (auto f : files)
            if (f->Type() == FileType::TBD)
                ((SignalFile*)file)->AssocTBD() = (TBDFile*)f;


    // Update fields
    fields.clear();

    for (auto f : files) {
        auto ff = f->Fields();

        // Add color layer for spectra fields
        if (f->Type() == FileType::Signal) {
            for (auto field : *ff) {
                if (!strncmp(field->Catagory, "Spectra", 7)) {
                    colormap->AddColor(new ColorizerSpectra(&field->Data, &field->Colormap, &field->ElementCount, file->Filename(), &field->XRes, &field->YRes, &field->ColormapReady, &field->DataVersion, (*gGradients)[37], gGradients, filters));
                    break;
                }
            }
        }

        fields.insert(fields.end(), ff->begin(), ff->end());
    }

    // Set zoom to full time if first file
    if (first)
        SetCurrentZoom(fullTime, true);
    

    // Inform views of new file
    for (auto view : views)
        view->SessionAddFileCallback(file);

    return true;
}
std::vector<BFile*>* Session::Files()
{
    return &files;
}

void Session::SortMeasurements() {
    const uint mCount = measurements.size();

    for (uint i = 0; i < mCount; i++) {
        const PlotField* field = measurements[i]->Field();
        
        uint start = i;
        uint end   = i;

        for (; end < mCount; end++)
            if (measurements[end]->Field() != field)
                break;

        i = end - 1;

        std::sort(&measurements[start], &measurements[end], [](Measurement* const a, Measurement* const b) { return a->Position()->X < b->Position()->X; } );
    }
}
void Session::AddMeasurement(Measurement* measurement) {
    measurements.push_back(measurement);

    SortMeasurements();
}
void Session::AddMeasurements(const std::vector<Measurement*>& measurements) {
    for (const auto measurement : measurements)
        this->measurements.push_back(measurement);

    SortMeasurements();
}
std::vector<Measurement*>* Session::Measurements() {
    return &measurements;
}

void Session::RefreshFields() {
    for (auto file : files)
        file->RefreshFields();
}
const Birch::Timespan& Session::TotalTime() {
    return fullTime;
}
bool Session::NeedsUpdate() {
    for (auto file : files)
        if (file->NeedsUpdate())
            return true;

    return false;
}

void Session::ReloadFields() {
    SetCurrentZoom(currentView, false, true);
}

std::vector<PlotField*> Session::Fields() {
    return fields;
}

ColorStack* Session::Colormap() {
    return colormap;
}
FilterStack* Session::Filters() {
    return filters;
}

void Session::AddView(IView* view) {
    views.push_back(view);
}
std::vector<IView*>* Session::Views() {
    return &views;
}
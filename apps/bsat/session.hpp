/********************************
* Birch session - contains current view information and manages loaded files
*
* Author: Levi Miller
* Date created: 12/7/2020
*********************************/

#ifndef _SESSION_
#define _SESSION_

#include <thread>
#include <vector>

#include "imgui.h"
#include "implot.h"

//#include "analysis.h"
#include "stopwatch.hpp"
#include "bfile.hpp"

#include "include/birch.h"

#include "annotation.hpp"
#include "measure.hpp"

#include "view.hpp"

class Session {
public:
    // The portion of the file currently loaded
    const Birch::Timespan& CurrentZoom();
    // Change portion of the file currently loaded
    void SetCurrentZoom(const Birch::Timespan& time, bool live, bool force = false);
    // The total time span of the file
    const Birch::Timespan& TotalTime();

    // Is data currently being loaded
    bool IsFetching();

    // Add a file to the session
    bool AddFile(char* path, BFile** outfile, Plugin* loader);

    // Add a measurement to the session
    void AddMeasurement(Measurement* measurement);
    void AddMeasurements(const std::vector<Measurement*>& measurements);
    void SortMeasurements();

    // Add a view to the session
    void AddView(IView* view);
    std::vector<IView*>* Views();

    std::vector<Measurement*>* Measurements();

    // Refresh data for all fields
    void RefreshFields();
    // Reload data for all fields
    void ReloadFields();

    std::vector<BFile*>* Files();

    bool NeedsUpdate();

    std::vector<PlotField*> Fields();

    ColorStack*  Colormap();
    FilterStack* Filters();

    ImVec4& Color();

    Session(volatile float* progressBar);
    ~Session();

private:
    void fetchData(BFile* file, Birch::Timespan time, bool live);

    std::vector<std::thread*> fetchThreads;

    std::vector<BFile*> files;
    std::vector<IView*> views;
    std::vector<Measurement*> measurements;

    ColorStack* colormap;
    FilterStack* filters;

    ImVec4 color = ImVec4(0, 0.25, .5, 1);

    std::vector<PlotField*> fields;

    //The current Birch::Timespan being displayed
    Birch::Timespan currentView;
    //The full Birch::Timespan of the session
    Birch::Timespan fullTime = Birch::Timespan(0, 0);
    //Progressbar control
    volatile float* progressBar;
    //Stop fetching files
    volatile bool cancelFetch = false;
};
#endif
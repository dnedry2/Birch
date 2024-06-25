#include "plotter.hpp"
#include "logger.hpp"

#include <string>

static std::string tuneToolHelp = 
    "Adds a frequency shift to the signal.\n"
    "Left click and drag on the PM field or single click in FM.";

Plotter::TuneTool::TuneTool(Plotter* plotter) : Tool("Tune", tuneToolHelp.c_str(), true, textures["bicons/IconTune.svg"], textures["bicons/IconTune.svg"], textures["bicons/IconTune.svg"], ImVec2(-2, -2), ImVec2(-2, -2), ImGuiKey_T), plotter(plotter) { }

void Plotter::TuneTool::on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) {
    controls->VerticalMod = ImGuiModFlags_Alt;
    controls->HorizontalMod = ImGuiModFlags_Alt;
    controls->BoxSelectMod = ImGuiModFlags_Alt;

    controls->BoxSelectButton = -1;
    controls->PanButton = -1;

    *flags = ImPlotFlags_Crosshairs;
}

void Plotter::TuneTool::on_click(Plot* plot) {
    if (!inProgress && ImPlot::IsPlotHovered()) {
        start = ImPlot::GetPlotMousePos();
        inProgress = true;
    }
}

void Plotter::TuneTool::on_update(Plot* plot) {
    if (inProgress && !strncmp(plot->SelectedField()->Name, "PM", 2)) {
        if (!ImPlot::IsPlotHovered()) {
            inProgress = false;
            return;
        }

        auto end = ImPlot::GetPlotMousePos();

        double xa[] = { start.x, end.x };
        double ya[] = { start.y, end.y };

        ImPlot::PlotLine("Tune", xa, ya, 2);
    }
    if (!strncmp(plot->SelectedField()->Name, "Spectra", 7)) {
        auto pltField = plot->SelectedField();

        double tune = reinterpret_cast<SignalFile*>(pltField->File)->Tune() / -1000000;

        if (ImPlot::DragLineY("##tune_spec_drag", &tune, false)) {
            reinterpret_cast<SignalFile*>(pltField->File)->Tune() = tune * -1000000;
            dragged = true;
        }
    }

    if (dragged && !ImGui::GetIO().MouseDown[0]) {
        dragged = false;
        plotter->ForceNextReload();
    }
}

void Plotter::TuneTool::on_release(Plot* plot) {
    if (ImPlot::IsPlotHovered()) {
        auto pltField = plot->SelectedField();

        if (reinterpret_cast<BFile*>(pltField->File)->Type() == FileType::Signal) {
            if (!strncmp(pltField->Name, "PM", 2)) {
                auto sr = reinterpret_cast<SignalFile*>(pltField->File)->SampleRate();
                auto end = ImPlot::GetPlotMousePos();

                double slope = (end.y - start.y) / (end.x - start.x);
                double shift = (1.0 / 360.0) * sr * (-slope) * 0.000001;

                reinterpret_cast<SignalFile*>(pltField->File)->Tune() += shift * 0.1;

                plotter->ForceNextReload();
            } else if (!strncmp(pltField->Name, "Spectra", 7)) {
                reinterpret_cast<SignalFile*>(pltField->File)->Tune() = ImPlot::GetPlotMousePos().y * -1000000;
                plotter->ForceNextReload();
            } else if (!strncmp(pltField->Name, "FM", 2)) {
                reinterpret_cast<SignalFile*>(pltField->File)->Tune() += ImPlot::GetPlotMousePos().y * -1000000;
                plotter->ForceNextReload();
            } else {
                DispWarning("Plotter::TuneTool::on_release", "%s", "Tuning field must be 'PM', 'FM', or 'Spectra'");
            }
        } else {
            DispWarning("Plotter::TuneTool::on_release", "%s", "Data type must be signal to tune");
        }
    }

    inProgress = false;
}

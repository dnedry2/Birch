#include "plotter.hpp"
#include "logger.hpp"

#include <string>

static std::string panToolHelp = "Left click and drag to pan the plot.";

Plotter::PanTool::PanTool() : Tool("Pan", panToolHelp.c_str(), false, textures["bicons/IconPan.svg"], textures["bicons/IconPan.svg"], textures["bicons/CursorFist.svg"], ImVec2(8, 8), ImVec2(8, 8), ImGuiKey_F) { }

void Plotter::PanTool::on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) {
    controls->VerticalMod = ImGuiModFlags_Shift;
    controls->HorizontalMod = ImGuiModFlags_Alt;
    controls->BoxSelectMod = ImGuiModFlags_None;

    controls->BoxSelectButton = -1;
    controls->PanButton = ImGuiMouseButton_Left;

    *flags = ImPlotFlags_None;
}
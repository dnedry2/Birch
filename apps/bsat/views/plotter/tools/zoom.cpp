#include "plotter.hpp"
#include "logger.hpp"

#include <string>

static std::string zoomToolHelp =
"Left click and drag to zoom in on the plot.\n"
"Double click to auto fit the y-axis.\n"
"Middle click to go back to the previous zoom level.";

Plotter::ZoomTool::ZoomTool() : Tool("Zoom", zoomToolHelp.c_str(), false, textures["bicons/IconZoom.svg"], textures["bicons/CursorZoom.svg"], textures["bicons/CursorZoom.svg"], ImVec2(5, 5), ImVec2(5, 5), ImGuiKey_D, "Zoom") { }

void Plotter::ZoomTool::on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) {
    controls->VerticalMod = ImGuiModFlags_None;
    controls->HorizontalMod = ImGuiModFlags_Shift;
    controls->BoxSelectMod = ImGuiModFlags_None;

    controls->BoxSelectButton = ImGuiMouseButton_Left;
    controls->PanButton = ImGuiMouseButton_Right;

    controls->BoxSelectCancelButton = ImGuiMouseButton_Right;

    controls->UnidirectionalZoom = true;

    *flags = ImPlotFlags_Crosshairs;
}

Plotter::ZoomToolX::ZoomToolX() : Tool("X-Zoom", zoomToolHelp.c_str(), false, textures["bicons/X-Zoom.svg"], textures["bicons/X-Zoom.svg"], textures["bicons/X-Zoom.svg"], ImVec2(5, 5), ImVec2(5, 5), ImGuiKey_D, "Zoom") { }

void Plotter::ZoomToolX::on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) {
    controls->VerticalMod = ImGuiModFlags_Shift;
    controls->HorizontalMod = ImGuiModFlags_None;
    controls->BoxSelectMod = ImGuiModFlags_Shift;

    controls->BoxSelectButton = ImGuiMouseButton_Left;
    controls->PanButton = ImGuiMouseButton_Right;

    controls->BoxSelectCancelButton = ImGuiMouseButton_Right;

    controls->UnidirectionalZoom = false;

    *flags = ImPlotFlags_Crosshairs;
}

Plotter::ZoomToolY::ZoomToolY() : Tool("Y-Zoom", zoomToolHelp.c_str(), false, textures["bicons/Y-Zoom.svg"], textures["bicons/Y-Zoom.svg"], textures["bicons/Y-Zoom.svg"], ImVec2(5, 5), ImVec2(5, 5), ImGuiKey_D, "Zoom") { }

void Plotter::ZoomToolY::on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) {
    controls->VerticalMod = ImGuiModFlags_Shift;
    controls->HorizontalMod = ImGuiModFlags_None;
    controls->BoxSelectMod = ImGuiModFlags_Shift;

    controls->BoxSelectButton = ImGuiMouseButton_Left;
    controls->PanButton = ImGuiMouseButton_Right;

    controls->BoxSelectCancelButton = ImGuiMouseButton_Right;

    controls->UnidirectionalZoom = false;

    *flags = ImPlotFlags_Crosshairs;
}

Plotter::ZoomToolXY::ZoomToolXY() : Tool("XY-Zoom", zoomToolHelp.c_str(), false, textures["bicons/XY-Zoom.svg"], textures["bicons/XY-Zoom.svg"], textures["bicons/XY-Zoom.svg"], ImVec2(5, 5), ImVec2(5, 5), ImGuiKey_D, "Zoom") { }

void Plotter::ZoomToolXY::on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) {
    controls->VerticalMod = ImGuiModFlags_Shift;
    controls->HorizontalMod = ImGuiModFlags_Shift;
    controls->BoxSelectMod = ImGuiModFlags_None;

    controls->BoxSelectButton = ImGuiMouseButton_Left;
    controls->PanButton = ImGuiMouseButton_Right;

    controls->BoxSelectCancelButton = ImGuiMouseButton_Right;

    controls->UnidirectionalZoom = false;

    *flags = ImPlotFlags_Crosshairs;
}
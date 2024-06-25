#include "plotter.hpp"
#include "logger.hpp"

#include <string>

static std::string colorToolHelp = 
    "Left click and drag to select a region to color.\n"
    "Hold shift to select multiple regions.\n"
    "Hold ctrl to use a single color.\n"
    "Hold alt to select the current y-axis range.";


Plotter::ColorTool::ColorTool(ColorStack* colors, Session* session, Colormap_Widget* widget)
                   : Tool("Color", colorToolHelp.c_str(), true, textures["bicons/IconColor.svg"], textures["bicons/CursorColor.svg"], textures["bicons/CursorColor.svg"], ImVec2(0, 0), ImVec2(0, 0), ImGuiKey_C),
                     globalColormap(colors), session(session), widget(widget) { }

void Plotter::ColorTool::on_selection(ImPlotInputMap *controls, ImPlotFlags *flags)
{
    controls->VerticalMod = ImGuiModFlags_Alt;
    controls->HorizontalMod = ImGuiModFlags_None;
    controls->BoxSelectMod = ImGuiModFlags_Alt;

    controls->QueryButton = ImGuiMouseButton_Left;
    controls->QueryMod = ImGuiModFlags_None;

    controls->BoxSelectButton = ImGuiMouseButton_Right;

    controls->PanButton = ImGuiMouseButton_Middle;

    *flags = ImPlotFlags_Query | ImPlotFlags_Crosshairs;
}

void Plotter::ColorTool::on_release(Plot* plot) {
    if (ImPlot::IsPlotQueried())
    {
        auto range = ImPlot::GetPlotQuery();

        Colorizer *layer;

        if (!ImGui::IsKeyDown(ImGuiKey_LeftShift) && !ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
            globalColormap->Get_Colors()->clear();

        if (widget->Solid())
            layer = new ColorizerSingle(&plot->SelectedField()->Data, plot->SelectedField()->Name, (range.Y.Max - range.Y.Min) / 2 + range.Y.Min, (range.Y.Max - range.Y.Min) / 2, widget->Get_SelectedColor());
        else
            layer = new ColorizerRange(&plot->SelectedField()->Data, plot->SelectedField()->Name,  (range.Y.Max - range.Y.Min) / 2 + range.Y.Min, (range.Y.Max - range.Y.Min) / 2, widget->Get_SelectedGradient(), gGradients);

        globalColormap->Get_Colors()->push_back(layer);

        session->RefreshFields();
        ImPlot::HidePlotQuery();
    }
}

void Plotter::ColorTool::on_middle_click(Plot* plot) {
    // Remove all color layers from this field
    if (ImPlot::IsPlotHovered()) {
        auto field   = plot->SelectedField();
        auto& colors = *session->Colormap()->Get_Colors();
        const auto fieldName = plot->SelectedField()->Name;

        std::vector<int> toRemove;

        const int size = colors.size();
        for (int i = 0; i < size; i++)
            if (!strcmp(colors[i]->FieldName(), fieldName))
                toRemove.push_back(i);

        for (int i = toRemove.size() - 1; i >= 0; i--) {
            delete colors[toRemove[i]];
            colors.erase(colors.begin() + toRemove[i]);
        }

        session->RefreshFields();
    }
}
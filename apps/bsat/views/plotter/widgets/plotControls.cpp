#include "plotter.hpp"

Plotter::PlotControls_Widget::PlotControls_Widget(int id, GLuint icon, Plotter* plotter) : Widget(id, icon, std::string("Plot Controls"))
{
    settings = new PlotSettings();
    this->plotter = plotter;
}
void Plotter::PlotControls_Widget::Render()
{
    if (beginWidget(this)) {
        ImGui::InputInt("Plots shown", &settings->PlotCount);

        {
            float col[4] = { settings->Background.x, settings->Background.y, settings->Background.z, settings->Background.w };
            if (ImGui::ColorEdit4("Background", col, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel))
                settings->Background = ImVec4(col[0], col[1], col[2], col[3]);
        }
        {
            float col[4] = { settings->GridColor.x, settings->GridColor.y, settings->GridColor.z, settings->GridColor.w };
            if (ImGui::ColorEdit4("Grid", col, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel))
                settings->GridColor = ImVec4(col[0], col[1], col[2], col[3]);
        }

        ImGui::Checkbox("Minimap",       &settings->ShowMinimap);
        ImGui::Checkbox("Live Load",     &settings->LiveLoad);
        ImGui::Checkbox("Plot Clipping", &settings->ShowErrors);

        if (settings->ShowErrors)
            ImGui::SliderFloat("Error Alpha", &settings->ErrorAlpha, 0.0f, 1.0f);
    }
    endWidget();
}
Plotter::PlotControls_Widget::~PlotControls_Widget() { }
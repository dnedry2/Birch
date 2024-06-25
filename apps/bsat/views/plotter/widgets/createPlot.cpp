#include "plotter.hpp"

Plotter::CreatePlot_Widget::CreatePlot_Widget(int id, Tool** tool, std::vector<Plot*>* plots, Session* session, PlotSettings* settings, PlotSyncSettings* syncSettings)
: Widget(id, 0, std::string("")), tool(tool), plots(plots), session(session), settings(settings), syncSettings(syncSettings)
{ }

void Plotter::CreatePlot_Widget::Render()
{
    ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_FrameBg));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4(ImGuiCol_FrameBg));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_FrameBg));
    ImGui::Button("Drop fields here", ImVec2(-1, 30));

    ImGui::PopStyleColor(3);

    if (ImGui::BeginDragDropTarget())
    {
        if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("DND_PLOT"))
        {
            Plot* plot = new Plot(settings, syncSettings);

            plot->AddField(*((PlotField **)payload->Data));
            plots->push_back(plot);
        }

        ImGui::EndDragDropTarget();
    }
}

void Plotter::CreatePlot_Widget::RenderMini(int index)
{
    ImGui::SetCursorPos(ImGui::GetCursorPos() - ImVec2(0, 7));

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0, 0, 0, 0));
    ImGui::Button("##dummy", ImVec2(-1, 5));

    ImGui::PopStyleColor(3);

    ImGui::SetCursorPos(ImGui::GetCursorPos() - ImVec2(0, 7));

    if (ImGui::BeginDragDropTarget())
    {
        if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("DND_PLOT"))
        {
            Plot* plot = new Plot(settings, syncSettings);

            plot->AddField(*((PlotField **)payload->Data));
            plots->insert(plots->begin() + index, plot);
        }

        ImGui::EndDragDropTarget();
    }
}
Plotter::CreatePlot_Widget::~CreatePlot_Widget() { }
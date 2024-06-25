#include "plotter.hpp"

#include "include/birch.h"

using namespace Birch;

double Plotter::MeasurementDisplay_Widget::unitToMult(Unit s)
{
    switch (s)
    {
    case Unit::s:
        snprintf(dispUnitStr, 6, " (s)");
        return 1;
    case Unit::ms:
        snprintf(dispUnitStr, 6, " (ms)");
        return 1000;
    case Unit::us:
        snprintf(dispUnitStr, 7, " (µs)");
        return 1000000;
    }
    return 0;
}
void Plotter::MeasurementDisplay_Widget::Render()
{
    if (beginWidget(this)) {
    /*
        if (measCount != measurements->size()) {
            stable_sort(measurements->begin(), measurements->end(), 
            [](const Measurement*& a, const Measurement*& b) -> bool
            { 
                return a->Group > b->Group;
            });

            measCount = measurements->size();
        }
    */

        ImGui::SetNextItemWidth(ImGui::GetWindowSize().x / 2 - 10);
        ImGui::Combo("##mode", (int *)&mode, "Screen\0Search\0Smart\0");
        ImGui::SameLine();

        ImGui::SetNextItemWidth(ImGui::GetWindowSize().x / 2 - 9);
        ImGui::Combo("##units", (int *)&dispUnit, "sec\0ms\0µs\0");

        const PlotField* lastField = nullptr;

        if (ImGui::BeginChildFrame(39483330, ImVec2(0, ImGui::GetWindowSize().y - 116), 0)) {
            const uint mCount = measurements->size();
            PlotField* lastField = nullptr;

            for (uint i = 0; i < mCount; i++) {
                if ((*measurements)[i]->Field() != lastField) {
                    lastField = (*measurements)[i]->Field();

                    uint lastEl = i;
                    for (; lastEl < mCount; lastEl++)
                        if ((*measurements)[lastEl]->Field() != lastField)
                            break;

                    //printf("%u\n", lastEl - i);

                    ImGui::PushID(39483331 + i);
                    if (ImGui::CollapsingHeader(lastField->Name)) {
                        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 8);

                        if (ImGui::BeginChildFrame(39483330 + i, ImVec2(0, ((lastEl - i) + 3) * 20 + 10), 0)) {
                            if (ImGui::BeginTable("MeasureTable", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable))
                            {
                                double unitMult = unitToMult(dispUnit);

                                char xHead[11];
                                char dxHead[11];
                                snprintf(xHead, 11, "X%s", dispUnitStr);
                                ImGui::TableSetupColumn(xHead);
                                snprintf(dxHead, 11, "ΔX%s", dispUnitStr);
                                ImGui::TableSetupColumn(dxHead);
                                ImGui::TableSetupColumn("Y");
                                ImGui::TableSetupColumn("ΔY");
                                ImGui::TableHeadersRow();

                                Point<double> lastPos = *(*measurements)[i]->Position();

                                //auto pos = (*measurements)[i]

                                //lastPos[0] = pos->X;
                                //lastPos[1] = pos->Y;

                                double avg[] = { 0, 0, 0, 0 };
                                uint runLen = 0;

                                for (; i < mCount; i++) {
                                    if ((*measurements)[i]->Field() != lastField) {
                                        i--;
                                        break;
                                    }

                                    Measurement* measurement = (*measurements)[i];

                                    ImGui::TableNextRow();

                                    ImGui::TableNextColumn();
                                    ImGui::Text("%.3lf", measurement->Position()->X * unitMult);
                                    avg[0] += measurement->Position()->X * unitMult;

                                    ImGui::TableNextColumn();
                                    ImGui::Text("%.3lf", (measurement->Position()->X - lastPos.X) * unitMult);
                                    avg[1] += (measurement->Position()->X - lastPos.X) * unitMult;

                                    ImGui::TableNextColumn();
                                    ImGui::Text("%.3lf", measurement->Position()->Y);
                                    avg[2] += measurement->Position()->Y;

                                    ImGui::TableNextColumn();
                                    ImGui::Text("%.3lf", measurement->Position()->Y - lastPos.Y);
                                    avg[3] += measurement->Position()->Y - lastPos.Y;

                                    lastPos = *measurement->Position();

                                    //lastPos[0] = measurement->Position()->X;
                                    //lastPos[1] = measurement->Position()->Y;

                                    runLen++;
                                }

                                if (runLen != 0)
                                {
                                    ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
                                    
                                    ImGui::TableNextColumn();
                                    ImGui::Text("Mean");

                                    ImGui::TableNextRow(ImGuiTableRowFlags_Headers);

                                    avg[0] /= runLen;

                                    ImGui::TableNextColumn();

                                    char avgStr[16];
                                    snprintf(avgStr, 16, "%.3lf", avg[0]);
                                    if (ImGui::Selectable(avgStr)) {

                                    }
                                    
                                    //ImGui::Text("%.3lf", avg[0]);

                                    avg[1] /= runLen - 1;

                                    ImGui::TableNextColumn();

                                    snprintf(avgStr, 16, "%.3lf", avg[1]);
                                    if (ImGui::Selectable(avgStr)) {
                                        
                                    }

                                    //ImGui::Text("%.3lf", avg[1]);

                                    avg[2] /= runLen;

                                    ImGui::TableNextColumn();
                                    snprintf(avgStr, 16, "%.3lf", avg[2]);
                                    if (ImGui::Selectable(avgStr)) {

                                    }
                                    
                                    //ImGui::Text("%.3lf", avg[2]);

                                    avg[3] /= runLen - 1;

                                    ImGui::TableNextColumn();
                                    snprintf(avgStr, 16, "%.3lf", avg[3]);
                                    if (ImGui::Selectable(avgStr)) {

                                    }     
                                    //ImGui::Text("%.3lf", avg[3]);

    /*
                                    ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
                                    ImGui::TableNextColumn();

                                    if (ImGui::Button("Raster")) {

                                    }

                                    ImGui::TableNextColumn();

                                    if (ImGui::Button("Histogram")) {
                                        
                                    }

                                    ImGui::TableNextColumn();

                                    if (ImGui::Button("Raster")) {
                                        
                                    }

                                    ImGui::TableNextColumn();

                                    if (ImGui::Button("Histogram")) {
                                        
                                    }   
                                    */             
                                }

                                ImGui::EndTable();
                            }
                        }

                        ImGui::EndChildFrame();
                    } else {
                        for (; i < mCount; i++)
                            if ((*measurements)[i]->Field() != lastField)
                                break;
                        
                        i--;
                    }
                    ImGui::PopID();
                }
            }
    /*
            if (false && ImGui::CollapsingHeader("Table")) {
                if (ImGui::BeginTable("MeasureTable", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable))
                {
                    double unitMult = unitToMult(dispUnit);

                    char xHead[10];
                    char dxHead[10];
                    snprintf(xHead, 10, "X%s", dispUnitStr);
                    ImGui::TableSetupColumn(xHead);
                    snprintf(dxHead, 10, "dX%s", dispUnitStr);
                    ImGui::TableSetupColumn(dxHead);
                    ImGui::TableSetupColumn("Y");
                    ImGui::TableSetupColumn("dY");
                    ImGui::TableHeadersRow();

                    double lastPos[] = {0, 0};

                    if (measurements->size() != 0)
                    {
                        auto pos = measurements->front()->Position();

                        lastPos[0] = pos->X;
                        lastPos[1] = pos->Y;
                    }

                    double avg[] = {0, 0, 0, 0};

                    for (auto measurement : *measurements)
                    {
                        ImGui::TableNextRow();

                        ImGui::TableNextColumn();
                        ImGui::Text("%.3lf", measurement->Position()->X * unitMult);
                        avg[0] += measurement->Position()->X * unitMult;

                        ImGui::TableNextColumn();
                        ImGui::Text("%.3lf", (measurement->Position()->X - lastPos[0]) * unitMult);
                        avg[1] += (measurement->Position()->X - lastPos[0]) * unitMult;

                        ImGui::TableNextColumn();
                        ImGui::Text("%.3lf", measurement->Position()->Y);
                        avg[2] += measurement->Position()->Y;

                        ImGui::TableNextColumn();
                        ImGui::Text("%.3lf", measurement->Position()->Y - lastPos[1]);
                        avg[3] += measurement->Position()->Y - lastPos[1];

                        lastPos[0] = measurement->Position()->X;
                        lastPos[1] = measurement->Position()->Y;
                    }

                    ImGui::TableNextRow(ImGuiTableRowFlags_Headers);

                    if (measurements->size() != 0)
                    {
                        ImGui::TableNextColumn();
                        ImGui::Text("Mean");

                        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);

                        avg[0] /= measurements->size();

                        ImGui::TableNextColumn();
                        ImGui::Text("%.3lf", avg[0]);

                        avg[1] /= measurements->size() - 1;

                        ImGui::TableNextColumn();
                        ImGui::Text("%.3lf", avg[1]);

                        avg[2] /= measurements->size();

                        ImGui::TableNextColumn();
                        ImGui::Text("%.3lf", avg[2]);

                        avg[3] /= measurements->size() - 1;

                        ImGui::TableNextColumn();
                        ImGui::Text("%.3lf", avg[3]);
                    }

                    ImGui::EndTable();
                }
            }
        */
        }
        ImGui::EndChildFrame();

        if (ImGui::Button("Clear"))
            measurements->clear();
        
        ImGui::SameLine();
        ImGui::Combo("##dispMode", (int *)&dispMode, "None\0X\0Y\0XY\0ΔX\0ΔY\0ΔXY\0");

        ImGui::SameLine();
        ImGui::ColorEdit4("Color", selectedColor, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
    }
    endWidget();
}
Plotter::MeasurementDisplay_Widget::MeasurementDisplay_Widget(GLuint icon, std::vector<Measurement *> *measurements, int id) : Widget(id, icon, std::string("Measure"))
{
    this->measurements = measurements;
    this->ToolID = "Measure";
}
Plotter::MeasurementDisplay_Widget::~MeasurementDisplay_Widget() { }
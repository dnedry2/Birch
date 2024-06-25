#include "plotter.hpp"
#include "logger.hpp"

#include <string>

using namespace Birch;

static std::string panToolHelp =
    "Left click to take a measurement.\n"
    "Hold shift for multi-cursor mode.\n"
    "Double click a measurement to see more information.";

static inline bool compDouble(double a, double b, double e) {
    return fabs(a - b) < e;
}

// Finds the nearest value in a sorted array
static inline uint findNearest(const double* input, uint len, double needle) {
    auto it = std::lower_bound(input, input + len, needle);
    return std::distance(input, std::min(it, input + len - 1));
}


Plotter::MeasureTool::MeasureTool(ImPlotLimits* limits, Session* session, MeasurementDisplay_Widget* widget, Plotter* parent)
                     : Tool("Measure", panToolHelp.c_str(), true, textures["bicons/IconMeasure.svg"], textures["bicons/CursorMeasure.svg"], textures["bicons/CursorMeasure.svg"], ImVec2(7, 16), ImVec2(7, 16), ImGuiKey_S),
                       limits(limits), session(session), widget(widget), parent(parent) { }

void Plotter::MeasureTool::on_selection(ImPlotInputMap *controls, ImPlotFlags *flags)
{
    controls->VerticalMod = ImGuiModFlags_Shift;
    controls->HorizontalMod = ImGuiModFlags_Shift;
    controls->BoxSelectMod = ImGuiModFlags_Alt;

    controls->QueryButton = ImGuiMouseButton_Left;
    controls->QueryMod = ImGuiModFlags_None;

    controls->BoxSelectButton = ImGuiMouseButton_Right;

    controls->PanButton = ImGuiMouseButton_Middle;

    *flags = ImPlotFlags_Query | ImPlotFlags_Crosshairs;
    pflags = flags;
}

void Plotter::MeasureTool::on_update(Plot* plot) {
    const static uint pip_tex             = textures["bicons/DecPip.svg"];
    const static uint measurePreview_tex  = textures["bicons/DecPipShadow.svg"];
    const static uint measurePreview2_tex = textures["bicons/DecPipShadowLight.svg"];
    const static uint measureCursor_tex   = textures["bicons/CursorMeasure.svg"];
    const static uint cursor_tex          = textures["bicons/IconCursor.svg"];
    const static uint pipEmpty_tex        = textures["bicons/DecPipEmpty.svg"];

    static auto* imGuiIO = &ImGui::GetIO();

    MeasurementDisplay_Widget::MeasureMode measureMode = *widget->Get_Mode();

    ImPlotPoint mousePos    = ImPlot::GetPlotMousePos();
    ImPlotPoint mousePosPix = ImPlot::PlotToPixels(mousePos, 0);

    const auto field = plot->SelectedField();

    const double *y      = field->Data;
    const double *x      = field->Timing;
    const bool   *mask   = field->FilterMask;
    const auto    count  = *field->ElementCount;

    if (ImPlot::IsPlotHovered())
        mouseHoverPip = false;

    Measurement* selMeas = nullptr;

    // Plot measurements
    Measurement* lastMeas = nullptr;
    auto measurements = session->Measurements();

    // Don't plot measurements if there are too many. TODO: Add a limit when selecting
    if (measurements->size() > 512)
        goto skipMeasurementPlotting;

    for (const auto& measurement : *measurements)
    {
        ImGui::PushID(&measurement);
        if (plot->HasField(measurement->Field()))
        {
            if (measurement->Position()->X >= limits->X.Min && measurement->Position()->X <= limits->X.Max)
            {
                if (limits->X.Max - limits->X.Min < (measurement->Bounds()->End - measurement->Bounds()->Start) * 4)
                {
                    auto ppp = ImPlotPoint(measurement->Position()->X, measurement->Position()->Y);
                    ImPlot::PlotImage("##measure", t_ImTexID((*measurement->Screen() || *measurement->DataIdx() == -1? pipEmpty_tex : pip_tex)), ppp, ImVec2(16, 16), ImVec2(-8, -16));

                    //bool DragPoint(const char* id, double* x, double* y, bool show_label, const ImVec4& col, float radius) {
                    if (ImPlot::DragPoint("##pip", &measurement->Position()->X, &measurement->Position()->Y, false, ImVec4(0, 1, 1, 0), 16)) {
                        *measurement->Screen() = true;

                        if (lastMeas != nullptr && lastMeas->Field() == measurement->Field() && lastMeas->Position()->X > measurement->Position()->X)
                            needSort = true;

                        // Snap if close to found point
                        if (measureMode == MeasurementDisplay_Widget::MeasureMode::Smart) {
                            // Find preview point
                            double tol   = 200;
                            uint   pIdx  = SmartSearchBox(x, y, mask, count, mousePos, tol);
                            auto   pPos  = ImPlotPoint(x[pIdx], y[pIdx]);
                            auto   pDist = sqrt(pow(pPos.x - mousePos.x, 2) + pow(pPos.y - mousePos.y, 2));

                            if (pDist < tol)
                            {
                                ImPlot::PlotImage("##measure", t_ImTexID(measurePreview_tex), pPos, ImVec2(16, 16), ImVec2(-7, -16));

                                needSnap  = true;
                                snapPoint = Point<double>(pPos.x, pPos.y);
                                snapMeas  = measurement;
                            } else {
                                needSnap = false;
                            }
                        }
                    } else {
                        if (needSnap && !imGuiIO->MouseDown[0] && measurement == snapMeas) {
                            needSnap  = false;
                        
                            *measurement->Position() = snapPoint;
                            *measurement->Screen()   = false;

                            if (lastMeas != nullptr && lastMeas->Field() == measurement->Field() && lastMeas->Position()->X > measurement->Position()->X) {
                                needSort = true;
                                sortMeas = measurement;
                            }
                        }

                        if (needSort && measurement == sortMeas) {
                            session->SortMeasurements();
                            needSort = false;
                        }
                    }

                    

                    ppp.x -= 8;
                    ppp.y -= 16;
                    auto pixpos = ImPlot::PlotToPixels(ppp, 0);

                    if (!mouseHoverPip) {
                        mouseHoverPip = mousePosPix.x >= pixpos.x - 8 && mousePosPix.x <= pixpos.x + 24 && mousePosPix.y >= pixpos.y - 24 && mousePosPix.y <= pixpos.y + 8;
                        selMeas = measurement;
                    }

                    const auto& disp  = *widget->Get_DispMode();
                    const bool  delta = (disp == MeasureDisplay::DX || disp == MeasureDisplay::DY || disp == MeasureDisplay::DXY);
                    const bool  skip  = delta && (lastMeas == nullptr || lastMeas->Field() != measurement->Field());

                    // Plot measurement annotation
                    if (disp != MeasureDisplay::None && !skip) {
                        ImPlot::PushPlotClipRect();

                        char txtBuf[128];
                        if (measurement->AltText()->size() == 0) {
                            switch (disp) {
                                case MeasureDisplay::X:
                                    snprintf(txtBuf, 128, "X: %.2lf", measurement->Position()->X * widget->XMult());
                                    break;
                                case MeasureDisplay::Y:
                                    snprintf(txtBuf, 128, "Y: %.2lf", measurement->Position()->Y);
                                    break;
                                case MeasureDisplay::XY:
                                    snprintf(txtBuf, 128, "X: %.2lf\nY: %.2lf", measurement->Position()->X * widget->XMult(), measurement->Position()->Y);
                                    break;
                                case MeasureDisplay::DX:
                                    snprintf(txtBuf, 128, "ΔX: %.2lf", (measurement->Position()->X - lastMeas->Position()->X) * widget->XMult());
                                    break;
                                case MeasureDisplay::DY:
                                    snprintf(txtBuf, 128, "ΔY: %.2lf", measurement->Position()->Y - lastMeas->Position()->Y);
                                    break;
                                case MeasureDisplay::DXY:
                                    snprintf(txtBuf, 128, "ΔX: %.2lf\nΔY: %.2lf", (measurement->Position()->X - lastMeas->Position()->X) * widget->XMult(), measurement->Position()->Y - lastMeas->Position()->Y);
                                    break;
                                default:
                                    break;
                            }
                        } else {
                            snprintf(txtBuf, 128, "%s", measurement->AltText()->c_str());
                        }

                        const ImVec2 txt_size = ImGui::CalcTextSize(txtBuf);
                        const ImVec2 size = txt_size + ImPlot::GetStyle().AnnotationPadding * 2 + ImVec2(2, 0);

                        ImVec2 pos;
                        ImVec2 posPlot;

                        if (delta) {
                            posPlot = ImVec2((lastMeas->Position()->X + (measurement->Position()->X - lastMeas->Position()->X) * 0.5) + *measurement->XOffset(),
                                             (lastMeas->Position()->Y + (measurement->Position()->Y - lastMeas->Position()->Y) * 0.5) + *measurement->YOffset());
                            
                            pos = ImPlot::PlotToPixels(posPlot);
                            pos.x -= txt_size.x / 2;
                            pos.y -= txt_size.y / 2 + 2;
                        } else {
                            pos = ImPlot::PlotToPixels(ImVec2(measurement->Position()->X, measurement->Position()->Y));
                            pos.x -= txt_size.x / 2;
                            pos.y -= txt_size.y + 20;
                        }

                        ImDrawList* drawList = ImGui::GetWindowDrawList();

                        // Render box
                        ImRect rect(pos, pos + size);

                        // Adjust y offset
                        if (delta) {
                            // Check if cursor is inside box
                            if (mousePosPix.x >= rect.Min.x && mousePosPix.x <= rect.Max.x && mousePosPix.y >= rect.Min.y && mousePosPix.y <= rect.Max.y) {
                                double xDummy = posPlot.x;
                                double yDummy = posPlot.y;

                                if (ImGui::IsMouseDoubleClicked(0)) {
                                    *measurement->AltText() = std::string(txtBuf);
                                    ImGui::OpenPopup("##textInput");
                                } else if (ImPlot::DragPoint("##dragBox", &xDummy, &yDummy, false, ImVec4(0, 0, 0, 0), rect.GetWidth() * 0.5)) {
                                    *measurement->XOffset() += xDummy - posPlot.x;
                                    *measurement->YOffset() += yDummy - posPlot.y;
                                }
                            }
                        }

                        // Alt text input
                        bool textChanged = false;
                        ImGui::SetNextWindowSize(ImVec2(200, 42));
                        ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().MousePos.x, ImGui::GetIO().MousePos.y), ImGuiCond_Appearing);
                        if (ImGui::BeginPopup("##textInput")) {
                            ImGui::SetNextItemWidth(184);
                            if (ImGui::InputTextMultiline("##textInput", txtBuf, 128, ImVec2(0, 0))) {
                                *measurement->AltText() = std::string(txtBuf);
                                textChanged = true;
                            }
                            if (ImGui::IsItemDeactivatedAfterEdit()) {
                                ImGui::CloseCurrentPopup();
                            }
                            ImGui::EndPopup();
                        }

                        if (textChanged) {
                            ImPlot::PopPlotClipRect();
                            ImGui::PopID();
                            continue;
                        }

                        // Set alpha to full opacity
                        uint foreColor = widget->Get_Color() & 0x00ffffff | 0xff000000;

                        uint backColor = widget->Get_Color();
                        backColor = (backColor & 0xFF000000) | ((backColor >> 1) & 0x00404040);


                        if (!*measurement->Screen() && !delta) {
                            // Find pdw for color / info view
                            auto x = measurement->Field()->Timing;
                            auto y = measurement->Field()->Data;
                            auto col = measurement->Field()->Colormap;
                            if (measurement->Position()->X != x[*measurement->DataIdx()] || measurement->Position()->Y != y[*measurement->DataIdx()]) {
                                auto mx = measurement->Position()->X;
                                auto my = measurement->Position()->Y;
                                
                                *measurement->DataIdx() = -1;

                                for (int i = 0; i < *measurement->Field()->ElementCount; i++) {
                                    if (compDouble(mx, x[i], 0.000001) && compDouble(my, y[i], 0.000001)) {
                                        *measurement->DataIdx() = i;
                                        break;
                                    }
                                }
                            }

                            if (*measurement->DataIdx() != -1) {
                                foreColor = col[*measurement->DataIdx()];
                            }
                        }

                        //float cn = .5;
                        //uint alphaMask=0xff000000;
                        //backColor = (foreColor|alphaMask) & ((uint)(alphaMask*cn) | (~alphaMask));


                        drawList->AddRectFilled(rect.Min, rect.Max, backColor, 5);
                        drawList->AddRect(rect.Min, rect.Max, foreColor, 5);

                        drawList->AddText(pos + ImVec2(4, 2), 0xFFFFFFFF, txtBuf);

                        // Render lines
                        if (delta) {
                            auto pos1 = ImPlot::PlotToPixels(ImVec2(lastMeas->Position()->X    + *measurement->XOffset(), lastMeas->Position()->Y    + *measurement->YOffset()));
                            auto pos2 = ImPlot::PlotToPixels(ImVec2(measurement->Position()->X + *measurement->XOffset(), measurement->Position()->Y + *measurement->YOffset()));

                            // Render vertical lines for y offset
                            if (*measurement->YOffset() != 0) {
                                auto pos3 = ImPlot::PlotToPixels(ImVec2(lastMeas->Position()->X,    lastMeas->Position()->Y));
                                auto pos4 = ImPlot::PlotToPixels(ImVec2(measurement->Position()->X, measurement->Position()->Y));

                                // Don't allow lines to overlap pip
                                // TODO: makes measurement look bigger than it is
                                /*
                                if (pos1.y < pos3.y) {
                                    pos3.y -= 24;
                                } else {
                                    pos3.y += 8;
                                }

                                if (pos2.y < pos4.y) {
                                    pos4.y -= 24;
                                } else {
                                    pos4.y += 8;
                                }
                                */

                                drawList->AddLine(pos1, pos3, foreColor);
                                drawList->AddLine(pos2, pos4, foreColor);
                            } else {
                                // Add some space if y offset is not used
                                pos1.x += 10;
                                pos2.x -= 10;
                            }

                            float slope = (pos2.y - pos1.y) / (pos2.x - pos1.x);
                            float intercept = pos1.y - slope * pos1.x;

                            drawList->AddLine(pos1, ImVec2(rect.GetTL().x - 10, slope * (rect.GetTL().x - 10) + intercept), foreColor);
                            drawList->AddLine(ImVec2(rect.GetTR().x + 10, slope * (rect.GetTR().x + 10) + intercept), pos2, foreColor);
                        }

                        ImPlot::PopPlotClipRect();
                    }
                
                }
            }
        }
        ImGui::PopID();

        lastMeas = measurement;
    }

    skipMeasurementPlotting:

    if (mouseHoverPip) {
        cursorNormal = cursor_tex;
        *pflags = ImPlotFlags_None;
    } else {
        cursorNormal = measureCursor_tex;
        *pflags = ImPlotFlags_Crosshairs;
    }

    static ImPlotPoint previewPos;
    static ImPlotPoint lastPreviewPos;
    static int previewIdx;
    static bool placed = true;

    bool findAll = measureMode == MeasurementDisplay_Widget::MeasureMode::Smart && (ImGui::IsKeyDown(ImGuiKey_LeftShift) || ImGui::IsKeyDown(ImGuiKey_RightShift)) && session->Measurements()->size() > 0;
    
    std::vector<int> allPoints = std::vector<int>();

    if (ImPlot::IsPlotHovered() && imGuiIO->MouseClicked[0])
        mouseStart = Point<double>(mousePos.x, mousePos.y);


    const auto fieldType = ((BFile*)plot->SelectedField()->File)->Type();

    // Determine new preview position

    // Disable previews if mouse is moving (for box select)
    bool boxSelect = false;
    if (ImPlot::IsPlotHovered() && imGuiIO->MouseDown[0] && (mouseStart.X != mousePos.x || mouseStart.Y != mousePos.y) && fieldType == FileType::TBD) {
        boxSelect = true;
        boxSelComplete = false;
    }

    // Calculate preview position
    if (ImPlot::IsPlotHovered() && !mouseHoverPip && !boxSelect)
    {
        const bool spectra = !strncmp("Spectra", plot->SelectedField()->Name, 7);

        switch (measureMode)
        {
        case MeasurementDisplay_Widget::MeasureMode::Screen:
            previewPos = mousePos;
            previewIdx = -1;
            break;
        case MeasurementDisplay_Widget::MeasureMode::Search:
            previewIdx = findNearest(x, count, mousePos.x);
            previewPos = ImPlotPoint(x[previewIdx], y[previewIdx]);
            break;
        case MeasurementDisplay_Widget::MeasureMode::Smart:
            if (spectra && (lastMousePos.x != mousePos.x || lastMousePos.y != mousePos.y)) {
                auto spPos = SmartSearchSpectra(field->Data, field->XRes, field->YRes, Birch::Point<double>(mousePos.x, mousePos.y), field->XLimits, field->YLimits, field->MaxValue);
                previewPos = ImPlotPoint(spPos.X, spPos.Y);
            } else 

            if (lastMousePos.x != mousePos.x || lastMousePos.y != mousePos.y || findAll) {
                auto plotLimits = plot->Limits();
                
                previewIdx = SmartSearch(x, y, mask, count, Birch::Span<double>(plotLimits.Y.Min, plotLimits.Y.Max), mousePos);
                previewPos = ImPlotPoint(x[previewIdx], y[previewIdx]);

                // Multicursor mode
                if (findAll) {
                    // Position to be iterated foward each time a new point is found
                    auto curMousePos = mousePos;


                    // Find last measurement in for this field;
                    Measurement* lastFieldMeas = nullptr;
                    const uint mCount = measurements->size();
                    const auto selField = plot->SelectedField();

                    for (uint i = mCount - 1; i >= 0; i--) {
                        if (measurements->at(i)->Field() == selField) {
                            lastFieldMeas = measurements->at(i);
                            break;
                        }
                    }

                    // Break if there are no measurements in this field
                    if (lastFieldMeas != nullptr) {
                        // Calculate the first window - will be updated as new points are found
                        double window = x[previewIdx] - lastFieldMeas->Position()->X;
                        double lastX  = x[previewIdx];

                        // Iterate until the position moves out of the plot
                        while ((curMousePos.x += window) < plotLimits.X.Max && window > 0) {
                            // Find the next point
                            int  nextIdx = SmartSearch(x, y, mask, count, Span<double>(plotLimits.Y.Min, plotLimits.Y.Max), curMousePos);
                            auto nextPos = ImPlotPoint(x[nextIdx], y[nextIdx]);

                            // Plot cursor and preview
                            ImPlot::PlotImage("##measure", t_ImTexID(measureCursor_tex),   curMousePos, ImVec2(16, 16), ImVec2(-7, -16));
                            ImPlot::PlotImage("##measure", t_ImTexID(measurePreview2_tex), nextPos,     ImVec2(16, 16), ImVec2(-7, -16));

                            allPoints.push_back(nextIdx);

                            // Update the window
                            window = x[nextIdx] - lastX;
                            lastX  = x[nextIdx];
                        }
                    }
                }
            }
            
            lastMousePos = mousePos;

            break;
        }

        ImPlot::PlotImage("##measure", t_ImTexID(measurePreview_tex), previewPos, ImVec2(16, 16), ImVec2(-7, -16));
    }

    if (lastPreviewPos.x != previewPos.x || lastPreviewPos.y != previewPos.y)
        placed = false;

    // Draw box selection
    if (boxSelect) {
        ImPlot::PushPlotClipRect();

        auto dl = ImPlot::GetPlotDrawList();

        ImPlotPoint boxSelStart = ImPlotPoint(mouseStart.X, mouseStart.Y);
        ImPlotPoint boxSelEnd   = mousePos;

        if (boxSelStart.x > boxSelEnd.x)
            std::swap(boxSelStart.x, boxSelEnd.x);
        if (boxSelStart.y < boxSelEnd.y)
            std::swap(boxSelStart.y, boxSelEnd.y);

        dl->AddRectFilled(ImPlot::PlotToPixels(boxSelStart), ImPlot::PlotToPixels(boxSelEnd), IM_COL32(255, 255, 255, 24), 5);
        dl->AddRect(ImPlot::PlotToPixels(boxSelStart),       ImPlot::PlotToPixels(boxSelEnd), IM_COL32(255, 255, 255, 255), 5);

        ImPlot::PopPlotClipRect();
    }

    // Add new measurement
    if (imGuiIO->MouseReleased[0] && ImPlot::IsPlotHovered() && !placed) {
        if (!boxSelect && boxSelComplete) {
            if (!mouseHoverPip)
            {
                placed = true;

                Point<double> mPos    = Point<double>(previewPos.x, previewPos.y);
                Span<double>  mBounds = Span<double>(limits->X.Min, limits->X.Max);

                session->AddMeasurement(new Measurement(mPos, mBounds, plot->SelectedField(), measureMode == MeasurementDisplay_Widget::MeasureMode::Screen, measureMode == MeasurementDisplay_Widget::MeasureMode::Screen ? -1 : previewIdx));

                if (findAll) {
                    for (auto idx : allPoints) {
                        session->AddMeasurement(new Measurement(Point<double>(x[idx], y[idx]), mBounds, plot->SelectedField(), measureMode == MeasurementDisplay_Widget::MeasureMode::Screen, measureMode == MeasurementDisplay_Widget::MeasureMode::Screen ? -1 : previewIdx));
                    }
                }
            }
        } else if (!boxSelComplete) {
            boxSelComplete = true;
            placed = true;

            if (fieldType == FileType::Signal) {
                DispWarning("Plotter::MeasureTool::on_update", "Box selection is (currently) only supported for time based data!");
            } else {
                // Determine the bounds of the box selection
                Span<double> xBounds = Span<double>(mouseStart.X, mousePos.x);
                Span<double> yBounds = Span<double>(mouseStart.Y, mousePos.y);

                if (xBounds.Start > xBounds.End)
                    std::swap(xBounds.Start, xBounds.End);
                if (yBounds.Start > yBounds.End)
                    std::swap(yBounds.Start, yBounds.End);
                
                Span<uint> idxBounds = Span<uint>(findNearest(x, count, xBounds.Start), findNearest(x, count, xBounds.End));

                // Find all points within the bounds
                const auto mBounds = Span<double>(limits->X.Min, limits->X.Max);
                const auto selField = plot->SelectedField();
                auto mms = std::vector<Measurement*>();

                #pragma omp parallel for
                for (uint i = idxBounds.Start; i <= idxBounds.End; i++)
                    if (y[i] >= yBounds.Start && y[i] <= yBounds.End && mask[i])
                        #pragma omp critical
                        mms.push_back(new Measurement(Point<double>(x[i], y[i]), mBounds, selField, false, i));
                
                // Add all measurements
                session->AddMeasurements(mms);
            }
        }
    }

    // Show measurement details
    if (imGuiIO->MouseDoubleClicked[0] && mouseHoverPip && selMeas != nullptr && *selMeas->DataIdx() != -1) {
        ImGui::OpenPopup("Measurement");

        std::string str = "Time:\t" + std::to_string(selMeas->Position()->X) + "\n\n";

        BFile* file = reinterpret_cast<BFile*>(selMeas->Field()->File);

        for (auto field : *file->Fields()) {
            str = str + std::string(field->Name) + std::string(":\t") + std::to_string(field->Data[*selMeas->DataIdx()]) + std::string("\n");
        }

        snprintf(pulseMeasureText, 4096, "%s", str.data());

        parent->cancelFit = true;
    }

    if (ImGui::BeginPopup("Measurement"))
    {
        ImGui::InputTextMultiline("##pmeasure", pulseMeasureText, 4096, ImVec2(0, 0), ImGuiInputTextFlags_ReadOnly);
        ImGui::EndPopup();
    }


    lastPreviewPos = previewPos;
}

void Plotter::MeasureTool::renderGraphics(Plot* plot) {
    
}
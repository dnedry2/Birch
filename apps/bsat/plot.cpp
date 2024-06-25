#include "plot.hpp"

#include <cstring>
#include <algorithm>

#include "ImPlot_Extras.h"

#include "svg.hpp"

#include "session.hpp"
#include "logger.hpp"

static bool limitsEqual(const ImPlotLimits& a, const ImPlotLimits& b) {
	return a.X.Min == b.X.Min && a.X.Max == b.X.Max && a.Y.Min == b.Y.Min && a.Y.Max == b.Y.Max;
}
static ImPlotAxisFlags sciNot(double yMax) {
    return yMax > 10000 ? ImPlotAxisFlags_SciNotation : ImPlotAxisFlags_None;
}
static void renderCursor(Plot* plot, Tool** currentTool)
{
    //const static GLuint pointerTex = textures["bicons/CursorCursor.svg"];
    //const static GLuint handTex    = textures["bicons/CursorHand.svg"];
    //const static GLuint fistTex    = textures["bicons/CursorFist.svg"];

    const static GLuint pointerTex = textures["bicons/CursorCursor.svg"];
    const static GLuint handTex    = textures["bicons/CursorHand.svg"];
    const static GLuint fistTex    = textures["bicons/CursorFist.svg"];

    if (ImGui::IsPopupOpen("FIR Designer", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel))
        return;

    auto& io = ImGui::GetIO();

    if (ImPlot::IsPlotHovered() && (!plot->IsLegendHovered() || plot->IsQueried())) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left))
            (*currentTool)->MouseClicked(plot);
        else if (ImGui::IsMouseReleased(ImGuiMouseButton_Left))
            (*currentTool)->MouseReleased(plot);
        else if (ImGui::IsMouseReleased(ImGuiMouseButton_Middle))
            (*currentTool)->MouseMiddleClicked(plot);

        (*currentTool)->RenderCursor(io.MousePos);
    } else if (plot->IsLegendHovered())
        ImGui::GetForegroundDrawList(ImGui::GetCurrentWindow())->AddImage(t_ImTexID(pointerTex), io.MousePos - ImVec2(3, 0), io.MousePos + ImVec2(16, 16) - ImVec2(3, 0), ImVec2(0, 0));
    else if (ImPlot::IsPlotXAxisHovered() || ImPlot::IsPlotYAxisHovered()) {
        ImGui::GetForegroundDrawList(ImGui::GetCurrentWindow())->AddImage(io.MouseDown[0] ? t_ImTexID(fistTex) : t_ImTexID(handTex), io.MousePos - ImVec2(8, 8), io.MousePos + ImVec2(16, 16) - ImVec2(8, 8), ImVec2(0, 0));
        ImGui::SetMouseCursor(ImGuiMouseCursor_None);
    }
}


PlotField::PlotField(char* name, char* catagory, volatile bool* fetching, double* data, double* time, unsigned long* elCount, PlotFieldRenderer* renderer, ImU32* volatile colormap, bool* volatile mask, void* file, ImU32* volatile shadedColormap)
{
    Name = name;
    Data = data;
    Timing = time;
    ElementCount = elCount;
    Catagory = catagory;
    Fetching = fetching;

    Renderer = renderer;

    Colormap = colormap;
    ShadedColormap = shadedColormap;
    FilterMask = mask;

    LoadStatus = Shit;

    File = file;
}
void PlotField::IncVersion() {
    // The overflow is intentional...
    // Version doesn't need to be unique, just different than it was
    DataVersion++;
}

PlotFieldContainer::PlotFieldContainer(PlotField* field) : Field(field) { }

PlotBadge::PlotBadge(const std::string& tooltip, GLuint texture)
          : tooltipText(tooltip), texture(texture) { }

void PlotBadge::Render(const ImPlotPoint& plotPos, const ImVec2& uv0, const ImVec2& size) {
    ImPlot::PlotImage("badge", t_ImTexID(texture), plotPos, size, uv0);

    // Check if badge region is hovered and render tooltip
    ImVec2& mousePos = ImGui::GetIO().MousePos;
    auto offset = ImPlot::PlotToPixels(plotPos) + uv0;
    ImRect badgeRect = ImRect(offset, offset + size);

    if (badgeRect.Contains(mousePos))
        tooltip.Render(tooltipText.c_str());
}

void Plot::AddField(PlotField* field) {
    // Check if field already exists in plot
    if (HasField(field)) {
        DispWarning("Plot::AddField", "Field %s already exists in plot", field->Name);
        return;
    }

    fields.push_back(PlotFieldContainer(field));
}
bool Plot::RemoveField(const std::string& name) {
    const char* nameC = name.c_str();
    uint nameLen = name.length();

    for (auto it = fields.begin(); it != fields.end(); ++it) {
        if (!strncmp(nameC, it->Field->Name, nameLen)) {
            fields.erase(it);
            return true;
        }
    }

    return false;
}
bool Plot::RemoveField(const PlotField* field) {
    for (auto it = fields.begin(); it != fields.end(); ++it) {
        if (it->Field == field) {
            fields.erase(it);
            return true;
        }
    }

    return false;
}

bool Plot::HasField(const std::string& name) const {
    const char* nameC = name.c_str();
    uint nameLen = name.length();

    for (auto it = fields.begin(); it != fields.end(); ++it)
        if (!strncmp(nameC, it->Field->Name, nameLen))
            return true;

    return false;
}
bool Plot::HasField(const PlotField* field) const {
    for (auto it = fields.begin(); it != fields.end(); ++it)
        if (it->Field == field)
            return true;

    return false;
}

void Plot::FitNext() {
    fitNext = true;
}

bool Plot::IsLogScale() const {
    return logScale;
}
bool Plot::IsClosed() const {
    return closed;
}
bool Plot::IsQueried() const {
    return queried;
}
bool Plot::IsAutoFitting() const {
    return autofit;
}
bool Plot::IsLegendHovered() const {
    return legendHov;
}

const ImRect& Plot::PlotRect() const {
    static auto nullRect = ImRect(0, 0, 0, 0);

    if (imPlotPlot == nullptr)
        return nullRect;
    
    return imPlotPlot->FrameRect;
}
const ImPlotLimits& Plot::Query() const {
    return query;
}
const ImPlotLimits& Plot::Limits() const {
    return limits;
}
const std::vector<PlotFieldContainer>& Plot::Fields() const {
    return fields;
}

PlotField* Plot::SelectedField() const {
    if (selectedField == nullptr)
        return nullptr;

    return selectedField->Field;
}
PlotFieldContainer* Plot::SelectedFieldContainer() const {
    return selectedField;
}

Plot::Plot(PlotSettings* settings, PlotSyncSettings* syncSettings)
     : settings(settings), syncSettings(syncSettings)
{
    badgeOverY  = new PlotBadge("Data hidden above y-axis limits", textures["bicons/BadgeDataOver.svg"]);
    badgeUnderY = new PlotBadge("Data hidden below y-axis limits", textures["bicons/BadgeDataUnder.svg"]);
    badgeGood   = new PlotBadge("Data fully loaded",               textures["bicons/BadgeGood.svg"]);
    badgeBad    = new PlotBadge("Data stale",                      textures["bicons/BadgeBad.svg"]);
    badgeWarn   = new PlotBadge("Data preview",                    textures["bicons/BadgeWarning.svg"]);
    badgeFetch  = new PlotBadge("Data is currently fetching",      textures["bicons/BadgeFetching.svg"]);
    badgeTune   = new PlotBadge("Data is baseband tuned",          textures["bicons/BadgeTune.svg"]);
    badgeFilter = new PlotBadge("Data is filtered",                textures["bicons/BadgeFilter.svg"]);
    badgeInvY   = new PlotBadge("Y-axis is inverted",              textures["bicons/BadgeInvertY.svg"]);
    badgeInvX   = new PlotBadge("X-axis is inverted",              textures["bicons/BadgeInvertX.svg"]);
    badgeLogY   = new PlotBadge("Y-axis is logarithmic",           textures["bicons/BadgeLog.svg"]);

    mmCtx = ImPlot::CreateContext();
}
Plot::~Plot() {
    delete badgeOverY;
    delete badgeUnderY;
    delete badgeGood;
    delete badgeBad;
    delete badgeWarn;
    delete badgeFetch;
    delete badgeTune;
    delete badgeFilter;
    delete badgeInvY;
    delete badgeInvX;
    delete badgeLogY;

    ImPlot::DestroyContext(mmCtx);
}


void Plot::renderPlotInfo()
{
    ImPlotLimits currentLimits = ImPlot::GetPlotLimits();

    int badgeCount = 0;

    ImPlotPoint plotPos = ImPlotPoint(invertedX ? currentLimits.X.Min : currentLimits.X.Max, invertedY ? currentLimits.Y.Min : currentLimits.Y.Max);

    if (overY) {
        badgeOverY->Render(plotPos, ImVec2(-26 - (20 * badgeCount), 10), ImVec2(16, 16));
        badgeCount++;
    }
    if (underY) {
        ImPlotPoint plotPosLow = ImPlotPoint(invertedX ? currentLimits.X.Min : currentLimits.X.Max, invertedY ? currentLimits.Y.Max : currentLimits.Y.Min);
        badgeUnderY->Render(plotPosLow, ImVec2(-26, -26), ImVec2(16, 16));
    }

    switch (status) {
        case 0: // full
            if (!hideGood) {
                badgeGood->Render(plotPos, ImVec2(-26 - (20 * badgeCount), 10), ImVec2(16, 16));
                hideGood = goodTimer.Now() > goodTime;
                badgeCount++;
            }
            break;
        case 1: // preview
            badgeWarn->Render(plotPos, ImVec2(-26 - (20 * badgeCount), 10), ImVec2(16, 16));
            goodTimer.Reset();
            hideGood = false;
            badgeCount++;
            break;
        case 2: // stale
            badgeBad->Render(plotPos, ImVec2(-26 - (20 * badgeCount), 10), ImVec2(16, 16));
            goodTimer.Reset();
            hideGood = false;
            badgeCount++;
            break;
    }

    if (lastFetching) {
        badgeFetch->Render(plotPos, ImVec2(-26 - (20 * badgeCount), 10), ImVec2(16, 16));
        badgeCount++;
    }

    if (tuned) {
        badgeTune->Render(plotPos, ImVec2(-26 - (20 * badgeCount), 10), ImVec2(16, 16));
        badgeCount++;
    }

    if (filtered) {
        badgeFilter->Render(plotPos, ImVec2(-26 - (20 * badgeCount), 10), ImVec2(16, 16));
        badgeCount++;
    }

    if (invertedY) {
        badgeInvY->Render(plotPos, ImVec2(-26 - (20 * badgeCount), 10), ImVec2(16, 16));
        badgeCount++;
    }

    if (invertedX) {
        badgeInvX->Render(plotPos, ImVec2(-26 - (20 * badgeCount), 10), ImVec2(16, 16));
        badgeCount++;
    }

    if (logY) {
        badgeLogY->Render(plotPos, ImVec2(-26 - (20 * badgeCount), 10), ImVec2(16, 16));
        badgeCount++;
    }
}

void Plot::Render(const ImVec2& size, Tool** currentTool, Session* session, void (*ctxMenu)(void* data), void* ctxMenuData) {
    // Close plot if all fields were removed
    if (fields.size() == 0) {
        closed = true;
        return;
    }

    closed = false;


    if (fitNext || (autofit && lastFetching))
        ImPlot::FitNextPlotAxes(false);
    
    // Disable keyboard controls if text input is active
    bool textInput = ImGui::GetIO().WantTextInput;

    // Reset status data
    status       = PlotField::Status::Full;
    invertedX    = false;
    invertedY    = false;
    overY        = false;
    underY       = false;
    lastFetching = false;
    tuned        = false;
    filtered     = false;
    logY         = false;

    // Merge field status data
    for (const auto& cont : fields) {
        const auto field = cont.Field;

        if (field == nullptr)
            continue;
        
        // Display custom ticks for fields which contain them
        if (field->Ticks > 0)
            ImPlot::SetNextPlotTicksY(field->TickVals, field->Ticks, field->TickText);
        
        if (field->LogScale) {
            imPlotPlot->YAxis[0].Flags |= ImPlotAxisFlags_LogScale;
            logScale = true;
        }

        tuned        |= field->Tuned;
        filtered     |= field->Filtered;
        lastFetching |= *field->Fetching;

        // If multiple fields, display the worst status only
        if (field->LoadStatus > status)
            status = field->LoadStatus;
    }

    singleMode = (*currentTool)->UseSinglePlot();

    // Setup ImPlot styling
    ImPlot::PushStyleColor(ImPlotCol_PlotBg,    settings->Background);
    ImPlot::PushStyleColor(ImPlotCol_XAxisGrid, settings->GridColor);
    ImPlot::PushStyleColor(ImPlotCol_YAxisGrid, settings->GridColor);

    ImPlot::GetStyle().Use24HourClock = true;
    ImPlot::GetStyle().UseISO8601     = true;

    ImPlotFlags plotFlags = ImPlotFlags_NoLegend | ImPlotFlags_NoTitle;
    ImPlotInputMap inputMap;

    bool disableCrosshairs = false;

    // Apply synchoronization settings if applicable
    if (syncSettings != nullptr) {
        if (!setNextY) {
            ImPlot::LinkNextPlotLimits(&syncSettings->limits->X.Min, &syncSettings->limits->X.Max, NULL, NULL);
        } else {
            ImPlot::LinkNextPlotLimits(&syncSettings->limits->X.Min, &syncSettings->limits->X.Max, &limits.Y.Min, &limits.Y.Max);
            setNextY = false;
        }
        
        plotFlags  = *syncSettings->flags;
        plotFlags |= ImPlotFlags_NoLegend | ImPlotFlags_NoTitle;

        inputMap = *syncSettings->controls;

        disableCrosshairs = !*syncSettings->anyHoveredLast;
    } else if (setNextY) {
        ImPlot::LinkNextPlotLimits(NULL, NULL, &limits.Y.Min, &limits.Y.Max);
        setNextY = false;
    }

    // Disable crosshairs if no plots are hovered
    // TODO: Plots without sync settings will display crosshairs erroneously
    if (disableCrosshairs)
        plotFlags &= ~ImPlotFlags_Crosshairs;


    // Render the plot
    ImGui::PushID(this);

    if (ImPlot::BeginPlot("BirchPlot", NULL, NULL, size, plotFlags, ImPlotAxisFlags_NoInitialFit, sciNot(maxY))) {
        imPlotPlot = ImPlot::GetCurrentPlot();

        if (syncSettings != nullptr)
            *syncSettings->anyHovered |= ImPlot::IsPlotHovered();

        ImPlot::GetInputMap() = inputMap;

        // Check for invert and log scale
        invertedX = ImHasFlag(imPlotPlot->XAxis.Flags,    ImPlotAxisFlags_Invert);
        invertedY = ImHasFlag(imPlotPlot->YAxis[0].Flags, ImPlotAxisFlags_Invert);
        logY      = ImHasFlag(imPlotPlot->YAxis[0].Flags, ImPlotAxisFlags_LogScale);


        // Determine if fields need to be rerasterized
        // This is the case if:
        //      - Plot size changed
        //      - Plot limits changed
        //      - Field data changed

        // Check if plot size changed
        const auto plotRect = PlotRect();
        stale |= plotRect.GetArea() != lastPlotRect.GetArea();
        // Check if plot limits changed
        const auto currentLimits = ImPlot::GetPlotLimits();
        const bool limitsChanged = !limitsEqual(lastLimits, currentLimits);
        stale |= limitsChanged;

        // Don't rerasterize if plot is being dragged
        if (limitsChanged)
            dragLockTimer.Reset();

        const bool dragging = imPlotPlot->XAxis.Dragging 
                           || imPlotPlot->YAxis[0].Dragging
                           || (ImPlot::IsPlotHovered() && ImGui::IsMouseDragging(ImGuiMouseButton_Left))
                           || dragLockTimer.Now() < dragLockTime;
		
        // Tell fields to rerasterize if necessary
        if (stale && !dragging) {
            for (auto& f : fields)
                f.Rasterized = false;

            stale = false;
        }
        // Check if field data changed and tell fields to rerasterize if necessary
        for (auto& f : fields) {
            if (f.LastVersion != f.Field->DataVersion) {
                f.Rasterized  = false;
                f.LastVersion = f.Field->DataVersion;
            }
        }

        // Determine whether to use scientific notation
        const auto yRange = imPlotPlot->YAxis[0].Range;
        maxY = std::max(std::abs(yRange.Max), std::abs(yRange.Min));

        // Single field mode enabled
        if (singleMode && !lastSingleMode) {
            // Disable all but one field. Store previous configuration to restore plots to how they were before single field mode was enabled.
            selectedField = nullptr;
            for (int i = 0; i < fields.size() - 1; i++) {
                fields[i].Restore = fields[i].Disabled;

                if (fields[i].Disabled)
                    continue;
                
                selectedField = &fields[i];

                fields[i].Disabled = true;
            }

            // If no fields were enabled, enable the last one
            if (selectedField == nullptr) {
                selectedField = &fields[fields.size() - 1];
                selectedField->Disabled = false;
                selectedField->Restore  = true;
            }
        }

        // Single field mode disabled
        if (!singleMode && lastSingleMode) {
            // Restore previous field configuration
            for (auto& field : fields)
                if (field.Disabled)
                    field.Disabled = field.Restore;
            
            selectedField = nullptr;
        }


        // Render the fields
        uint plottedFields = 0;
        for (auto& cont : fields) {
            const auto& field = cont.Field;

            if (field == nullptr) {
                DispWarning("Plot::Render", "A field was null. (Skipping)");
                continue;
            }
            if (field->Data == nullptr) {
                DispWarning("Plot::Render", "Field %s contained null data. (Skipping)", field->Name);
                continue;
            }
            
            if (*field->ElementCount == 0)
                continue;
            
            // Skip disabled fields
            if (cont.Disabled)
                continue;

            // Update over / under indicators
            overY  |= field->YLimits.End   > currentLimits.Max().y;
            underY |= field->YLimits.Start < currentLimits.Min().y;

            // Actually render the field
            // If an item is hovered, only render that item
            if (hoveredField != nullptr) {
                if (hoveredField->Field == field) {
                    field->Renderer->RenderPlot(*currentTool, &cont);
                }
            } else {
                field->Renderer->RenderPlot(*currentTool, &cont);
            }

            // Render error regions
            if (settings->ShowErrors) {
                if (field->ErrorRegions.size() > 0) {
                    for (auto& region : field->ErrorRegions) {
                        ImPlot::PushPlotClipRect();

                        auto dl = ImPlot::GetPlotDrawList();

                        auto tl = ImPlot::PlotToPixels(region.Start, limits.Y.Max);
                        auto br = ImPlot::PlotToPixels(region.End,   limits.Y.Min);

                        dl->AddLine(ImVec2(tl.x, plotRect.Min.y), ImVec2(br.x, plotRect.Min.y), IM_COL32(255, 0, 0, 255), 4);
                        dl->AddLine(ImVec2(tl.x, plotRect.Max.y), ImVec2(br.x, plotRect.Max.y), IM_COL32(255, 0, 0, 255), 4);

                        dl->AddRectFilled(tl, br, IM_COL32(255, 0, 0, settings->ErrorAlpha * 255));

                        ImPlot::PopPlotClipRect();
                    }
                }
            }

            // Render filter bands and annotations
            if (session != nullptr) {
                // Render filter bands on y axes
                for (auto filter : *session->Filters()->Get_Filters()) {
                    if (filter->IsFieldReference(field))
                        filter->RenderPlotPreviewSmall(0);
                }

                // Render annotations
                for (Annotation *notation : field->Annotations) {
                    notation->Render(this);
                    textInput |= ImGui::GetIO().WantTextInput;
                }
            }

            // Set status to warning if the field's texture dimensions are the wrong size
            if (cont.Rasterized && ((BFile*)field->File)->Type() == FileType::TBD) {
                auto plotLimits = ImPlot::GetPlotLimits(0);

                if (cont.TextureDimensions[1].x - cont.TextureDimensions[0].x != plotLimits.X.Max - plotLimits.X.Min)
                    status = PlotField::Status::Preview;
                if (cont.TextureDimensions[0].y - cont.TextureDimensions[1].y != plotLimits.Y.Max - plotLimits.Y.Min)
                    status = PlotField::Status::Preview;
            }

            plottedFields++;
        }

        // Handle keyboard controls
        if (ImPlot::IsPlotHovered() && !textInput) {
            if (ImGui::IsKeyDown(ImGuiKey_X)) {
                if (!lockReverse)
                    std::reverse(fields.begin(), fields.end());
                
                lockReverse = true;
            } else {
                lockReverse = false;
            }
            if (ImGui::IsKeyDown(ImGuiKey_Tab)) {
                if (!lockShift && fields.size() > 1)
                    std::rotate(fields.begin(), fields.begin() + 1, fields.end());
                
                lockShift = true;
            } else {
                lockShift = false;
            }
            if (ImGui::IsKeyDown(ImGuiKey_Slash) || ImGui::IsKeyDown(ImGuiKey_KeypadDivide)) {
                const double a = (limits.Y.Max - limits.Y.Min) / 4;

                limits.Y.Min += a;
                limits.Y.Max -= a;

                setNextY = true;
            }
            if (ImGui::IsKeyDown(ImGuiKey_8) || ImGui::IsKeyDown(ImGuiKey_KeypadMultiply)) {
                const double a = (limits.Y.Max - limits.Y.Min) / 4;

                limits.Y.Min -= a;
                limits.Y.Max += a;

                setNextY = true;
            }
            if (ImGui::IsKeyDown(ImGuiKey_DownArrow) || ImGui::IsKeyDown(ImGuiKey_KeypadSubtract) || ImGui::IsKeyDown(ImGuiKey_Minus)) {
                const double a = (limits.Y.Max - limits.Y.Min) / 4;

                limits.Y.Min -= a;
                limits.Y.Max -= a;

                setNextY = true;
            }
            if (ImGui::IsKeyDown(ImGuiKey_UpArrow) || ImGui::IsKeyDown(ImGuiKey_KeypadAdd) || ImGui::IsKeyDown(ImGuiKey_Equal)) {
                const double a = (limits.Y.Max - limits.Y.Min) / 4;

                limits.Y.Min += a;
                limits.Y.Max += a;

                setNextY = true;
            }
            if (ImGui::IsKeyDown(ImGuiKey_PageUp)) {
                const double a = (limits.Y.Max - limits.Y.Min);

                limits.Y.Min += a;
                limits.Y.Max += a;

                setNextY = true;
            }
            if (ImGui::IsKeyDown(ImGuiKey_PageDown)) {
                const double a = (limits.Y.Max - limits.Y.Min);

                limits.Y.Min -= a;
                limits.Y.Max -= a;

                setNextY = true;
            }
        }

        // Handle drag n drop onto plot
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("DND_PLOT")) {
                AddField(*(PlotField**)payload->Data);
            }
        }

        // Update query status
        if ((queried = ImPlot::IsPlotQueried())) {
            query = ImPlot::GetPlotQuery();
        }

        // Update and render tool
        if (*currentTool != nullptr)
            (*currentTool)->PlotUpdate(this);

        // Render the cursor
        renderCursor(this, currentTool);
        // Render plot info badges
        renderPlotInfo();

        // Render context menu
        if (ImGui::BeginPopup("##PlotContext")) {
            if (ctxMenu != nullptr) {
                ctxMenu(ctxMenuData);
                ImGui::Separator();
            }

            if (ImGui::BeginMenu("Fields"))
            {
                for (const auto& fieldCont : fields)
                {
                    const auto field = fieldCont.Field;

                    ImGui::SetNextWindowSize(ImVec2(160, 0), ImGuiCond_Always);
                    if (ImGui::BeginMenu(field->Name))
                    {
                        ImGui::PushID(field->Name);

                        if (field->Renderer->RenderWidget())
                            if (session != nullptr)
                                session->ReloadFields();

                        ImGui::PopID();

                        ImGui::EndMenu();
                    }
                }

                ImGui::EndMenu();
            }

            ImGui::Separator();
            // ImPlot stuff will render here during endplot
            ImGui::EndPopup();
        }


        // Render legend
        {
            ImGui::PushID(this);
            ImVec2 legendPos = ImPlot::GetPlotPos() + ImVec2(10, 10);
            ImGui::SetNextWindowPos(legendPos, ImGuiCond_Always, ImVec2(0, 0));

            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImPlot::GetStyleColorU32(ImPlotCol_LegendBg));
            ImGui::PushStyleColor(ImGuiCol_Border,  ImPlot::GetStyleColorU32(ImPlotCol_LegendBorder));
            ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);

            // Calc legend size
            ImVec2 legendSize = ImVec2(0, ImGui::GetStyle().FramePadding.y);
    
            for (uint i = 0; i < fields.size(); i++) {
                auto field = fields[i].Field;

                if (field == nullptr)
                    continue;

                const auto style = ImGui::GetStyle();

                legendSize.x = std::max(legendSize.x, ImGui::CalcTextSize(field->Name, 0, true).x + style.ItemSpacing.x + style.FramePadding.x * 2 + style.ItemSpacing.x * 2);
                legendSize.y += ImGui::GetTextLineHeightWithSpacing();// + style.FramePadding.y;
            }

            //legendSize.y -= ImGui::GetStyle().ItemSpacing.y;
            legendSize.y -= 2;

            hoveredField = nullptr;

            if (ImGui::BeginChildFrame(ImGui::GetID("legend"), legendSize)) {
                ImGui::PopStyleColor(2);
                ImGui::PopStyleVar();

                std::vector<int> eraseMe;
                for (int i = 0; i < fields.size(); i++) {
                    auto& fieldCont = fields[i];
                    auto  field     = fieldCont.Field;

                    if (field == nullptr)
                        continue;
                    
                    ImGui::PushID(field);

                    ImGui::BeginGroup();

                    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImPlot::GetStyleColorU32(ImPlotCol_LegendBg));
                    ImGui::PushStyleColor(ImGuiCol_Border,  ImPlot::GetStyleColorU32(ImPlotCol_LegendBorder));
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);

                    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
                    if (ImGui::Button("##fieldicon", ImVec2(16, 16))) {
                        fieldCont.Disabled = !fieldCont.Disabled;
                    }
                    ImGui::PopStyleVar();

                    ImGui::SameLine();

                    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y);

                    if (fieldCont.Disabled)
                        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                    else
                        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_Text));

                    ImGui::TextUnformatted(fields[i].Field->Name, ImGui::FindRenderedTextEnd(fields[i].Field->Name));

                    ImGui::PopStyleColor();

                    ImGui::PopStyleColor(2);
                    ImGui::PopStyleVar();

                    ImGui::EndGroup();

                    // Handle drag n drop from legend
                    if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
                        ImGui::SetDragDropPayload("DND_PLOT", &field, sizeof(PlotField*));
                        ImGui::TextUnformatted(field->Name, ImGui::FindRenderedTextEnd(field->Name));
                        renderMinimap(field, ImVec2(256, 64));
                        ImGui::EndDragDropSource();

                        eraseMe.push_back(i);
                    }

                    ImGui::OpenPopupOnItemClick("FieldContext", ImGuiPopupFlags_MouseButtonRight);

                    ImGui::SetNextWindowSize(ImVec2(300, -1), ImGuiCond_Always);
                    if (ImGui::BeginPopupContextWindow("FieldContext")) {
                        ImGui::TextEx(field->Name, ImGui::FindRenderedTextEnd(field->Name));
                        ImGui::TextDisabled("%s, %s", field->Catagory, ((BFile*)field->File)->Filename());
                        renderMinimap(field, ImVec2(ImGui::GetContentRegionAvailWidth(), 64));

                        if (field->Renderer->RenderWidget()) {
                            if (session != nullptr) {
                                BFile* file = (BFile*)field->File;

                                session->ReloadFields();
                            }
                        }

                        ImGui::EndPopup();
                    }

                    ImGui::PopID();
                }

                // Remove any fields which were drug away
                for (int i : eraseMe)
                    fields.erase(fields.begin() + i);
            } else {
                ImGui::PopStyleColor(2);
                ImGui::PopStyleVar();
            }
            ImGui::EndChildFrame();

            ImGui::PopID();
        }


		lastPlotRect   = plotRect;
        lastLimits     = limits;
        limits         = currentLimits;
        lastSingleMode = singleMode;

        ImPlot::EndPlot();
    }



    ImGui::PopID();

    ImPlot::PopStyleColor(3);

    // Cleanup for next run
    fitNext = false;
}

void Plot::renderMinimap(PlotField* field, const ImVec2& size) {
    if (field == nullptr) {
        DispWarning("Plot::renderMinimap", "Field was null");
        return;
    }

    auto prevCtx = ImPlot::GetCurrentContext();
    ImPlot::SetCurrentContext(mmCtx);

    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(2, 2));

    ImPlot::FitNextPlotAxes(true, true);
    if (ImPlot::BeginPlot("##minimap", 0, 0, size, ImPlotFlags_CanvasOnly, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations)) {
        BFile* file = (BFile*)field->File;
        auto mmDispField = file->Minimap(field);

        if (mmDispField == nullptr) {
            ImPlot::EndPlot();
            return;
        }

        if (file->Type() == FileType::Signal) {
            ImPlot::PlotLine("mm", mmDispField->Timing, mmDispField->Data, *mmDispField->ElementCount);
        } else {
            ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 1);
            ImPlot::PlotScatter("mm", mmDispField->Timing, mmDispField->Data, *mmDispField->ElementCount);
            ImPlot::PopStyleVar();
        }

        ImPlot::EndPlot();
    }

    ImPlot::PopStyleVar();
    ImPlot::SetCurrentContext(prevCtx);
}
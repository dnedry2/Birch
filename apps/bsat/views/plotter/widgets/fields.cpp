#include "plotter.hpp"

#include "svg.hpp"

#ifdef _WIN64
static char *strcasestr(const char *haystack, const char *needle)
{
    /* Edge case: The empty string is a substring of everything. */
    if (!needle[0])
        return (char *)haystack;

    /* Loop over all possible start positions. */
    for (size_t i = 0; haystack[i]; i++)
    {
        bool matches = true;
        /* See if the string matches here. */
        for (size_t j = 0; needle[j]; j++)
        {
            /* If we're out of room in the haystack, give up. */
            if (!haystack[i + j])
                return NULL;

            /* If there's a character mismatch, the needle doesn't fit here. */
            if (tolower((unsigned char)needle[j]) !=
                tolower((unsigned char)haystack[i + j]))
            {
                matches = false;
                break;
            }
        }
        if (matches)
            return (char *)(haystack + i);
    }
    return NULL;
}
#endif

// Simple string to int "hash". used to sort plot field catagories
static uint stringHash(const char* text)
{
    uint num = 0;
    for (int i = 0;; i++)
    {
        if (text[i] == '\0')
            return num;

        num *= 1000;
        num += text[i];
    }
}

static bool compareField(const PlotField *a, const PlotField *b)
{
    return stringHash(a->Catagory) > stringHash(b->Catagory);
}

Plotter::Fields_Widget::Fields_Widget(int ID, GLuint icon, Tool** tool, std::vector<Plot*>* plots, Session* session, PlotSyncSettings* syncSettings, PlotSettings* settings) : Widget(ID, icon, std::string("Fields"))
{
    this->plots        = plots;
    this->tool         = tool;
    this->filter[0]    = '\0';
    this->session      = session;
    this->syncSettings = syncSettings;
    this->settings     = settings;
}

static void renderToolTip(void* plotfield) {
    if (plotfield == nullptr)
        return;

    PlotField* field = (PlotField*)plotfield;

    if (ImGui::BeginTooltip()) {
        ImPlot::FitNextPlotAxes(true, true);

        if (ImPlot::BeginPlot("##mmPrevTT", 0, 0, ImVec2(256, 64), ImPlotFlags_CanvasOnly, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations)) {
            if (false && field->Renderer != nullptr) { //todo.... 
                //field->Renderer->RenderPlot();
            } else {
                BFile* file = (BFile*)field->File;
                auto mmDispField = file->Minimap(field);

                if (mmDispField == nullptr) {
                    ImPlot::EndPlot();
                    ImGui::EndTooltip();
                    return;
                }

                if (file->Type() == FileType::Signal) {
                    ImPlot::PlotLine("mm", mmDispField->Timing, mmDispField->Data, *mmDispField->ElementCount);
                } else {
                    ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 1);
                    ImPlot::PlotScatter("mm", mmDispField->Timing, mmDispField->Data, *mmDispField->ElementCount);
                    ImPlot::PopStyleVar();
                }
            }

            ImPlot::EndPlot();
        }

        ImGui::EndTooltip();
    }
}

void Plotter::Fields_Widget::Render()
{
    //static uint search = textures["bicons/IconZoom.svg"];
    static uint search = textures["bicons/IconZoom.svg"];

    // This needs to be done before rendering
    // Otherwise the minimap will not show if the field widget is not open
    if (minimap == nullptr)
        if (session != nullptr)
            if (session->Fields().size())
                minimap = session->Fields().front();
        

    if (beginWidget(this)) {
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 5);
        ImGui::Text("Minimap:");
        ImGui::SameLine();
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 5);

        std::string mmText = "(No fields)";

        if (minimap != nullptr)
            mmText = minimap->Name;

        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvailWidth());
        if (ImGui::BeginCombo("##MinimapSel", mmText.c_str())) {
            for (const auto& field : session->Fields()) {
                bool selected = minimap == field;
                
                if (ImGui::Selectable(field->Name, selected))
                    minimap = field;

                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
                fieldHoverTip.Render(field, &renderToolTip, field);
                ImGui::PopStyleVar();
            }
            ImGui::EndCombo();
        }

        ImGui::Separator();

        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvailWidth());
        ImGui::InputTextWithHint("##filterFields", "Search", filter, 128, ImGuiInputTextFlags_AutoSelectAll);
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() - 28);
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 6);
        ImGui::Image(t_ImTexID(search), ImVec2(14, 14));
        //ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 6);

        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGui::GetStyleColorVec4(ImGuiCol_FrameBg));
        if (ImGui::BeginChildFrame(ImGui::GetID(this), ImVec2(0, settingsOpen ? -170 : 0)))
        {
            ImGui::PopStyleColor();

            // Fields will already be sorted by catagory
            char initial[] = "\0";
            char *lastCatagory = initial;
            for (auto field : session->Fields())
            {
                if (strncmp(lastCatagory, field->Catagory, 8) != 0)
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(.75, 1, .75, 1));
                    ImGui::Text("%s", field->Catagory);
                    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 6);
                    ImGui::Separator();
                    ImGui::PopStyleColor();

                    lastCatagory = field->Catagory;
                }

                if (strcasestr(field->Name, filter) != NULL)
                {
                    if (ImGui::Selectable(field->Name, selected == field, ImGuiSelectableFlags_SpanAvailWidth)) {
                        if (selected == field)
                            selected = nullptr;
                        else
                            selected = field;
                    }

                    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
                    fieldHoverTip.Render(field->Name, &renderToolTip, field);
                    ImGui::PopStyleVar();

                    // Add new plot on double click
                    if (ImGui::IsItemClicked() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                    {
                        Plot* plot = new Plot(settings, syncSettings);

                        plot->AddField(field);
                        plots->push_back(plot);
                    }

                    if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
                    {
                        ImGui::SetDragDropPayload("DND_PLOT", &field, sizeof(field));
                        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
                        renderToolTip(field);
                        ImGui::PopStyleVar();
                        ImGui::EndDragDropSource();
                    }
                }
            }
        }
        else
        {
            ImGui::PopStyleColor();
        }
        ImGui::EndChildFrame();

        settingsOpen = selected != nullptr;

        if (settingsOpen) {
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGui::GetStyleColorVec4(ImGuiCol_FrameBg));
            if (ImGui::BeginChildFrame(123413485, ImVec2(0, 160)))
            {
                ImGui::PopStyleColor();

                selected->Renderer->RenderWidget();
            }
            else
            {
                ImGui::PopStyleColor();
            }
            ImGui::EndChildFrame();
        }
    }
    endWidget();
}
const PlotField* Plotter::Fields_Widget::Minimap() {
    return minimap;
}
Plotter::Fields_Widget::~Fields_Widget() { }
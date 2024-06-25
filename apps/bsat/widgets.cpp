#include <algorithm>
#include <cstdio>

#include "GL/gl3w.h"
#include "imgui_internal.h"

#include "tooltip.hpp"
#include "widgets.hpp"
#include "bfile.hpp"

#include "defines.h"

using namespace Birch;

// Window resources
static GLuint input;

void InitWidgets(GLuint x, GLuint expand, GLuint hide, GLuint visible, GLuint hidden, GLuint search_t, GLuint input_t)
{
    input = input_t;
}

bool beginWidget(Widget *widget)
{
    ImGui::PushID(widget);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar;
    if (!widget->Docked)
        flags |= ImGuiWindowFlags_MenuBar;

    if (ImGui::Begin(widget->Name().c_str(), NULL, flags)) {
        widget->Docked = ImGui::IsWindowDocked();

        // Render custom menubar if widget is not docked
        if (!widget->Docked && ImGui::BeginMenuBar())
        {
            if (widget->Icon != 0)
            {
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 5);
                ImGui::Image(t_ImTexID(widget->Icon), ImVec2(16, 16));
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 5);
            }

            ImGui::Text("%s", widget->SafeName().c_str());

            /*
            float width = ImGui::GetWindowSize().x;
            ImGui::SetCursorPos(ImVec2(width - 28, 2));

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));

            if (ImGui::ImageButton(t_ImTexID(widget->Collapsed ? windowExpand : windowHide), ImVec2(12, 12)))
                widget->Collapsed = !widget->Collapsed;

            ImGui::PopStyleColor();
            */

            ImGui::EndMenuBar();
        }

        return true;
    }

    return false;
}
void endWidget()
{
    ImGui::End();
    ImGui::PopID();
}

static std::string buildWindowTitle(const std::string& name, int id) {
    return std::string(name) + "##" + std::to_string(id);
}
static ImU32 floatToCol(const float* rgb)
{
    return ImGui::ColorConvertFloat4ToU32(ImVec4(rgb[0], rgb[1], rgb[2], 1));
}

Widget::~Widget() {}
Widget::Widget(int id, GLuint icon, std::string name)
{
    Icon = icon;
    ID = id;
    this->name = buildWindowTitle(name, id);
    this->safeName = name;
}
std::string Widget::Name()
{
    return name;
}
std::string Widget::SafeName()
{
    return safeName;
}

#pragma region Info_Widget
Info_Widget::Info_Widget(GLuint icon, BFile *file, Session* session, int id) : Widget(id, icon, std::string(file->Filename()))
{
    this->file    = file;
    this->session = session;

    type = file->Type();

    Closable = false;
}
void Info_Widget::Render()
{
    if (beginWidget(this)) {
        // Data associations
        BFile* assoc  = nullptr;
        std::string preview = "None";
        std::string label   = "TBD File: ";

        if (type == FileType::Signal) {
            auto* signal = (SignalFile*)file;

            if (signal->AssocTBD() != nullptr) {
                assoc   = signal->AssocTBD();
                preview = assoc->Filename();
            }
        } else {
            auto* tdb = (TBDFile*)file;

            if (tdb->AssocSig() != nullptr) {
                assoc   = tdb->AssocSig();
                preview = assoc->Filename();
            }

            label = "IQ File:";
        }

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 5);
        ImGui::Text("%s", label.c_str());
        
        ImGui::SameLine();

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 5);

        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        if (ImGui::BeginCombo("##assoc_picker", preview.c_str())) {
            // Display either TDB or signal file types, Whichever is not the current type
            const auto* files = session->Files();

            for (int i = 0; i < files->size(); i++) {
                if (files->at(i)->Type() == type)
                    continue;

                bool selected = assoc == files->at(i);

                if (ImGui::Selectable(files->at(i)->Filename(), selected)) {
                    if (type == FileType::Signal)
                        ((SignalFile*)file)->AssocTBD() = (TBDFile*)files->at(i);
                    else
                        ((TBDFile*)file)->AssocSig() = (SignalFile*)files->at(i);
                }
            }

            

            ImGui::EndCombo();
        }

        // Render plugin's info
        file->RenderSidebar();
    }

    endWidget();
}
#pragma endregion

#pragma region Colormap_Widget
Colormap_Widget::Colormap_Widget(int id, GLuint icon, Session* session, Gradient* defaultGradient, Gradient *defaultSpectraGradient, std::vector<Gradient*>* gradients) : Widget(id, icon, std::string("Colormap"))
{
    //this->plotter = plotter;

    this->session = session;

    selectedGradient = defaultGradient;
    spectraGradient  = defaultSpectraGradient;
    this->gradients = gradients;
    this->ToolID = "Color";
}

ImU32 Colormap_Widget::Get_SelectedColor()
{
    return floatToCol(selectedColor);
}
ImU32 Colormap_Widget::Get_BaseColor()
{
    return floatToCol(baseColor);
}
Gradient* Colormap_Widget::Get_SpectraColor() {
    return spectraGradient;
}
void Colormap_Widget::AddColor(Colorizer* color) {
    DispWarning("Colormap_Widget::AddColor", "Should not be called!");
    session->Colormap()->AddColor(color);
}

static bool colorLayerDropTarget(int i, ColorStack* colormap) {
    bool changed = false;

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,  ImVec2(0, 0));

    if (ImGui::BeginChild("##ddTarget", ImVec2(0, 3), false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoDecoration))
        ImGui::EndChild();

    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 2);
    
    ImGui::PopStyleVar(2);

    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ColorLayer"))
        {
            int payload_n = *(const int*)payload->Data;

            if (payload_n != i)
            {
                Colorizer* layer = colormap->Get_Colors()->at(payload_n);

                colormap->Get_Colors()->erase(colormap->Get_Colors()->begin() + payload_n);
                colormap->Get_Colors()->insert(colormap->Get_Colors()->begin() + i, layer);

                changed = true;
            }
        }

        ImGui::EndDragDropTarget();
    }

    return changed;
}

void Colormap_Widget::Render() {
    if (beginWidget(this)) {
        bool changed = false;

        // Main Settings
        //Gradient* dummyg;
        //bool      dummyb;

        session->Colormap()->Set_BaseColor(Get_BaseColor());
        //changed |= GradientPicker("Color Tool Gradient", &selectedGradient, ImVec2(25, 25), gradients, &dummyg, &dummyb);
        //changed |= ImGui::ColorEdit3("Color Tool Solid", selectedColor, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);


        // Select first avaiable field if none is selected
        if (selectedField == nullptr && session->Fields().size() != 0) {
            for (auto field : session->Fields())
            {
                if (strncmp(field->Catagory, "PDW", 3))
                    continue;
                
                selectedField = field;

                break;
            }
        }

        if (ImGui::BeginTabBar("##colormapTabBar")) {
            if (ImGui::BeginTabItem("TBD")) {
                const float width = ImGui::GetContentRegionAvail().x - 42;

                ImGui::SetNextItemWidth(width - 90);
                if (ImGui::BeginCombo("##colorFieldSel", selectedField != nullptr ? selectedField->Name : "(No fields)"))
                {
                    for (auto field : session->Fields())
                    {
                        if (strncmp(field->Catagory, "PDW", 3))
                            continue;

                        bool selected = selectedField == field;

                        if (ImGui::Selectable(field->Name, &selected))
                            selectedField = field;
                    }
                    ImGui::EndCombo();
                }

                fieldTip.Render("Select the field to add a layer to");

                ImGui::SameLine();
                ImGui::SetNextItemWidth(90);

                int solidInt = solid ? 1 : 0;
                ImGui::Combo("##solid/gradient", &solidInt, "Gradient\0Solid\0");
                solid = solidInt == 1;

                solidTip.Render("Add a solid color layer or a gradient layer");

                ImGui::SameLine();

                if (solid) {
                    ImGui::ColorEdit3("", selectedColor, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
                }
                else {
                    Gradient* dummyg;
                    bool      dummyb;

                    GradientPicker("", &selectedGradient, ImVec2(25, 25), gradients, &dummyg, &dummyb);
                }

                ImGui::SetNextItemWidth(80);
                int rangeInt = range ? 1 : 0;
                ImGui::Combo("##range/quantile", &rangeInt, "Center\0Range\0");
                range = rangeInt == 1;

                rangeTip.Render("Input values as a center and range or as a min and max");

                ImGui::SameLine();

                RangeInput(&inputCenter, &inputWidth, range, ImVec2(-42, 0));

                ImGui::SameLine();

                ImGui::SetNextItemWidth(42);
                if (ImGui::Button("Add"))
                {
                    swapFocus = true;

                    if (selectedField != nullptr)
                    {
                        Colorizer *color;

                        if (solid)
                            color = new ColorizerSingle(&selectedField->Data, selectedField->Name, inputCenter, inputWidth, Get_SelectedColor());
                        else
                            color = new ColorizerRange(&selectedField->Data,  selectedField->Name, inputCenter, inputWidth, Get_SelectedGradient(), gradients);

                        session->Colormap()->AddColor(color);
                        changed = true;
                    }
                }

                ImGui::SameLine();
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 4);
                ImGui::Image(t_ImTexID(input), ImVec2(20, 20));
                //ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 6);

                // Render list of color layers
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
                ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGui::GetStyleColorVec4(ImGuiCol_FrameBg));
                if (ImGui::BeginChildFrame(3000, ImVec2(0, -35)))
                {
                    ImGui::PopStyleVar();
                    ImGui::PopStyleColor();

/*
                    const int count = colormap->Get_Colors()->size();
                    bool odd = false;
                    for (int i = count - 1; i >= 0; i--, odd ^= 1)
                    {
                        ImGui::PushID();
                    }
*/

                    bool odd = false;
                    int i = 0;
                    for (auto layer : *session->Colormap()->Get_Colors())
                    {
                        ImGui::PushID(&layer);

                        changed |= colorLayerDropTarget(i, session->Colormap());

                        changed |= layer->RenderWidget(i, odd, range);

                        // Drag and drop to reorder
                        if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
                        {
                            ImGui::SetDragDropPayload("ColorLayer", &i, sizeof(int));

                            if (ImGui::BeginChild("##ddFrame", ImVec2(256, 52), false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoDecoration))
                            {
                                layer->RenderWidget(i, odd, range);
                                ImGui::EndChild();
                            }

                            ImGui::EndDragDropSource();
                        }

                        i++;
                        odd = !odd;

                        ImGui::PopID();
                    }

                    i--; // i = last index

                    changed |= colorLayerDropTarget(i, session->Colormap());

                    // Base color "layer". Not a real layer, must always be last
                    ImGui::BeginGroup();
                    ImVec2 cursor = ImGui::GetCursorScreenPos();
                    ImVec2 frameEnd = ImVec2(cursor.x + ImGui::GetWindowContentRegionWidth(), cursor.y + 52);

                    ImGui::RenderFrame(cursor, frameEnd, odd? 0x16000000 : 0x32000000, true, 5);

                    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 12);
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 6);
                    changed |= ImGui::ColorEdit3("Base color", baseColor, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);

                    ImGui::EndGroup();
                }
                else
                {
                    ImGui::PopStyleVar();
                    ImGui::PopStyleColor();
                }
                ImGui::EndChildFrame();

                if (ImGui::Button("Clear"))
                {
                    changed = true;
                    session->Colormap()->Get_Colors()->clear();
                }

                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Spectra")) {
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
                ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGui::GetStyleColorVec4(ImGuiCol_FrameBg));

                if (ImGui::BeginChildFrame(ImGui::GetID("spectraColors"), ImVec2(0, 0))) {
                    ImGui::PopStyleVar();
                    ImGui::PopStyleColor();

                    bool odd = false;
                    int i = 0;
                    for (auto col : *session->Colormap()->Get_SpectraColors()) {
                        changed |= col->RenderWidget(i++, odd, false);
                        odd = !odd;
                    }
                }
                else
                {
                    ImGui::PopStyleVar();
                    ImGui::PopStyleColor();
                }
                ImGui::EndChildFrame();

                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }

        if (changed)
            version++;
    }
    endWidget();
}

Colormap_Widget::~Colormap_Widget() { }
#pragma endregion
#pragma region Filter_Widget
Filter_Widget::Filter_Widget(int id, GLuint icon, Session* session) : Widget(id, icon, std::string("Filter"))
{
    this->session = session;
    this->ToolID = "Filter";
}
void Filter_Widget::Render()
{
    if (beginWidget(this)) {
        if (selectedField == nullptr && session->Fields().size() != 0)
            selectedField = (session->Fields())[0];

        if (ImGui::BeginCombo("##filterFieldSel", selectedField != nullptr ? selectedField->Name : "(No fields)"))
        {
            for (auto field : session->Fields())
            {
                bool selected = selectedField == field;

                if (ImGui::Selectable(field->Name, &selected))
                    selectedField = field;
            }
            ImGui::EndCombo();
        }

        ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        ImGui::Combo("##pass/reject", &pass, "Pass\0Reject\0");
        
        
        bool changed = false;

        ImGui::SetNextItemWidth(80);
        int rangeInt = range ? 1 : 0;
        ImGui::Combo("##range/quantile", &rangeInt, "Center\0Range\0");
        range = rangeInt == 1;

        rangeTip.Render("Input values as a center and range or as a min and max");

        ImGui::SameLine();

        RangeInput(&inputCenter, &inputWidth, range, ImVec2(-42, 0));

        ImGui::SameLine();

        ImGui::SetNextItemWidth(42);
        if (ImGui::Button("Add"))
        {
            swapFocus = true;

            if (selectedField != nullptr)
            {
                double qMin = inputCenter - inputWidth * 0.5;
                double qMax = inputCenter + inputWidth * 0.5;

                TBDFilter *filter = new TBDFilter(&selectedField->Data, selectedField->Name, qMin, qMax, !pass);
                session->Filters()->AddFilter(filter);

                changed = true;
            }
        }

        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImGui::GetStyleColorVec4(ImGuiCol_FrameBg));
        if (ImGui::BeginChildFrame(1000001, ImVec2(0, -35)))
        {
            ImGui::PopStyleVar();
            ImGui::PopStyleColor();

            bool odd = false;
            int i = 0;
            for (auto layer : *session->Filters()->Get_Filters())
            {
                changed |= layer->RenderWidget(i++, odd, range);
                odd = !odd;
            }
        }
        else
        {
            ImGui::PopStyleVar();
            ImGui::PopStyleColor();
        }
        ImGui::EndChildFrame();

        if (ImGui::Button("Clear"))
        {
            changed = true;
            session->Filters()->Get_Filters()->clear();
        }

        ImGui::SameLine();
        changed |= ImGui::Checkbox("Link IQ", session->Filters()->Get_LinkPtr());

        if (changed)
            version++;
    }
    endWidget();
}
Filter_Widget::~Filter_Widget()
{
}
#pragma endregion

#pragma region PluginHost_Widget 
PluginHost_Widget::PluginHost_Widget(PluginIQProcessor* plugin, int id) : Widget(id, plugin->ToolbarIcon(), std::string(plugin->Name())) {
    this->plugin = plugin;
}
PluginHost_Widget::~PluginHost_Widget() {

}

void PluginHost_Widget::Render() {
    if (plugin->ShouldRenderWindow())
        plugin->RenderWindow();

    if (!plugin->ShouldRenderSidebar())
        return;

    if (beginWidget(this))
        plugin->RenderSidebar();
    
    endWidget();
}
#pragma endregion

bool WindowWidget(WindowFunc** current, const std::vector<WindowFunc*>* windows) {
    bool changed = false;

    ImGui::SetNextWindowSizeConstraints(ImVec2(512, 132), ImVec2(140, 512));
    if (ImGui::BeginCombo("##winsel", (*current)->Name())) {
        if (ImGui::BeginChild("#listbox", ImVec2(128, 256))) {
            for (auto& win : *windows) {
                if (ImGui::Selectable(win->Name(), *current == win)) {
                    *current = win;
                    changed = true;
                }
            }
            ImGui::EndChild();
        }

        ImGui::SameLine();

        (*current)->RenderWidget();

        ImGui::EndCombo();
    }
    return changed;
}


static ImVector<ImRect> s_GroupPanelLabelStack;

bool BeginGroupBox(const char* name, const ImVec2& size);
void EndGroupBox();

void BeginGroupPanel(const char* name, const ImVec2& size)
{
    ImGui::BeginGroup();

    auto cursorPos = ImGui::GetCursorScreenPos();
    auto itemSpacing = ImGui::GetStyle().ItemSpacing;
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));

    auto frameHeight = ImGui::GetFrameHeight();
    ImGui::BeginGroup();

    ImVec2 effectiveSize = size;
    if (size.x < 0.0f)
        effectiveSize.x = ImGui::GetContentRegionAvailWidth();
    else
        effectiveSize.x = size.x;
    ImGui::Dummy(ImVec2(effectiveSize.x, 0.0f));

    ImGui::Dummy(ImVec2(frameHeight * 0.5f, 0.0f));
    ImGui::SameLine(0.0f, 0.0f);
    ImGui::BeginGroup();
    ImGui::Dummy(ImVec2(frameHeight * 0.5f, 0.0f));
    ImGui::SameLine(0.0f, 0.0f);
    ImGui::TextUnformatted(name);
    auto labelMin = ImGui::GetItemRectMin();
    auto labelMax = ImGui::GetItemRectMax();
    ImGui::SameLine(0.0f, 0.0f);
    ImGui::Dummy(ImVec2(0.0, frameHeight + itemSpacing.y));
    ImGui::BeginGroup();

    //ImGui::GetWindowDrawList()->AddRect(labelMin, labelMax, IM_COL32(255, 0, 255, 255));

    ImGui::PopStyleVar(2);

#if IMGUI_VERSION_NUM >= 17301
    ImGui::GetCurrentWindow()->ContentRegionRect.Max.x -= frameHeight * 0.5f;
    ImGui::GetCurrentWindow()->WorkRect.Max.x          -= frameHeight * 0.5f;
    ImGui::GetCurrentWindow()->InnerRect.Max.x         -= frameHeight * 0.5f;
#else
    ImGui::GetCurrentWindow()->ContentsRegionRect.Max.x -= frameHeight * 0.5f;
#endif
    ImGui::GetCurrentWindow()->Size.x                   -= frameHeight;

    auto itemWidth = ImGui::CalcItemWidth();
    ImGui::PushItemWidth(ImMax(0.0f, itemWidth - frameHeight));

    s_GroupPanelLabelStack.push_back(ImRect(labelMin, labelMax));
}

void EndGroupPanel()
{
    ImGui::PopItemWidth();

    auto itemSpacing = ImGui::GetStyle().ItemSpacing;

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));

    auto frameHeight = ImGui::GetFrameHeight();

    ImGui::EndGroup();

    //ImGui::GetWindowDrawList()->AddRectFilled(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(0, 255, 0, 64), 4.0f);

    ImGui::EndGroup();

    ImGui::SameLine(0.0f, 0.0f);
    ImGui::Dummy(ImVec2(frameHeight * 0.5f, 0.0f));
    ImGui::Dummy(ImVec2(0.0, frameHeight - frameHeight * 0.5f - itemSpacing.y));

    ImGui::EndGroup();

    auto itemMin = ImGui::GetItemRectMin();
    auto itemMax = ImGui::GetItemRectMax();
    //ImGui::GetWindowDrawList()->AddRectFilled(itemMin, itemMax, IM_COL32(255, 0, 0, 64), 4.0f);

    auto labelRect = s_GroupPanelLabelStack.back();
    s_GroupPanelLabelStack.pop_back();

    ImVec2 halfFrame = ImVec2(frameHeight * 0.25f, frameHeight) * 0.5f;
    ImRect frameRect = ImRect(itemMin + halfFrame, itemMax - ImVec2(halfFrame.x, 0.0f));
    labelRect.Min.x -= itemSpacing.x;
    labelRect.Max.x += itemSpacing.x;
    for (int i = 0; i < 4; ++i)
    {
        switch (i)
        {
            // left half-plane
            case 0: ImGui::PushClipRect(ImVec2(-FLT_MAX, -FLT_MAX), ImVec2(labelRect.Min.x, FLT_MAX), true); break;
            // right half-plane
            case 1: ImGui::PushClipRect(ImVec2(labelRect.Max.x, -FLT_MAX), ImVec2(FLT_MAX, FLT_MAX), true); break;
            // top
            case 2: ImGui::PushClipRect(ImVec2(labelRect.Min.x, -FLT_MAX), ImVec2(labelRect.Max.x, labelRect.Min.y), true); break;
            // bottom
            case 3: ImGui::PushClipRect(ImVec2(labelRect.Min.x, labelRect.Max.y), ImVec2(labelRect.Max.x, FLT_MAX), true); break;
        }

        ImGui::GetWindowDrawList()->AddRect(
            frameRect.Min, frameRect.Max,
            ImColor(ImGui::GetStyleColorVec4(ImGuiCol_Border)),
            halfFrame.x);

        ImGui::PopClipRect();
    }

    ImGui::PopStyleVar(2);

#if IMGUI_VERSION_NUM >= 17301
    ImGui::GetCurrentWindow()->ContentRegionRect.Max.x += frameHeight * 0.5f;
    ImGui::GetCurrentWindow()->WorkRect.Max.x          += frameHeight * 0.5f;
    ImGui::GetCurrentWindow()->InnerRect.Max.x         += frameHeight * 0.5f;
#else
    ImGui::GetCurrentWindow()->ContentsRegionRect.Max.x += frameHeight * 0.5f;
#endif
    ImGui::GetCurrentWindow()->Size.x                   += frameHeight;

    ImGui::Dummy(ImVec2(0.0f, 0.0f));

    ImGui::EndGroup();
}



bool BeginGroupBox(const char* name, const ImVec2& size)
{
    ImGui::PushID(name);

    if (ImGui::BeginChildFrame(ImGui::GetID(name), size, ImGuiWindowFlags_MenuBar)) {
        if (ImGui::BeginMenuBar()) {
            ImGui::TextUnformatted(name);
            ImGui::EndMenuBar();
        }

        return true;
    }

    return false;
}
void EndGroupBox()
{
    ImGui::EndChildFrame();
    ImGui::PopID();
}



bool RangeInput(double* center, double* width, bool dispRange, const ImVec2& size) {
    bool changed = false;

    ImGui::PushID(&center);
    ImGui::BeginGroup();

    const float availWidth = size.x > 0 ? size.x : ImGui::GetContentRegionAvail().x + size.x;

    ImGui::PushMultiItemsWidths(2, availWidth);
    if (dispRange) {
        Birch::Span<float> range(*center - *width * 0.5, *center + *width * 0.5);

        changed |= ImGui::DragFloat("##min", &range.Start, 1.0, -FLT_MAX,    range.End, "Min: %.3f");
        ImGui::SameLine();
        changed |= ImGui::DragFloat("##max", &range.End,   1.0, range.Start, FLT_MAX,   "Max: %.3f");

        if (changed) {
            *center = range.Center();
            *width  = range.Length();
        }
    } else {
        float c = *center;
        float w = *width;

        changed |= ImGui::DragFloat("##center", &c, 1.0, -FLT_MAX, FLT_MAX, "Center: %.3f");
        ImGui::SameLine();
        changed |= ImGui::DragFloat("##pm",     &w, 1.0, 0.0,      FLT_MAX, "± %.3f");

        if (changed) {
            *center = c;
            *width  = w;
        }
    }

    ImGui::PopItemWidth();

    ImGui::EndGroup();
    ImGui::PopID();

    return changed;
}
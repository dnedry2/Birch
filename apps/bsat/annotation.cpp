#include "annotation.hpp"
#include "plot.hpp"
#include "ImPlot_Extras.h"
#include <GL/gl3w.h>

#include "defines.h"

//#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "shellExec.hpp"

GLuint rasterizeImage(const unsigned* pixels, int width, int height) {
    GLuint tex;

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glActiveTexture(tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, height, width, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    return tex;
}

static GLuint Note;

void InitAnnotations(GLuint note)
{
    Note = note;
}

static ImVec2 getPos(ImVec2 an, ImVec2 offset, ImVec2 size)
{
    ImVec2 pos = an;

    if (offset.x == 0)
        pos.x -= size.x / 2;
    else if (offset.x > 0)
        pos.x += offset.x;
    else
        pos.x -= size.x - offset.x;
    if (offset.y == 0)
        pos.y -= size.y / 2;
    else if (offset.y > 0)
        pos.y += offset.y;
    else
        pos.y -= size.y - offset.y;

    return pos;
}

Annotation::Annotation(const char* text, double* position, double* bounds, int plot, int id, bool img)
{
    Text = new char[128];
    strncpy(Text, text, 128);

    Position[0] = position[0];
    Position[1] = position[1];

    Bounds[0] = bounds[0];
    Bounds[1] = bounds[1];

    PlotID = plot;
    ID = id;

    snprintf(this->id, 16, "##ad_%d", id);
/*
    if (img) {
        int width;
        int height;
        int channels;
        unsigned char *buf = stbi_load("test.jpg", &width, &height, &channels, 4);

        this->img = rasterizeImage((unsigned int*)buf, width, height);

        free(buf);
    }
*/
};

static void AddUnderLine(ImColor col_)
{
  ImVec2 min = ImGui::GetItemRectMin();
  ImVec2 max = ImGui::GetItemRectMax();
  max.y = max.y - 2;
  min.y = max.y;
  ImGui::GetWindowDrawList()->AddLine(min, max, col_, 1.0f);
}

void Annotation::Render(void* plotv)
{
    Plot* plot = (Plot*)plotv;

    if (Position[0] >= plot->Limits().X.Min && Position[0] <= plot->Limits().X.Max) {
        if (plot->Limits().X.Max - plot->Limits().X.Min < (Bounds[1] - Bounds[0]) * 4) {
            if (img) {
                auto tr = ImPlot::PlotToPixels(ImPlotPoint(Position[0], Position[1]));
                if (ImGui::IsMouseHoveringRect(ImVec2(tr.x, tr.y - 20), ImVec2(tr.x + 20, tr.y))) {
                    ImPlot::PlotImage("##Annotations", t_ImTexID(Note), ImPlotPoint(Position[0], Position[1]), ImVec2(18, 18), ImVec2(0, -20));
                    ImGui::BeginTooltip();
                    ImGui::Image(t_ImTexID(img), ImVec2(128, 128));
                    ImGui::EndTooltip();
                } else {
                    ImPlot::PlotImage("##Annotations", t_ImTexID(Note), ImPlotPoint(Position[0], Position[1]), ImVec2(18, 18), ImVec2(0, -20));
                }
            } else {
                ImPlot::PushPlotClipRect();

                ImVec2 offset = ImVec2(Position[0] > plot->Limits().X.Min + (plot->Limits().X.Max - plot->Limits().X.Min) / 2 ? -16 : 16, -16);
                const ImVec2 txt_size = ImGui::CalcTextSize(Text);
                const ImVec2 size = txt_size + ImPlot::GetStyle().AnnotationPadding * 2 + ImVec2(2, 0);
                ImVec2 basePos = ImPlot::PlotToPixels(ImVec2(Position[0], Position[1]));
                ImVec2 pos = getPos(basePos, offset, size);

                ImDrawList* drawList = ImGui::GetWindowDrawList();

                //render box
                ImRect rect(pos, pos + size);
                ImVec2 corners[4] = { rect.GetTL(), rect.GetTR(), rect.GetBR(), rect.GetBL() };
                int min_corner = 0;
                float min_len = FLT_MAX;
                for (int c = 0; c < 4; ++c) {
                    float len = ImLengthSqr(basePos - corners[c]);
                    if (len < min_len) {
                        min_corner = c;
                        min_len = len;
                    }
                }
                drawList->AddLine(basePos, corners[min_corner], Color + 206698591541);
                drawList->AddRectFilled(rect.Min, rect.Max, Color, 5);
                drawList->AddRect(rect.Min, rect.Max, Color + 206698591541, 5);

                //render text
                ImVec2 cPos = ImGui::GetCursorScreenPos();
                ImGui::SetCursorScreenPos(pos - ImVec2(2, 4));

                ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0, 0, 0, 0));
                ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0, 0, 0, 0));
                ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0, 0, 0, 0));

                ImGui::SetNextItemWidth(txt_size.x + 10);

                ImGuiInputTextFlags flags = ImGuiInputTextFlags_NoHorizontalScroll | ImGuiInputTextFlags_AutoSelectAll;
                const bool ctrlDown = ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl);

                if (ImGui::BeginChild(id, ImVec2(txt_size.x + 10, txt_size.y + 10), false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
                    if (!ctrlDown) {
                        ImGui::InputText(id, Text, 128, flags);
                        if (initial) {
                            ImGui::SetFocusID(ImGui::GetID(id), ImGui::GetCurrentWindow());
                            initial = false;
                        }
                    } else {
                        ImGui::SetCursorScreenPos(pos + ImVec2(3, 1));
                        ImGui::Text("%s", Text);
                        if (ImGui::IsItemHovered(0)) {
                            ImGui::SetFocusID(ImGui::GetID(Text), ImGui::GetCurrentWindow());
                            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                            AddUnderLine(ImColor(255, 255, 255, 255));
                        }
                        if (ImGui::IsItemClicked(0))
                            OpenWebpage(Text);
                    }
                    ImGui::EndChild();
                }

                ImGui::PopStyleColor(3);

                ImGui::SetCursorScreenPos(cPos);

                ImPlot::PopPlotClipRect();
            }
        } else {
            ImPlot::PlotImage("##Annotations", t_ImTexID(Note), ImPlotPoint(Position[0], Position[1]), ImVec2(18, 18), ImVec2(0, -20));
        }
    }
}
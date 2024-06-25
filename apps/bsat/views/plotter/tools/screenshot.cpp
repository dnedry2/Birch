#include "plotter.hpp"
#include "logger.hpp"

#include "nfd.h"
#include "stb_image_write.h"

#include <string>

static inline void highlightCurrentPlot(ImU32 color) {
    auto dl = ImGui::GetWindowDrawList();

    auto hoveredMin = ImGui::GetItemRectMin();
    auto hoveredMax = ImGui::GetItemRectMax();

    dl->AddRectFilled(hoveredMin, hoveredMax, color, 5);

}

static std::string screenshotToolHelp = 
    "Click to take a screenshot of a plot.\n"
    "Hold shift to select multiple plots.";

Plotter::ScreenshotTool::ScreenshotTool(const std::vector<Tool*>* tools)
                        : Tool("Screenshot", screenshotToolHelp.c_str(), false, textures["bicons/IconCamera.svg"], textures["bicons/CursorCamera.svg"], textures["bicons/CursorCamera.svg"], ImVec2(8, 8), ImVec2(8, 8), ImGuiKey_G),
                          tools(tools) { }

void Plotter::ScreenshotTool::on_selection(ImPlotInputMap* controls, ImPlotFlags* flags) {
    controls->VerticalMod = -1;
    controls->HorizontalMod = -1;
    controls->BoxSelectMod = -1;

    controls->BoxSelectButton = -1;
    controls->PanButton       = -1;

    *flags = ImPlotFlags_None | ImPlotFlags_NoMousePos;

    controlsPtr = controls;
    flagsPtr    = flags;

    if (!screenshotInProgress)
        selected.clear();
}

void Plotter::ScreenshotTool::on_update(Plot* plot) {
    // Need to let it loop through all plots once first, removing all of the highlights
    // Once the clicked plot has been rendered twice, we know all plots have been re rendered
    // (it has to be twice, because the clicked plot may not be the last plot rendered)
    // This is also where other tool's plot graphics are rendered
    if (popupOpen && !captured && !captureReady) {
        // Render all tool overlays
        for (auto& tool : *tools) {
            if (tool == this)
                continue;
            
            tool->RenderGraphics(plot);
        }

        if (plot == clickedPlot) {
            clickedPlotRC++;
            if (clickedPlotRC > 2) {
                clickedPlotRC = 0;
                captureReady = true;
            }
        }

        return;
    }

    // Take screenshot before drawing any highlights, if needed
    if (popupOpen && !captured && captureReady) {
        // Get screenshot pixel region
        ImVec2 offset = ImGui::GetWindowViewport()->Pos;
        ImVec2 size   = ImGui::GetWindowViewport()->Size;

        // Get pixels from each selected plot, then merge them at the end
        struct ImgData {
            uint* pixels;
            uint  width;
            uint  height;
        };
        std::vector<ImgData> plotRenders;

        for (const auto& plt : selected) {
            auto rect = plt->PlotRect();

            rect.Min.x -= offset.x;
            rect.Min.y -= offset.y;
            rect.Max.x -= offset.x;
            rect.Max.y -= offset.y;

            // Lower left corner
            ImVec2 pixStart = ImVec2(rect.Min.x, size.y - rect.Max.y);

            // Copy region to pix buf
            uint  width  = static_cast<uint>(rect.GetWidth());
            uint  height = static_cast<uint>(rect.GetHeight());
            uint* pixels = new uint[width * height];

            glReadPixels(pixStart.x, pixStart.y, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

            // Flip image vertically
            for (uint i = 0; i < height / 2; i++)
                for (uint j = 0; j < width; j++)
                    std::swap(pixels[i * width + j], pixels[(height - i - 1) * width + j]);

            plotRenders.push_back({ pixels, width, height });
        }

        // Merge images vertically
        screenshotHeight = 0;
        screenshotWidth  = 0;

        for (const auto& img : plotRenders) {
            screenshotHeight += img.height;
            screenshotWidth   = std::max(screenshotWidth, img.width);
        }

        screenshotPixels = new uint[screenshotWidth * screenshotHeight];

        uint yOffset = 0;
        for (const auto& img : plotRenders) {
            for (uint i = 0; i < img.height; i++)
                memcpy(screenshotPixels + (yOffset + i) * screenshotWidth, img.pixels + i * img.width, img.width * sizeof(uint));

            yOffset += img.height;

            delete[] img.pixels;
        }

        // Create texture
        glGenTextures(1, &screenshotTexture);
        glBindTexture(GL_TEXTURE_2D, screenshotTexture);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, screenshotWidth, screenshotHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, screenshotPixels);

        // Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        captured = true;
    }

    // Sreenshot save dialog
    if (ImGui::BeginPopupModal("Save Screenshot", &popupOpen)) {
        // Scale image to fit window
        uint xRes = ImGui::GetContentRegionAvailWidth();
        uint yRes = xRes * screenshotHeight / screenshotWidth;
        ImGui::Image(t_ImTexID(screenshotTexture), ImVec2(xRes, yRes));

        if (ImGui::Button("Save")) {
            // Get file name with nfd
            nfdchar_t* outPath = nullptr;
            nfdresult_t result = NFD_SaveDialog("png", nullptr, &outPath);

            if (result == NFD_OKAY) {
                stbi_write_png(outPath, screenshotWidth, screenshotHeight, 4, screenshotPixels, screenshotWidth * 4);
                free(outPath);
            }
        }

        ImGui::SameLine();

        if (ImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
            popupOpen = false;
        }
        
        ImGui::EndPopup();
    }

    // Cleanup when dialog closed
    if (!popupOpen && lastPopupOpen) {
        selected.clear();
        
        delete[] screenshotPixels;
        screenshotPixels = nullptr;

        captured = false;
        
        glDeleteTextures(1, &screenshotTexture);
        screenshotTexture = 0;
    }

    lastPopupOpen = popupOpen;

    if (popupOpen)
        return;

    // Highlight selected plots
    bool currentSelected = false;
    for (const auto& sel : selected) {
        if (plot == sel) {
            highlightCurrentPlot(IM_COL32(128, 128, 128, 64));

            currentSelected = true;

            // Called on every plot, so only draw one rect per plot
            break;
        }
    }

    bool shift = ImGui::IsKeyDown(ImGuiKey_LeftShift) || ImGui::IsKeyDown(ImGuiKey_RightShift);
    bool takeScreenshot = false;

    if (ImPlot::IsPlotFrameHovered()) {
        if (currentSelected)
            highlightCurrentPlot(IM_COL32(128, 128, 128, 64));
        else
            highlightCurrentPlot(IM_COL32(128, 128, 128, 128));

        if (ImGui::IsMouseClicked(0)) {
            if (shift) {
                if (!currentSelected) {
                    selected.push_back(plot);
                } else {
                    selected.erase(std::remove(selected.begin(), selected.end(), plot), selected.end());
                }
            } else {
                selected.clear();
                selected.push_back(plot);

                takeScreenshot = true;
            }
        }
    }

    if (takeScreenshot || (lastShiftDown && !shift) && !selected.empty()) {
        ImGui::OpenPopup("Save Screenshot");
        popupOpen    = true;
        clickedPlot  = plot;
        captured     = false;
        captureReady = false;

        clickedPos = ImGui::GetMousePos();
    }

    lastShiftDown = shift;
}
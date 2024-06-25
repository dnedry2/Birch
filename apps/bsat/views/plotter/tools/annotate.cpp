#include "plotter.hpp"
#include "logger.hpp"

#include <string>

const static std::string annotateToolHelp = "Left click on the plot to add an annotation.";

Plotter::AnnotateTool::AnnotateTool(ImPlotLimits* limits, AnnotationDisplay_Widget* widget)
                      : Tool("Annotate", annotateToolHelp.c_str(), true, textures["bicons/IconAnnotate.svg"], textures["bicons/IconAnnotate.svg"], textures["bicons/IconAnnotate.svg"], ImVec2(0, -3), ImVec2(0, -3), ImGuiKey_A),
                        limits(limits), widget(widget) { }

void Plotter::AnnotateTool::on_selection(ImPlotInputMap *controls, ImPlotFlags *flags)
{
    *flags = ImPlotFlags_Crosshairs;
}

void Plotter::AnnotateTool::on_release(Plot* plot) {
    if (ImPlot::IsPlotHovered())
    {
        auto pos = ImPlot::GetPlotMousePos();

        double posArr[] = { pos.x, pos.y };
        double boundsArr[] = { limits->X.Min, limits->X.Max };

        const char txt[] = "Enter text";
        Annotation *note = new Annotation(txt, posArr, boundsArr, 0, (*gNextID)++, true);

        plot->SelectedField()->Annotations.push_back(note);

        widget->AddAnnotation(note);
        widget->Update();
    }
}
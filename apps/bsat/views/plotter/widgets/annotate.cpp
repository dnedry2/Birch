#include "plotter.hpp"

Plotter::AnnotationDisplay_Widget::AnnotationDisplay_Widget(int id, GLuint icon, Plotter* plotter) : Widget(id, icon, std::string("Annotations"))
{
    this->ToolID = "Annotate";
    this->plotter = plotter;
}
void Plotter::AnnotationDisplay_Widget::Render()
{
    // TODO: Completely rewrite this; I didn't know how to use ImGui when I originally wrote it.
    if (beginWidget(this)) {

        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() - 10);
        ImGui::ListBox("##annot_list", &currentSelection, textList, itemCount, maxDisp);

        if (currentSelection != -1)
            plotter->AnnotateJump();

        currentSelection = -1;
    }
    endWidget();
}
void Plotter::AnnotationDisplay_Widget::Update()
{
    itemCount = annotations.size();

    free(textList);

    textList = (char **)malloc(sizeof(char *) * itemCount);

    int i = 0;
    for (Annotation *annot : annotations)
        textList[i++] = annot->Text;

    currentSelection = -1;
}
Annotation* Plotter::AnnotationDisplay_Widget::Get_Selection()
{
    return annotations.at(currentSelection);
}
void Plotter::AnnotationDisplay_Widget::AddAnnotation(Annotation *annotation)
{
    annotations.push_back(annotation);
}
Plotter::AnnotationDisplay_Widget::~AnnotationDisplay_Widget()
{
    free(textList);
}
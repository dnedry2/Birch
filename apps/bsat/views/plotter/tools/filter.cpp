#include "plotter.hpp"
#include "logger.hpp"

#include <string>

using namespace Birch;

static inline bool between(double x, double low, double high) {
    return x >= low && x <= high;
}

static std::string filterToolHelp = 
    "Left click and drag to select a region to filter.\n"
    "Hold shift to select multiple regions.\n"
    "Hold ctrl and select in time to use a matched filter.";

Plotter::FilterTool::FilterTool(FilterStack* stack, Session* session, Plotter* plotter)
                    : Tool("Filter", filterToolHelp.c_str(), true, textures["bicons/IconFilter.svg"], textures["bicons/CursorFilter.svg"], textures["bicons/CursorFilter.svg"], ImVec2(7.5, 14), ImVec2(7.5, 14), ImGuiKey_E),
                      stack(stack), session(session), plotter(plotter) { }

void Plotter::FilterTool::on_selection(ImPlotInputMap *controls, ImPlotFlags *flags)
{
    setLimsNorm(controls, flags);
    this->map   = controls;
    this->flags = flags;
}

void Plotter::FilterTool::setLimsNorm(ImPlotInputMap* controls, ImPlotFlags* flags) {
    controls->VerticalMod = ImGuiModFlags_Alt;
    controls->HorizontalMod = ImGuiModFlags_None;
    controls->BoxSelectMod = ImGuiModFlags_Alt;

    controls->QueryButton = ImGuiMouseButton_Left;
    controls->QueryMod = ImGuiModFlags_None;

    controls->BoxSelectButton = ImGuiMouseButton_Right;

    controls->PanButton = ImGuiMouseButton_Middle;

    *flags = ImPlotFlags_Query | ImPlotFlags_Crosshairs;
}
void Plotter::FilterTool::setLimsBox(ImPlotInputMap* controls, ImPlotFlags* flags) {
    controls->VerticalMod = ImGuiModFlags_Shift;
    controls->HorizontalMod = ImGuiModFlags_Shift;
    controls->BoxSelectMod = ImGuiModFlags_Alt;

    controls->QueryButton = ImGuiMouseButton_Left;
    controls->QueryMod = ImGuiModFlags_None;

    controls->BoxSelectButton = ImGuiMouseButton_Right;

    controls->PanButton = ImGuiMouseButton_Middle;

    *flags = ImPlotFlags_Query | ImPlotFlags_Crosshairs;
}
void Plotter::FilterTool::setLimsX(ImPlotInputMap* controls, ImPlotFlags* flags) {
    controls->VerticalMod = ImGuiModFlags_None;
    controls->HorizontalMod = ImGuiModFlags_Alt;
    controls->BoxSelectMod = ImGuiModFlags_Alt;

    controls->QueryButton = ImGuiMouseButton_Left;
    controls->QueryMod = ImGuiModFlags_None;

    controls->BoxSelectButton = ImGuiMouseButton_Right;

    controls->PanButton = ImGuiMouseButton_Middle;

    *flags = ImPlotFlags_Query | ImPlotFlags_Crosshairs;
}


void Plotter::FilterTool::on_update(Plot* plot) {
    if (ImGui::IsKeyDown(ImGuiKey_LeftShift) || ImGui::IsKeyDown(ImGuiKey_RightShift))
    {
        for (auto filter : *stack->Get_Filters())
        {
            if (filter->IsFieldReference(plot->SelectedFieldContainer()->Field))
                filter->RenderPlotPreview(0);
        }
    }

    if (ImGui::IsKeyDown(ImGuiKey_LeftAlt) || ImGui::IsKeyDown(ImGuiKey_RightAlt)) {
        setLimsBox(map, flags);
    } else {
        setLimsNorm(map, flags);
    }


    if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl)) {
        auto type = ((BFile*)plot->SelectedFieldContainer()->Field->File)->Type();

        if (type == FileType::Signal)
            setLimsX(map, flags);
        else 
            setLimsBox(map, flags);
        
    } else {
        setLimsNorm(map, flags);
    }

    //setLimsNorm(map, flags);
}
void Plotter::FilterTool::on_release(Plot* plot) {
    if (ImPlot::IsPlotQueried())
    {
        auto range = ImPlot::GetPlotQuery(0);

        //bool pass  = !(ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl));
        bool pass  = true;
        bool multi = ImGui::IsKeyDown(ImGuiKey_LeftShift) || ImGui::IsKeyDown(ImGuiKey_RightShift);

        bool pdw = false; // ImGui::IsKeyDown(ImGuiKey_LeftAlt) || ImGui::IsKeyDown(ImGuiKey_RightAlt);
        //bool match = ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl);
        bool match = (ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl));

        if (!multi) {
            // Clear only filters on this field
            auto filters = stack->Get_Filters();

            auto field = plot->SelectedField();

            if (field == nullptr) {
                DispError("Plotter::FilterTool::on_release", "No field selected! (This should be impossible, bad things are about to happen)");
                return;
            }

            filters->erase(std::remove_if(
                filters->begin(), filters->end(),
                [=](Filter* filter) { 
                    return filter->IsFieldReference(field);
                }), filters->end());
        }

        if (match) {
            auto field = plot->SelectedField();
            auto file_ = (BFile*)field->File;
            if (file_->Type() == FileType::Signal)
            {
                auto file = (SignalFile*)file_;
                auto sel = file->FetchPortion(Birch::Timespan(range.X.Min, range.X.Max));

                stack->AddFilter(new MatchFilter(file->Data(), sel->RI(), sel->RQ(), *sel->ElementCount(), file->Filename()));
                plotter->ForceNextReload();
                delete sel;
            } else {
                auto file = (TBDFile*)file_;
                auto sig  = file->AssocSig();

                if (sig == nullptr) {
                    DispWarning("Plotter::FilterTool::on_release", "No associated signal found");
                } else {
                    // get pdw in time range
                    auto pd    = file->Field("PD");
                    auto peaks = sig->Field("Peaks");

                    if (pd != nullptr) {
                        // TODO: binary search
                        unsigned long i = 0;
                        bool found = false;
                        for (i = 0; i < *pd->ElementCount; i++) {
                            if (pd->Timing[i] >= range.X.Min) {
                                if (pd->Timing[i] > range.X.Max)
                                    break;

                                if (range.Y.Min <= field->Data[i] && field->Data[i] <= range.Y.Max) {
                                    found = true;

                                    auto sel = sig->FetchPortion(Birch::Timespan(pd->Timing[i], pd->Timing[i] + pd->Data[i]));

                                    stack->AddFilter(new MatchFilter(sig->Data(), sel->RI(), sel->RQ(), *sel->ElementCount(), sig->Filename()));
                                    stack->AddFilter(new TBDMatchFilter(&peaks->Data, &pd->Data, &peaks->ElementCount, strdup(sig->Filename())));
                                    plotter->ForceNextReload();
                                    delete sel;

                                    break; // remove later
                                }
                            }
                        }

                        if (!found) {
                            DispWarning("Plotter::FilterTool::on_release", "No record found in selection!");
                        } else {

                        }

                    } else {
                        DispError("Plotter::FilterTool::on_release", "No PD field found in file %s", file->Filename());
                    }
                }
            }
        } else if (!pdw) {
            auto field = plot->SelectedField();
            auto file_ = (BFile*)field->File;
            if (file_->Type() == FileType::Signal)
            {
                if (!strncmp(field->Name, "Spectra", 7)) {
                    auto file = (SignalFile*)file_;

                    auto in = Span<double>(range.Y.Min, range.Y.Max);

                    if (in.Start > in.End) {
                        auto t = in.Start;
                        in.Start = in.End;
                        in.End = t;
                    }

                    double c = (in.End - in.Start) / 2 + in.Start;
                    double w = (in.End - in.Start);

                    c *= 1000000;
                    w *= 1000000;

                    // spectra y axis needs fixed

                    stack->AddFilter(new FIRFilter(file->Data(), field, static_cast<Birch::PluginIQGetter*>(file->IOPlugin())->SampleRate(), w, c, 1024, file->Filename(), pass));
                    plotter->ForceNextReload();
                }
            }
            else {
                stack->AddFilter(new TBDFilter(&plot->SelectedField()->Data, plot->SelectedField()->Name, range.Y.Min, range.Y.Max, pass));
            }
        }
        else { // automatic filtering of tbd fields via box selection. matched filter is better, probably dont need this
        /*
            auto file_ = (Bfile *)plot->GetSelectedField()->File;
            if (file_->Get_Type() == SignalType::PDW)
            {
                auto file = (BfilePDW *)file_;
                auto reference = file->Get_6k();

                // The field the selection was done on
                TBDField *ref = GetField(reference, plot->GetSelectedField()->Name);
                TBDField *toa = GetField(reference, strdup("PTOA"));

                // TODO: Let user choose fields
                const int   fCount   = 3;
                const char* fNames[] = {"PD", "RF", "Intra"};

                TBDField  *tbdfields[fCount];
                PlotField *plotFields[fCount];
                double    lims[fCount][2];

                for (int i = 0; i < fCount; i++) {
                    tbdfields[i]  = GetField(reference, fNames[i]);
                    plotFields[i] = file->Get_Field(fNames[i]);

                    lims[i][0] = tbdfields[i]->MaxVal;
                    lims[i][1] = tbdfields[i]->MaxVal;
                }

                // calculate bands for each field
                for (int i = 0; i < toa->TotalElementCount; i++)
                {
                    if (between(toa->Elements[i], range.X.Min, range.X.Max) && 
                        between(ref->Elements[i], range.Y.Min, range.Y.Max))
                    {
                        for (int j = 0; j < fCount; j++)
                        {
                            lims[j][0] = min_(tbdfields[j]->Elements[i], lims[j][0]);
                            lims[j][1] = max_(tbdfields[j]->Elements[i], lims[j][1]);
                        }
                    }
                }

                const float tolerance = 0.25f; // TODO: Add tolerance setting

                for (int i = 0; i < fCount; i++) {
                    double range = lims[i][1] - lims[i][0];
                    lims[i][0] -= range * tolerance;
                    lims[i][1] += range * tolerance;

                    stack->AddFilter(new TBDFilter(&plotFields[i]->Data, plotFields[i]->Name, lims[i][0], lims[i][1], pass));
                }
            }
            else
            {
                DispWarning("Plotter::FilterTool::on_release", "%s", "Can't add a PDW filter to IQ data!!");
            }
            */
        }

        session->RefreshFields();
        session->ReloadFields();
        ImPlot::HidePlotQuery();
    }
}

void Plotter::FilterTool::on_middle_click(Plot* plot) {
    // Remove all filters on this field
    if (ImPlot::IsPlotHovered()) {
        auto field    = plot->SelectedField();
        auto& filters = *stack->Get_Filters();

        std::vector<int> toRemove;

        const int size = filters.size();
        for (int i = 0; i < size; i++)
            if (filters[i]->IsFieldReference(field))
                toRemove.push_back(i);

        for (int i = toRemove.size() - 1; i >= 0; i--) {
            delete filters[toRemove[i]];
            filters.erase(filters.begin() + toRemove[i]);
        }

        session->ReloadFields();
    }
}
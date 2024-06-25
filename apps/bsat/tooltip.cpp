#include "tooltip.hpp"

#include <cstdio>

#include "imgui.h"

void Tooltip::Render(const char* text, float delay) {
    Render(this, text, delay);
}
void Tooltip::Render(void (*func)(void* data), void* data, float delay) {
    Render(this, func, data, delay);
}
void Tooltip::Render(const void* id, const char* text, float delay) {
    Render(this, [](void* data) { ImGui::Text("%s", (const char*)data); }, (void*)text, delay);
}
void Tooltip::Render(const void* id, void (*func)(void* data), void* data, float delay) {
    auto lb = watches.lower_bound(id);
    if(!(lb != watches.end() && !(watches.key_comp()(id, lb->first))))
        watches[id] = Stopwatch();

    Stopwatch& timer = watches[id];

    if (ImGui::IsItemHovered()) {
        if (timer.Now() < delay)
            return;
        
        if (ImGui::BeginTooltip()) {
            func(data);
            ImGui::EndTooltip();
        }
    } else {
        timer.Reset();
    }
}
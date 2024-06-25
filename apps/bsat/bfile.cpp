#define IMGUI_DEFINE_MATH_OPERATORS
#include "bfile.hpp"
#include "logger.hpp"

using namespace Birch;

BFile* BFileFactory(const char* path, Plugin* plugin, ColorStack* colormap, FilterStack* filterstack) {
    if (plugin->Type == PluginType::IQGetter) {
        PluginIQGetter* getter = static_cast<PluginIQGetter*>(plugin);
        
        if (!getter->Open(path)) {
            DispError("BFileFactory", "Failed to open IQ file!");
            return nullptr;
        }

        return new SignalFile(getter, colormap, filterstack);
    }
    else if (plugin->Type == PluginType::TBDGetter) {
        PluginTBDGetter* getter = static_cast<PluginTBDGetter*>(plugin);
        
        if (!getter->Open(path)) {
            DispError("BFileFactory", "Failed to open TBD file!");
            return nullptr;
        }

        return new TBDFile(getter, colormap, filterstack);
        
    }

    DispError("BFileFactory", "Plugin was not of type 'IQGetter' or 'TBDGetter'");
    return nullptr;
}
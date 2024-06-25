#ifndef __BIRCH_PLUGIN__
#define __BIRCH_PLUGIN__

namespace Birch {
struct BMath;
};

enum class PluginType { IQGetter, TBDGetter, IQProcessor, TBDProcessor, Null};

class Plugin {
public:
    virtual ~Plugin() { }
    PluginType Type = PluginType::Null;

    // TODO: This is a hack...
    // These functions need to be placed in a seperate shared library

    // Plugins may call this to convert an SVG into a GLuint
    unsigned (*SubmitSVG)(const char*, unsigned, unsigned) = nullptr;

    // Access math functions
    Birch::BMath* Math = nullptr;
};

extern "C" typedef struct {
    const char* Name;
    PluginType Type;
    Plugin* (*Instance)();
    void    (*Destroy)(Plugin*);

    void*   _handle;
} PluginInterface;

extern "C" PluginInterface Interface();

#endif

#ifdef PLUGIN_CLIENT
#include <vector>
#include <string>

void DestroyInterface(PluginInterface* interface);
PluginInterface LoadPlugin(const std::string& path);
std::vector<PluginInterface> LoadPlugins(std::string dir);
#endif

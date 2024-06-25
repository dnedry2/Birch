#include <vector>
#include <string>
#include <filesystem>
#include <iostream>

#define PLUGIN_CLIENT
#include "include/plugin.h"

using std::cerr;
using std::endl;
using std::vector;

#if defined __unix || defined __APPLE__
    #include <dlfcn.h>
#elif defined _WIN32 || defined _WIN64
    #include <Windows.h>
#endif

void DestroyInterface(PluginInterface* plugin) {
#if defined __unix || defined __APPLE__
    dlclose(plugin->_handle);
#elif defined _WIN32 || defined _WIN64
    FreeLibrary((HMODULE)plugin->_handle);
#endif
}

PluginInterface LoadPlugin(const std::string& path) {
    using namespace std;
    typedef PluginInterface (*Interface)();
    Interface plug = nullptr;

#if defined __unix || defined __APPLE__
    void* handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_GLOBAL);

    if (!handle) {
        cerr << path << " - Cannot open library: " << dlerror() << endl;
        throw("Failed to load! (IO)");
    }

    dlerror();
    plug = (Interface)dlsym(handle, "Interface");

    const char *dlsym_error = dlerror();
    if (dlsym_error) {
        dlclose(handle);
        //cerr << "Cannot load symbols: " << dlsym_error << endl;
        throw("Failed to load!");
    }
#elif defined _WIN32 || defined _WIN64
    HMODULE handle = LoadLibraryA(path.c_str());
    if (handle == NULL) {
        cerr << "Cannot open library." << endl;
        throw("Failed to load! (IO)");
    }

    FARPROC plugp = GetProcAddress(handle,"Interface");

    if (plugp == NULL) {
        cerr << "Cannot load symbols." << endl;
        throw("Failed to load!");
    }

    plug = (Interface)plugp;
#endif

    PluginInterface out = plug();
    out._handle = handle;

    return out;
}

std::vector<PluginInterface> LoadPlugins(std::string dir) {
    vector<PluginInterface> out = vector<PluginInterface>();

    for (const auto& file : std::filesystem::directory_iterator(dir)) {
        try {
            if (std::filesystem::is_directory(file.path()))
                continue;
            
            out.push_back(LoadPlugin(file.path().string()));
        } catch (...) { }
    }

    return out;
}
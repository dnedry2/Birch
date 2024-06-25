#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#ifndef M_PI_4
#define M_PI_4 (M_PI / 4)
#endif

#ifndef M_PI_2
#define M_PI_2 (M_PI / 2)
#endif

#ifndef M_4_PI
#define M_4_PI (M_PI * 4)
#endif

#ifndef M_2_PI
#define M_2_PI (M_PI * 2)
#endif

#ifndef deg
#define deg(x) ((x) * (180 / M_PI))
#endif

#ifndef t_ImTexID
#define t_ImTexID(tex) ((void*)(intptr_t)(tex))
#endif

#ifndef min_
#define min_(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef max_
#define max_(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifdef _WIN64
#define strdup _strdup
#endif

typedef unsigned char      uchar;
typedef unsigned short     ushort;
typedef unsigned int       uint;
typedef unsigned long      ulong;
typedef unsigned long long ullong;
typedef long long          llong;

#include <string>
#include <map>
#include <vector>

#include "include/plugin.h"

struct MainThreadTask;

class Gradient;
class Shader;
class Config;

extern std::map<std::string, uint> textures;
extern std::map<std::string, std::string> markers;
extern volatile float* gProgressBar;
extern std::vector<MainThreadTask*> gMainThreadTasks;
extern std::map<std::string, Shader*> gShaders;
extern uint signalMaxSize;
extern volatile int* gNextID;
extern std::vector<Gradient*>* gGradients;
extern Config& gConfig;
extern std::vector<PluginInterface>* gPlugins;
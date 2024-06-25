// TODO: Clean up source.cpp, defines.h, etc and move into a bsat class

#ifndef __BSAT_HPP__
#define __BSAT_HPP__

#include <string>
#include <map>
#include <vector>

#include "birch.h"


class Gradient;
class View;
class Session;

class BSAT {
public:
    void Run();

    BSAT();
    ~BSAT();
private:
    std::vector<Session*>              sessions;
    std::map<std::string, uint>        textures;
    std::map<std::string, std::string> markers;
    std::vector<View*>                 views;
};

#endif
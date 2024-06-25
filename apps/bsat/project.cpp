#include "project.hpp"
#include "include/birch.h"

#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace Birch;
using std::vector;
using std::string;

static vector<string> split(const string& s, char seperator)
{
    vector<string> output;
    string::size_type prev_pos = 0, pos = 0;

    while((pos = s.find(seperator, pos)) != string::npos)
    {
        string substring( s.substr(prev_pos, pos-prev_pos) );
        output.push_back(substring);
        prev_pos = ++pos;
    }

    output.push_back(s.substr(prev_pos, pos-prev_pos)); // Last word

    return output;
}

vector<ProjectFile>& Project::Files() {
    return files;
}
void Project::Save(const char* path) {
    std::ofstream out(path);

    for (const auto& file : files) {
        out << file.path << ":" << file.plugin << ":" << std::endl;
    }

    out.close();
}
string Project::Path() {
    return path;
}
string Project::Directory() {
    return fs::path(path).parent_path();
}

Project::Project() {
    files = vector<ProjectFile>();
}
Project::Project(const char* path) {
    files = vector<ProjectFile>();
    this->path = path;

    std::ifstream in(path);

    string line;
    while (std::getline(in, line)) {
        auto parts = split(line, ':');
        auto fields = split(parts[2], ',');
        
        files.push_back({parts[0], parts[1], fields});
    }

    in.close();
}
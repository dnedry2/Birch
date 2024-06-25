#ifndef _BIRCH_PROJECT_H_
#define _BIRCH_PROJECT_H_

#include <vector>
#include <string>

struct ProjectFile
{
    std::string path;
    std::string plugin;
    std::vector<std::string> openFields;
    // Annotations and such later
};


class Project {
public:
    std::vector<ProjectFile>& Files();
    void Save(const char* path);
    std::string Path();
    std::string Directory();

    Project();
    Project(const char* path);
private:
    std::vector<ProjectFile> files;
    std::string path;
};

#endif
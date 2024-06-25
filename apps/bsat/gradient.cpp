#include "gradient.hpp"
#include "logger.hpp"

#include <cstdio>
#include <cstring>

Gradient::Gradient(const unsigned int* stops, const int stopCount, const char* name, const char* catagory, const int id) : StopCount(stopCount), CatagoryID(id) {
    Name = new char[strlen(name) + 1];
    strcpy((char*)Name, name);

    Catagory = new char[strlen(catagory) + 1];
    strcpy((char*)Catagory, catagory);

    Stops = new unsigned int[stopCount];
    memcpy((void*)Stops, stops, stopCount * sizeof(unsigned int));
}
Gradient::~Gradient() {
    delete[] Stops;
    delete[] Name;
    delete[] Catagory;
}

Gradient* Gradient::Load(const char* data) {
    char name[256];
    char catagory[256];
    int  catagoryID;
    int  stopCount;

    int cond = 0;
    const char* ptr = data;

    cond = sscanf(ptr, "Name: %s\n", name);
    ptr  = strchr(ptr, '\n');
    if (ptr == nullptr) goto failure;
    if (cond != 1)      goto failure;

    cond = sscanf(++ptr, "Catagory: %s\n", catagory);
    ptr  = strchr(ptr, '\n');
    if (ptr == nullptr) goto failure;
    if (cond != 1)      goto failure;

    cond = sscanf(++ptr, "CatagoryID: %d\n", &catagoryID);
    ptr  = strchr(ptr, '\n');
    if (ptr == nullptr) goto failure;
    if (cond != 1)      goto failure;

    cond = sscanf(++ptr, "Stop Count: %d\n", &stopCount);
    ptr  = strchr(ptr, '\n');
    if (ptr == nullptr) goto failure;
    if (cond != 1)      goto failure;

    goto success;

    failure:
    DispWarning("Gradient::Load", "Failed to load gradient");
    return nullptr;

    success:

    unsigned int* stops = new unsigned int[stopCount];

    for (int i = 0; i < stopCount; i++) {
        sscanf(++ptr, "%x\n", &stops[i]);
        ptr = strchr(ptr, '\n');

        if (ptr == nullptr) {
            DispWarning("Gradient::Load", "Failed to load gradient");
            delete[] stops;
            return nullptr;
        }
    }

    return new Gradient(stops, stopCount, name, catagory, catagoryID);
}

void Gradient::Save(const char* filepath) {
    FILE* file = fopen(filepath, "w");

    if (file == nullptr) {
        DispError("Gradient::Save", "Failed to open file %s", filepath);
        return;
    }

    fprintf(file, "Name: %s\n", Name);
    fprintf(file, "Catagory: %s\n", Catagory);
    fprintf(file, "CatagoryID: %d\n", CatagoryID);
    fprintf(file, "Stop Count: %d\n", StopCount);

    for (int i = 0; i < StopCount; i++)
        fprintf(file, "%x\n", Stops[i]);

    fclose(file);
}
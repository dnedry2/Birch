#ifndef _GRADIENT_
#define _GRADIENT_

struct Gradient {
    const unsigned int* Stops;
    const int StopCount;
    const char* Name;
    const char* Catagory;
    const int CatagoryID;

    Gradient(const unsigned int* stops, const int stopCount, const char* name, const char* catagory, const int id);
    ~Gradient();

    void Save(const char* filename);
    static Gradient* Load(const char* data);
};

#endif
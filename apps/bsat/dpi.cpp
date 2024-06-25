#include "dpi.hpp"
#include "logger.hpp"

void GetDPI(int screen, float* dpi, float* scale) {
    //use 96dpi as default if system is not recognised
    int sysDefault = 96;

    #ifdef __APPLE__
        sysDefault = 72;
    #elif !defined(_WIN32)
        DispWarning("GetDPI", "%s", "Could not determine system base DPI. Defaulting to 96.\n");
    #endif

    *scale = *dpi / sysDefault;
}

DPIAwareSizeConverter::DPIAwareSizeConverter(int display) {
    GetDPI(display, &dpi, &scale);
}

int DPIAwareSizeConverter::MMToPx(float mm) {
    return InchToPx(mm / 25.4);
}

int DPIAwareSizeConverter::InchToPx(float in) {
    //printf("dpi: %f\nscale: %f\n", dpi, scale);
    return in * dpi / scale;
}

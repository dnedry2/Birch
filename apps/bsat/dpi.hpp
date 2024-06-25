/********************************
* dpi - handles dpi awareness
*
* Author: Levi Miller
* Date created: 12/12/2020
*********************************/

#ifndef _DPI_
#define _DPI_

//#include <SDL2/SDL.h>
#include <GLFW/glfw3.h>

void GetDPI(int screen, float* dpi, float* scale);

class DPIAwareSizeConverter {
public:
    DPIAwareSizeConverter(int display = 0);

    int MMToPx(float mm); 
    int InchToPx(float in);
private:
    float dpi;
    float scale;
};

#endif
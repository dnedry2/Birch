#include <cmath>
#include "windowImpls.hpp"

Bartlett::Bartlett() : WindowFunc("Bartlett") { }
double* Bartlett::Build(int len) {
    double* win = new double[len];

    for (int i = 0; i < len; i++)
        win[i] = (2.0 / (len - 1)) * (((len - 1) / 2.0) - fabs(i - ((len - 1) / 2.0)));

    return win;
}
BartlettHann::BartlettHann() : WindowFunc("Bartlett-Hann") { }
double* BartlettHann::Build(int len) {
    double* win = new double[len];

    for (int i = 0; i < len; i++)
        win[i] = 0.62 - 0.48 * fabs((double)i / len - 0.5) + 0.38 * cos(2 * M_PI * ((double)i / len - 0.5));

    for (int i = 0; i < len; i++)
        win[i] = 0.62 - 0.48 * fabs(((double)i / (len - 1)) - 0.5) - 0.38 * cos((2.0 * M_PI * i) / (len - 1));

    return win;
}
Blackman::Blackman() : WindowFunc("Blackman") { }
double* Blackman::Build(int len) {
    double* win = new double[len];
    
    for (int i = 0; i < len; i++)
        win[i] = 0.42659 - 0.49656 * cos((2 * M_PI * i) / (len - 1)) + 0.076849 * cos((4 * M_PI * i) / (len - 1));

    return win;
}
BlackmanHarris::BlackmanHarris() : WindowFunc("Blackman-Harris") { }
double* BlackmanHarris::Build(int len) {
    double* win = new double[len];

    for (int i = 0; i < len; i++)
        win[i] = 0.35875 - 0.48829 * cos((2 * M_PI * i) / (len - 1)) + 0.14128 * cos((4 * M_PI * i) / (len - 1) - 0.01168 * cos((6 * M_PI * i) / (len - 1)));

    return win;
}
BlackmanNuttall::BlackmanNuttall() : WindowFunc("Blackman-Nuttall") { }
double* BlackmanNuttall::Build(int len) {
    double* win = new double[len];

    for (int i = 0; i < len; i++)
        win[i] = 0.3635819 - 0.4891775 * cos((2 * M_PI * i) / (len - 1)) + 0.1365995 * cos((4 * M_PI * i) / (len - 1)) - 0.0106411 * cos((6 * M_PI * i) / (len - 1));

    return win;
}
Bohman::Bohman() : WindowFunc("Bohman") { }
double* Bohman::Build(int len) {
    double* win = new double[len];

    double m = (len - 1) / 2;

    for (int i = 0; i < len; i++)
        win[i] = (1 - fabs(i / m - 1)) * cos(M_PI * fabs(i / m - 1)) + 1 / M_PI * sin(M_PI * fabs(i / m - 1));

    return win;
}
FlatTop::FlatTop() : WindowFunc("Flat-Top") { }
double* FlatTop::Build(int len) {
    double* win = new double[len];

    for (int i = 0; i < len; i++)
        win[i] = 0.21557895 - 0.41663158 * cos((2 * M_PI * i) / (len - 1)) + 0.277263158 * cos((4 * M_PI * i) / (len - 1)) - 0.083578947 * cos((6 * M_PI * i) / (len - 1)) + 0.006947368 * cos((8 * M_PI * i) / (len - 1));

    return win;
}
Gaussian2::Gaussian2() : WindowFunc("Gaussian 2") { }
double* Gaussian2::Build(int len) {
    double* win = new double[len];

    double a = 2;
    double m = (len - 1) / 2.0;

    for (int i = 0; i < len; i++)
        win[i] = exp(-0.5 * pow((i - m) / (a * m), 2));

    return win;
}
Gaussian4::Gaussian4() : WindowFunc("Gaussian 4") { }
double* Gaussian4::Build(int len) {
    double* win = new double[len];

    double a = 4;
    double m = (len - 1) / 2.0;

    for (int i = 0; i < len; i++)
        win[i] = exp(-0.5 * pow((i - m) / (a * m), 2));

    return win;
}
Gaussian6::Gaussian6() : WindowFunc("Gaussian 6") { }
double* Gaussian6::Build(int len) {
    double* win = new double[len];

    double a = 6;
    double m = (len - 1) / 2.0;

    for (int i = 0; i < len; i++)
        win[i] = exp(-0.5 * pow((i - m) / (a * m), 2));

    return win;
}
Hamming::Hamming() : WindowFunc("Hamming") { }
double* Hamming::Build(int len) {
    double* win = new double[len];

    for (int i = 0; i < len; i++)
        win[i] = 0.53836 - 0.46164 * cos((2 * M_PI * i) / (len - 1));

    return win;
}
Hann::Hann() : WindowFunc("Hann") { }
double* Hann::Build(int len) {
    double* win = new double[len];

    for (int i = 0; i < len; i++)
        win[i] = 0.5 * (1 - cos((2 * M_PI * i) / (len - 1)));

    return win;
}
HannPoisson3::HannPoisson3() : WindowFunc("Hann-Poisson 0.3") { }
double* HannPoisson3::Build(int len) {
    double* win = new double[len];

    double a = 0.3;
    double m = (len - 1) / 2.0;

    for (int i = 0; i < len; i++)
        win[i] = 0.5 * (1 - cos((M_PI * i) / m)) * exp(-a * (fabs(i - m) / m));

    return win;
}
HannPoisson5::HannPoisson5() : WindowFunc("Hann-Poisson 0.5") { }
double* HannPoisson5::Build(int len) {
    double* win = new double[len];

    double a = 0.5;
    double m = (len - 1) / 2.0;

    for (int i = 0; i < len; i++)
        win[i] = 0.5 * (1 - cos((M_PI * i) / m)) * exp(-a * (fabs(i - m) / m));

    return win;
}
HannPoisson7::HannPoisson7() : WindowFunc("Hann-Poisson 0.7") { }
double* HannPoisson7::Build(int len) {
    double* win = new double[len];

    double a = 0.7;
    double m = (len - 1) / 2.0;

    for (int i = 0; i < len; i++)
        win[i] = 0.5 * (1 - cos((M_PI * i) / m)) * exp(-a * (fabs(i - m) / m));

    return win;
}
KaiserBessel::KaiserBessel() : WindowFunc("Kaiser-Bessel") { }
double* KaiserBessel::Build(int len) {
    double* win = new double[len];

    for (int i = 0; i < len; i++)
        win[i] = 0.402 - 0.498 * cos((2 * M_PI * i) / (len - 1)) + 0.098 * cos((4 * M_PI * i) / (len - 1)) - 0.001 * cos((6 * M_PI * i) / (len - 1)); 

    return win;
}
Lanczos::Lanczos() : WindowFunc("Lanczos") { }
double* Lanczos::Build(int len) {
    double* win = new double[len];
    
    for (int i = 0; i < len; i++)
        win[i] = sin(M_PI * ((2.0 * i) / (len - 1) - 1)) / (M_PI * ((2.0 * i) / (len - 1) - 1));

    return win;
}
Nuttall::Nuttall() : WindowFunc("Nuttall") { }
double* Nuttall::Build(int len) {
    double* win = new double[len];

    for (int i = 0; i < len; i++)
        win[i] = 0.355768 - 0.487396 * cos((2 * M_PI * i) / (len - 1)) + 0.144232 * cos((4 * M_PI * i) / (len - 1)) - 0.012604 * cos((6 * M_PI * i) / (len - 1));

    return win;
}
Poisson2::Poisson2() : WindowFunc("Poisson 0.2") { }
double* Poisson2::Build(int len) {
    double* win = new double[len];

    double a = 0.2;
    double m = (len - 1) / 2.0;

    for (int i = 0; i < len; i++)
        win[i] = exp(-a * (fabs(i - m) / m));

    return win;
}
Poisson5::Poisson5() : WindowFunc("Poisson 0.5") { }
double* Poisson5::Build(int len) {
    double* win = new double[len];

    double a = 0.5;
    double m = (len - 1) / 2.0;

    for (int i = 0; i < len; i++)
        win[i] = exp(-a * (fabs(i - m) / m));

    return win;
}
Poisson8::Poisson8() : WindowFunc("Poisson 0.8") { }
double* Poisson8::Build(int len) {
    double* win = new double[len];

    double a = 0.8;
    double m = (len - 1) / 2.0;

    for (int i = 0; i < len; i++)
        win[i] = exp(-a * (fabs(i - m) / m));

    return win;
}
Rectangle::Rectangle() : WindowFunc("Rectangle") { }
double* Rectangle::Build(int len) {
    double* win = new double[len];

    for (int i = 0; i < len; i++)
        win[i] = 1;

    return win;
}
Tukey::Tukey() : WindowFunc("Tukey") { }
double* Tukey::Build(int len) {
    double* win = new double[len];

    double r = 0.5;
    double m = (len - 1) / 2.0;
    int off = (r / 2.0) * len;

    for (int i = 0; i < len; i++)
        win[i] = 1;

    for (int i = 0; i < off; i++)
        win[i] = 0.5 * (1 + cos((M_PI * (fabs(i - m) - r * m)) / ((1 - r) * m)));

    for (int i = len - off; i < len; i++)
        win[i] = 0.5 * (1 + cos((M_PI * (fabs(i - m) - r * m)) / ((1 - r) * m)));

    return win;
}
Welch::Welch() : WindowFunc("Welch") { }
double* Welch::Build(int len) {
    double* win = new double[len];

    for (int i = 0; i < len; i++)
        win[i] = 1 - pow((i - ((len - 1) / 2.0)) / ((len + 1) / 2.0), 2);

    return win;
}

const std::vector<WindowFunc*>* GetWindows() {
    static std::vector<WindowFunc*>* winList = nullptr;

    if (winList == nullptr) {
        winList = new std::vector<WindowFunc*>();

        winList->push_back(new Bartlett());
        winList->push_back(new BartlettHann());
        winList->push_back(new Blackman());
        winList->push_back(new BlackmanHarris());
        winList->push_back(new BlackmanNuttall());
        winList->push_back(new Bohman());
        winList->push_back(new FlatTop());
        winList->push_back(new Gaussian2());
        winList->push_back(new Gaussian4());
        winList->push_back(new Gaussian6());
        winList->push_back(new Hamming());
        winList->push_back(new Hann());
        winList->push_back(new HannPoisson3());
        winList->push_back(new HannPoisson5());
        winList->push_back(new HannPoisson7());
        winList->push_back(new KaiserBessel());
        winList->push_back(new Lanczos());
        winList->push_back(new Nuttall());
        winList->push_back(new Poisson2());
        winList->push_back(new Poisson5());
        winList->push_back(new Poisson8());
        winList->push_back(new Rectangle());
        winList->push_back(new Tukey());
        winList->push_back(new Welch());

        #pragma omp parallel for
        for (auto it = winList->begin(); it != winList->end(); it++)
            (*it)->Init();
    }

    return winList;
}
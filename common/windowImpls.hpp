#ifndef _WINDOW_IMPL_
#define _WINDOW_IMPL_

#include "window.hpp"
#include <vector>

class Bartlett : public WindowFunc {
public:
    double* Build(int len) override;
    Bartlett();
};
class BartlettHann : public WindowFunc {
public:
    double* Build(int len) override;
    BartlettHann();
};
class Blackman : public WindowFunc {
public:
    double* Build(int len) override;
    Blackman();
};
class BlackmanHarris : public WindowFunc {
public:
    double* Build(int len) override;
    BlackmanHarris();
};
class BlackmanNuttall : public WindowFunc {
public:
    double* Build(int len) override;
    BlackmanNuttall();
};
class Bohman : public WindowFunc {
public:
    double* Build(int len) override;
    Bohman();
};
class FlatTop : public WindowFunc {
public:
    double* Build(int len) override;
    FlatTop();
};
class Gaussian2 : public WindowFunc {
public:
    double* Build(int len) override;
    Gaussian2();
};
class Gaussian4 : public WindowFunc {
public:
    double* Build(int len) override;
    Gaussian4();
};
class Gaussian6 : public WindowFunc {
public:
    double* Build(int len) override;
    Gaussian6();
};
class Hamming : public WindowFunc {
public:
    double* Build(int len) override;
    Hamming();
};
class Hann : public WindowFunc {
public:
    double* Build(int len) override;
    Hann();
};
class HannPoisson3 : public WindowFunc {
public:
    double* Build(int len) override;
    HannPoisson3();
};
class HannPoisson5 : public WindowFunc {
public:
    double* Build(int len) override;
    HannPoisson5();
};
class HannPoisson7 : public WindowFunc {
public:
    double* Build(int len) override;
    HannPoisson7();
};
class KaiserBessel : public WindowFunc {
public:
    double* Build(int len) override;
    KaiserBessel();
};
class Lanczos : public WindowFunc {
public:
    double* Build(int len) override;
    Lanczos();
};
class Nuttall : public WindowFunc {
public:
    double* Build(int len) override;
    Nuttall();
};
class Poisson2 : public WindowFunc {
public:
    double* Build(int len) override;
    Poisson2();
};
class Poisson5 : public WindowFunc {
public:
    double* Build(int len) override;
    Poisson5();
};
class Poisson8 : public WindowFunc {
public:
    double* Build(int len) override;
    Poisson8();
};
class Rectangle : public WindowFunc {
public:
    double* Build(int len) override;
    Rectangle();
};
class Tukey : public WindowFunc {
public:
    double* Build(int len) override;
    Tukey();
};
class Welch : public WindowFunc {
public:
    double* Build(int len) override;
    Welch();
};

const std::vector<WindowFunc*>* GetWindows();

#endif
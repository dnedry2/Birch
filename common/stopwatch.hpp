#ifndef _STOPWATCH_H_
#define _STOPWATCH_H_

class Stopwatch {
public:
    double StartTime();
    double Now();
    void Reset();

    Stopwatch();
private:
    double start = 0;

    double getTime();
};

#endif
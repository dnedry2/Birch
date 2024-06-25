#if defined(__APPLE__) || defined(__linux__)
    #include <sys/time.h>
    #include <sys/resource.h>
#endif

#ifdef _WIN64
    #include <windows.h>
#endif

#include <cstdlib>
#include "stopwatch.hpp"

double Stopwatch::getTime()
{
#if defined(__APPLE__) || defined(__linux__)
	struct timeval t;
    struct timezone tzp;
    gettimeofday(&t, &tzp);
    return t.tv_sec + t.tv_usec*1e-6;
#endif

#ifdef _WIN64
	SYSTEMTIME time;
	GetSystemTime(&time);
	return time.wSecond + time.wMilliseconds*1e-3;
#else
	return 0;
#endif
}

Stopwatch::Stopwatch() {
    start = getTime();
}
double Stopwatch::StartTime() {
    return start;
}
double Stopwatch::Now() {
    return getTime() - start;
}
void Stopwatch::Reset() {
    start = getTime();
}
#ifndef __BIRCH_UTIL__
#define __BIRCH_UTIL__

typedef unsigned char      uchar;
typedef unsigned int       uint;
typedef unsigned long      ulong;
typedef unsigned long long ullong;

template <typename T>
class Point {
public:
    T X = 0;
    T Y = 0;

    Point operator+(const Point& val) { return Point(X + val.X, Y + val.Y); }
    Point operator-(const Point& val) { return Point(X - val.X, Y - val.Y); }
    Point operator*(const Point& val) { return Point(X * val.X, Y * val.Y); }
    Point operator/(const Point& val) { return Point(X / val.X, Y / val.Y); }

    Point operator+=(const Point& val) { X += val.X; Y += val.Y; return *this; }
    Point operator-=(const Point& val) { X -= val.X; Y -= val.Y; return *this; }
    Point operator*=(const Point& val) { X *= val.X; Y *= val.Y; return *this; }
    Point operator/=(const Point& val) { X /= val.X; Y /= val.Y; return *this; }

    double DistanceSq(const Point& other) const { return (X - other.X) * (X - other.X) + (Y - other.Y) * (Y - other.Y);}
    double Distance(const Point& other)   const { return sqrt(DistanceSq(other)); }

    Point() { }
    Point(const Point& copy) : X(copy.X), Y(copy.Y) { }
    Point(T x, T y) : X(x), Y(y) { }
};

template <typename T>
class Point3 {
public:
    T X = 0;
    T Y = 0;
    T Z = 0;

    Point3 operator+(const Point3& val) { return Point3(X + val.X, Y + val.Y, Z + val.Z); }
    Point3 operator-(const Point3& val) { return Point3(X - val.X, Y - val.Y, Z - val.Z); }
    Point3 operator*(const Point3& val) { return Point3(X * val.X, Y * val.Y, Z * val.Z); }
    Point3 operator/(const Point3& val) { return Point3(X / val.X, Y / val.Y, Z / val.Z); }

    Point3 operator+=(const Point3& val) { X += val.X; Y += val.Y; Z += val.Z; return *this; }
    Point3 operator-=(const Point3& val) { X -= val.X; Y -= val.Y; Z -= val.Z; return *this; }
    Point3 operator*=(const Point3& val) { X *= val.X; Y *= val.Y; Z *= val.Z; return *this; }
    Point3 operator/=(const Point3& val) { X /= val.X; Y /= val.Y; Z /= val.Z; return *this; }

    double DistanceSq(const Point3& other) const { return (X - other.X) * (X - other.X) + (Y - other.Y) * (Y - other.Y) + (Z - other.Z) * (Z - other.Z); }
    double Distance(const Point3& other)   const { return sqrt(DistanceSq(other)); }

    Point3() { }
    Point3(const Point3& copy) : X(copy.X), Y(copy.Y), Z(copy.Z) { }
    Point3(T x, T y, T z) : X(x), Y(y), Z(z) { }
};

template <typename T>
class Span {
public:
    T Start = 0;
    T End   = 0;

    T Length() const { return End - Start; }
    T Center() const { return Start + Length() * 0.5; }
    bool Overlaps(Span<T> other) const { return other.Start < End && other.End > Start; }
    void Clamp(Span<T> limit) {
        if (Start < limit.Start) Start = limit.Start;
        if (End   > limit.End)   End   = limit.End;
    }
    void Clamp(T min, T max) {
        if (Start < min) Start = min;
        if (End   > max) End   = max;
    }
    bool Contains(T val) const { return val >= Start && val <= End; }
    bool Contains(Span<T> other) const { return other.Start >= Start && other.End <= End; }
    void Reverse() { T tmp = Start; Start = End; End = tmp; }
    void Fit(T val) {
        if (val < Start) Start = val;
        if (val > End)   End   = val;
    }

    bool operator==(const Span<T>& other) const { return Start == other.Start && End == other.End; }
    bool operator!=(const Span<T>& other) const { return Start != other.Start || End != other.End; }

    Span() { };
    Span(T start, T end) : Start(start), End(end) { }
    Span(T val) : Start(val), End(val) { }
};

template <typename T>
struct Complex {
    T R = 0;
    T I = 0;

    Complex operator+(const Complex& val) const { return Complex(R + val.R, I + val.I); }
    Complex operator-(const Complex& val) const { return Complex(R - val.R, I - val.I); }
    Complex operator*(const Complex& val) const {
        T real = R * val.R - I * val.I;
        T imag = R * val.I + val.R * I;
        return Complex(real, imag);
    }
    Complex operator/(const Complex& val) const {
        T denom = val.R * val.R + val.I * val.I;
        T real = (R * val.R + I * val.I) / denom;
        T imag = (I * val.R - R * val.I) / denom;
        return Complex(real, imag);
    }
    Complex& operator+=(const Complex& val) {
        R += val.R;
        I += val.I;
        return *this;
    }
    Complex& operator-=(const Complex& val) {
        R -= val.R;
        I -= val.I;
        return *this;
    }
    Complex& operator*=(const Complex& val) {
        T real = R * val.R - I * val.I;
        T imag = R * val.I + val.R * I;
        R = real;
        I = imag;
        return *this;
    }
    Complex& operator/=(const Complex& val) {
        T denom = val.R * val.R + val.I * val.I;
        T real = (R * val.R + I * val.I) / denom;
        T imag = (I * val.R - R * val.I) / denom;
        R = real;
        I = imag;
        return *this;
    }

    T Magnitude() const { return sqrt(R * R + I * I); }

    Complex() { }
    Complex(T r) : R(r), I(0) { }
    Complex(T r, T i) : R(r), I(i) { }
};

using Timespan = Span<double>;

#endif
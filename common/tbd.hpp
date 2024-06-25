#ifndef __TBD_H_
#define __TBD_H_

#include <vector>
#include "include/birch.h"

class FieldLabel {
public:
    const char* Label() const;
    double Value() const;

    FieldLabel(const char* label, double value);
    ~FieldLabel();

    FieldLabel (const FieldLabel&) = delete;
    FieldLabel& operator= (const FieldLabel&) = delete;
private:
    char* label = nullptr;
    double value;
};

class TBDField {
public:
    const char*        Name();
    volatile unsigned* ElementCount();
    unsigned           MaxElementCount();
    double*            Data();

    void AddFieldLabel(const char* name, double value);
    void ClearLabels();
    
    const std::vector<FieldLabel*>* Labels();

    TBDField(const char* name, unsigned size);
    ~TBDField();

    TBDField (const TBDField&) = delete;
    TBDField& operator= (const TBDField&) = delete;
private:
    char* name = nullptr;
    volatile unsigned elCount;
    const unsigned size;

    std::vector<FieldLabel*>* labels = nullptr;

    double* data = nullptr;
};

void ParseTBD(double* out, char* data, unsigned elcount, unsigned recSize, const Birch::TBDSubrecord* sbr, TBDField* field);

#endif
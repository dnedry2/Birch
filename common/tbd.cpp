#include "tbd.hpp"

#include <cstring>
#include <string>

using namespace Birch;

const char* FieldLabel::Label() const {
    return label;
}
double FieldLabel::Value() const {
    return value;
}
FieldLabel::FieldLabel(const char* name, double value) {
    this->value = value;
    
    auto nameLen = strnlen(name, 128) + 1;
    label = new char[nameLen];
    strncpy(label, name, nameLen);
}
FieldLabel::~FieldLabel() {
    delete[] label;
}


const char* TBDField::Name() {
    return name;
}
volatile unsigned* TBDField::ElementCount() {
    return &elCount;
}
unsigned TBDField::MaxElementCount() {
    return size;
}
const std::vector<FieldLabel*>* TBDField::Labels() {
    return labels;
}
void TBDField::AddFieldLabel(const char* name, double value) {
    labels->push_back(new FieldLabel(name, value));
}
void TBDField::ClearLabels() {
    for (auto l : *labels)
        delete l;

    labels->clear();
}
double* TBDField::Data() {
    return data;
}

TBDField::TBDField(const char* name, unsigned size) : size(size) {
    auto nameLen = strlen(name);
    this->name = new char[nameLen];
    strncpy(this->name, name, nameLen);

    this->data = new double[size];
    elCount = 0;

    labels = new std::vector<FieldLabel*>();
}
TBDField::~TBDField() {
    for (auto l : *labels)
        delete l;

    delete labels;
    delete[] name;
}

template<typename T>
static inline void _parseTBD(double* out, char* data, unsigned elcount, unsigned offset, unsigned recSize) {
    char* pos = data + offset;

    for (unsigned i = 0; i < elcount; i++, pos += recSize)
        out[i] = *reinterpret_cast<T*>(pos);
}

static inline void _parseTBDAscii(double* out, char* data, unsigned elcount, unsigned recSize, const TBDSubrecord* sbr, TBDField* field) {
    char* pos = data + sbr->Offset;

    unsigned cVal = field->Labels()->size();

    for (unsigned i = 0; i < elcount; i++, pos += recSize) {
        for (const auto& fl : *field->Labels()) {
            if (!strncmp(fl->Label(), pos, sbr->Size)) {
                out[i] = fl->Value();
                goto next;
            }
        }

        out[i] = cVal;
        field->AddFieldLabel(std::string(pos, sbr->Size).c_str(), cVal++);
        
        next:;
    }
}

void ParseTBD(double* out, char* data, unsigned elcount, unsigned recSize, const TBDSubrecord* sbr, TBDField* field) {
    switch (sbr->Format)
    {
        case Birch::TBDSubrecord::RecFormat::Int8:
            _parseTBD<int8_t>(out, data, elcount, sbr->Offset, recSize);
            break;
        case Birch::TBDSubrecord::RecFormat::Int16:
            _parseTBD<int16_t>(out, data, elcount, sbr->Offset, recSize);
            break;
        case Birch::TBDSubrecord::RecFormat::Int32:
            _parseTBD<int32_t>(out, data, elcount, sbr->Offset, recSize);
            break;
        case Birch::TBDSubrecord::RecFormat::Int64:
            _parseTBD<int64_t>(out, data, elcount, sbr->Offset, recSize);
            break;
        case Birch::TBDSubrecord::RecFormat::Float:
            _parseTBD<float>(out, data, elcount, sbr->Offset, recSize);
            break;
        case Birch::TBDSubrecord::RecFormat::Double:
            _parseTBD<double>(out, data, elcount, sbr->Offset, recSize);
            break;
        case Birch::TBDSubrecord::RecFormat::Text:
            _parseTBDAscii(out, data, elcount, recSize, sbr, field);
            break;
        default:
            break;
    }

    if (sbr->LabelCount > 0 && sbr->Format != Birch::TBDSubrecord::RecFormat::Text) {
        field->ClearLabels();

        for (uint i = 0; i < sbr->LabelCount; i++) {
            field->AddFieldLabel(sbr->Labels[i].text, sbr->Labels[i].value);
        }
    }
}
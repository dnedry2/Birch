#include <cstring>
#include <filesystem>
namespace fs = std::filesystem;

#include "birch.h"
#include "endian.hxx"

#include "imgui.h"

#ifndef t_ImTexID
#define t_ImTexID(tex) ((void*)(intptr_t)(tex))
#endif

using namespace Birch;

static unsigned icon = 0;

static unsigned dataFormatSizes[] = {
    1, // Int8
    2, // Int16
    4, // Int32
    8, // Int64
    4, // Float
    8  // Double
};

class RawImport : public PluginIQGetter {
public:
    const char* FileExtensions() override { return ""; }
    double      ClipThreshold() override { return 0; }
    const char* SafeName() override { return "Raw"; }
	Birch::DataFormat Format() override { return format; }
	double SampleRate() override { return sampleRate; }
	Timespan Time() override { return timespan; }
    unsigned long long TimeIdx(double time) override {
        if (time < 0)
            return 0;

        auto el = time / (1 / sampleRate);
        auto elCount = (fileSize - offset) / dataFormatSizes[(int)format];

        if (!real)
            elCount /= 2;

        if (el > elCount)
            return elCount;
        else
            return el;
    }

    bool Open(const char* path) override {
        if (file != nullptr) {
            delete file;
            file = nullptr;

            delete[] safeFileName;
            safeFileName = nullptr;

            delete[] filePath;
            filePath = nullptr;
        }

        file = fopen(path, "rb");

        if (file == NULL)
            return false;

        fseek(file, 0, SEEK_END);
        fileSize = ftell(file);
        rewind(file);

        auto elSize = dataFormatSizes[(int)format];
        if (!real)
            elSize *= 2;

        timespan = Birch::Timespan(0, (fileSize - offset) / elSize * (1 / sampleRate));

        safeFileName = strdup(fs::path(path).filename().c_str());
        filePath = strdup(path);

        return true;
    }
    bool Seek(size_t n) override {
        auto byte = n * dataFormatSizes[(int)format];

        if (!real)
            byte *= 2;

        byte += offset;

        return fseek(file, byte, SEEK_SET) == 0 ? true : false;
    }
    template<typename T>
    static inline void spacer(T* array, unsigned cnt) {
        for (int i = cnt / 2, j = 0; i > -1; i--, j += 2) {
            array[cnt - j] = array[i];
            array[cnt - j + 1] = 0;
        }
    }
    unsigned Read(char* dest, unsigned cnt) override {
        auto elSize = dataFormatSizes[(int)format];
        auto mult = real ? 1 : 2;

        elSize *= mult;

    	auto out = fread(dest, elSize, cnt, file);

        if (endian) {
            switch (format)
            {
                case DataFormat::Int8:
                break; // Silence warning
                case DataFormat::Int16:
                    ReverseArrayEndian((int16_t*)dest, cnt * mult);
                    break;
                case DataFormat::Int32:
                    ReverseArrayEndian((int32_t*)dest, cnt * mult);
                    break;
                case DataFormat::Int64:
                    ReverseArrayEndian((int64_t*)dest, cnt * mult);
                    break;
                case DataFormat::Float:
                    ReverseArrayEndian((float*)dest, cnt * mult);
                    break;
                case DataFormat::Double:
                    ReverseArrayEndian((double*)dest, cnt * mult);
                    break;
            }
        }

        if (!sign) {
            switch (format)
            {
                case DataFormat::Int8:
                {
                    auto* udest = (unsigned char *)dest;
                    for (unsigned i = 0; i < cnt * mult; i++)
                        dest[i] = udest[i] - UINT8_MAX / 2;
                }
                    break;
                case DataFormat::Int16:
                {
                    auto* udest = (unsigned short *)dest;
                    for (unsigned i = 0; i < cnt * mult; i++)
                        dest[i] = udest[i] - UINT16_MAX / 2;
                }
                    break;
                case DataFormat::Int32:
                {
                    auto* udest = (unsigned int *)dest;
                    for (unsigned i = 0; i < cnt * mult; i++)
                        dest[i] = udest[i] - UINT32_MAX / 2;
                }
                    break;
                case DataFormat::Int64:
                {
                    auto* udest = (unsigned long long *)dest;
                    for (unsigned i = 0; i < cnt * mult; i++)
                        dest[i] = udest[i] - UINT64_MAX / 2;
                }
                    break;
                case DataFormat::Float:
                break;
                case DataFormat::Double:
                break;
            }
        }

        // Need to zero out the imaginary part
        // TODO: hilbert transform
        if (real) {
            switch (format)
            {
                case DataFormat::Int8:
                    spacer((int8_t*)dest, cnt * 2);
                    break;
                case DataFormat::Int16:
                    spacer((int16_t*)dest, cnt * 2);
                    break;
                case DataFormat::Int32:
                    spacer((int32_t*)dest, cnt * 2);
                    break;
                case DataFormat::Int64:
                    spacer((int64_t*)dest, cnt * 2);
                    break;
                case DataFormat::Float:
                    spacer((float*)dest, cnt * 2);
                    break;
                case DataFormat::Double:
                    spacer((double*)dest, cnt * 2);
                    break;
            }
        }

        return out;
    }

    unsigned long long SpanElementCount(Birch::Timespan span) override {
      return TimeIdx(span.End) - TimeIdx(span.Start);
    }
    const char* SafeFileName() override {
        return safeFileName;
    }
    const char* FilePath() override {
        return filePath;
    }

    void RenderInfo(void* gui) override {
        ImGui::SetCurrentContext(reinterpret_cast<ImGuiContext*>(gui));

        ImGui::PushID(this);

        ImGui::Text("%s", SafeFileName());

        if (ImGui::BeginChild(124, ImVec2(0, -1))) {
            bool changed = false;

            int fmt = 0;
            switch (format)
            {
            case DataFormat::Float:
                fmt = 0;
                break;
            case DataFormat::Double:    
                fmt = 1;
                break;
            case DataFormat::Int8:
                fmt = 2;
                break;
            case DataFormat::Int16:
                fmt = 3;
                break;
            case DataFormat::Int32:
                fmt = 4;
                break;
            case DataFormat::Int64:
                fmt = 5;
                break;
            }

            if (!sign)
                fmt += 4;

            if (ImGui::Combo("Format", (int*)&fmt, "Float\0Double\0Int8\0Int16\0Int32\0Int64\0Unsigned Int8\0Unsigned Int16\0Unsigned Int32\0Unsigned Int64\0")) {
                switch (fmt)
                {
                case 0:
                    format = DataFormat::Float;
                    sign = true;
                    break;
                case 1:
                    format = DataFormat::Double;
                    sign = true;
                    break;
                case 2:
                    format = DataFormat::Int8;
                    sign = true;
                    break;
                case 3:
                    format = DataFormat::Int16;
                    sign = true;
                    break;
                case 4:
                    format = DataFormat::Int32;
                    sign = true;
                    break;
                case 5:
                    format = DataFormat::Int64;
                    sign = true;
                    break;
                case 6:
                    format = DataFormat::Int8;
                    sign = false;
                    break;
                case 7:
                    format = DataFormat::Int16;
                    sign = false;
                    break;
                case 8:
                    format = DataFormat::Int32;
                    sign = false;
                    break;
                case 9:
                    format = DataFormat::Int64;
                    sign = false;
                    break;
                }

                changed = true;
            }

            if (ImGui::Combo("Endian", &endianVal, "Little\0Big\0")) {
                endian = SysBigEndian() != endianVal;
                changed = true;
            }

            changed |= ImGui::InputDouble("Sample rate", &sampleRate, 0, 0, "%.0f");

            if (ImGui::Combo("Complexity", &realVal, "Complex\0Real\0")) {
                real = realVal == 1;
                changed = true;
            }

            changed |= ImGui::InputInt("Offset", &offset); // Should be unsigned... does imgui support that? There's probably a min value setting, too drunk for that rn

            if (changed) {
                auto elSize = dataFormatSizes[(int)format];
                if (!real)
                    elSize *= 2;

                timespan = Birch::Timespan(0, (fileSize - offset) / elSize * (1 / sampleRate));
            }
        }
        ImGui::EndChild();

        ImGui::PopID();
    }

    RawImport() {
        Type = PluginType::IQGetter;
    }
    ~RawImport() {
        delete file;
        delete[] safeFileName;
        delete[] filePath;
    }

private:
	Birch::DataFormat format = Birch::DataFormat::Float;
	double sampleRate = 1800000;
	Timespan timespan;

    FILE* file = nullptr;
    size_t fileSize = 0;

    bool real   = false;
    int  realVal = 0;    // 0 = complex, 1 = real   
    bool endian = false; // Swap endian?
    int  endianVal = 0;  // 0 = little, 1 = big
    bool sign   = true;  // Is data signed?
    int  offset = 0;   // Offset in bytes

    char* safeFileName = nullptr;
    char* filePath = nullptr;
};


static Plugin* NewInstance() {
    return new RawImport();
}
static void Destroy(Plugin* in) {
    delete in;
}

extern "C" PluginInterface Interface() {
    PluginInterface pi = {"Raw Import", PluginType::IQGetter, &NewInstance, &Destroy};
    return pi;
}
#include <cstring>
#include <filesystem>
namespace fs = std::filesystem;

#include "birch.h"

#include "imgui.h"

#include <sndfile.h>

#ifndef t_ImTexID
#define t_ImTexID(tex) ((void*)(intptr_t)(tex))
#endif

using namespace Birch;

static unsigned icon = 0;

template<typename T>
static inline void spacer(T* array, unsigned cnt) {
    for (int i = cnt / 2, j = 0; i > -1; i--, j += 2) {
        array[cnt - j] = array[i];
        array[cnt - j + 1] = 0;
    }
}

class SNDFile : public Birch::PluginIQGetter {
public:
    const char*        SafeName()           override { return "Audio Import"; }
	Birch::DataFormat  Format()             override { return format; }
	double             SampleRate()         override { return sampleRate; }
	Timespan           Time()               override { return timespan; }
    unsigned long long TimeIdx(double time) override { return static_cast<unsigned long long>(time * sampleRate); }
    double             ClipThreshold()      override { return clipThreshold; }
    const char*        FileExtensions()     override { return fileExtensions; }

    bool Open(const char* path) override {
        if (file != nullptr) {
            sf_close(file);
            file = nullptr;

            delete[] filePath;
            filePath = nullptr;

            delete[] safeFileName;
            safeFileName = nullptr;
        }

        // Open file using libsndfile
        SF_INFO info;
        file = sf_open(path, SFM_READ, &info);

        if (file == nullptr) {
            printf("Failed to open file: %s\n", path);

            return false;
        }

        if (info.channels > 2 || info.channels < 1) {
            printf("Unsupported channel count: %d\n", info.channels);

            sf_close(file);
            file = nullptr;

            return false;
        }

        sampleRate = info.samplerate;
        channels   = info.channels;
        size       = info.frames;

        sf_count_t bufferSize = info.channels * info.frames;

        // TODO: Sometimes info.frames is max int64 which leads to overflow / crash
        // Not sure why i come back later        

        if (bufferSize <= 0) {
            printf("Invalid buffer size: %lld\n", bufferSize);

            sf_close(file);
            file = nullptr;

            return false;
        }

        if (info.channels == 1)
            bufferSize *= 2;

        buffer = new float[bufferSize];
        sf_count_t readCount = sf_readf_float(file, buffer, info.frames);

        sf_close(file);

        if (channels == 1)
            spacer(buffer, bufferSize);

        timespan = Birch::Timespan(0, info.frames / info.samplerate);

        safeFileName = strdup(fs::path(path).filename().c_str());
        filePath     = strdup(path);

        return true;
    }
    bool Seek(size_t n) override {
        pos = n;

        if (pos >= size) {
            pos = size;
            return false;
        }

        return true;
    }

    unsigned Read(char* dest, unsigned cnt) override {
        if (pos >= size)
            return 0;
        
        uint toRead = cnt;
        if (pos + cnt > size)
            toRead = size - pos;
        
        float* out = reinterpret_cast<float*>(dest);

        std::copy(buffer + pos * 2, buffer + (pos + toRead) * 2, out);

        return toRead;
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

        }
        ImGui::EndChild();

        ImGui::PopID();
    }

    SNDFile() {
        Type = PluginType::IQGetter;
    }
    ~SNDFile() {
        delete[] safeFileName;
        delete[] filePath;
    }

private:
	Birch::DataFormat format = Birch::DataFormat::Float;
	double sampleRate = 1;
	Timespan timespan;

    SNDFILE* file = nullptr;

    size_t size     = 0;
    size_t pos      = 0;
    uint   channels = 0;

    double clipThreshold = 0.99999;

    const char* fileExtensions = "wav,aiff,au,pvf,caf,flac,ogg,mp3";

    float* buffer = nullptr;

    char* safeFileName = nullptr;
    char* filePath     = nullptr;
};


static Plugin* NewInstance() {
    return new SNDFile();
}
static void Destroy(Plugin* in) {
    delete in;
}

extern "C" PluginInterface Interface() {
    PluginInterface pi = {"Audio Import", PluginType::IQGetter, &NewInstance, &Destroy};
    return pi;
}
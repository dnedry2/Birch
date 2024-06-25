#ifndef __BIRCH_IQ_SIGNAL__
#define __BIRCH_IQ_SIGNAL__

#include "util.h"
#include "plugin.h"

enum class DataFormat { Int8, Int16, Int32, Int64, Float, Double };

class PluginIQGetter : public Plugin {
public:
    // Returns format of samples
    virtual DataFormat Format() = 0;

    // Open file. Return true on success
    virtual bool Open(const char* path) = 0;
    // Return total timespan of file
    virtual Timespan Time() = 0;
    // Return elements in span
    virtual unsigned long long SpanElementCount(Birch::Timespan span) = 0;
    // Return element idx of time
    virtual unsigned long long TimeIdx(double time) = 0;
    // Seek element in file. Return true on success
    virtual bool Seek(size_t n) = 0;
    // Read cnt samples to dest. Advance position. Return total elements read
    virtual unsigned Read(char* dest, unsigned cnt) = 0;
    // Return signal sample rate
    virtual double SampleRate() = 0;
    // Render header / info display
    virtual void RenderInfo(void* gui) = 0;
    // Name in context menu
    virtual const char* SafeName() = 0;
    // Safe filename
    virtual const char* SafeFileName() = 0;
    // File path
    virtual const char* FilePath() = 0;
    // Clipping threshold
    virtual double ClipThreshold() = 0;
    // File extension filter (ex: "wav,mp3,ogg"). Leave empty for no filter
    virtual const char* FileExtensions() = 0;
    // Return true if the birch should provide a file path to the plugin (false disables file picker)
    virtual bool NoFilePath() = 0;
};

#endif
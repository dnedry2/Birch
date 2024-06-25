#ifndef __BIRCH_TBD_SIGNAL__
#define __BIRCH_TBD_SIGNAL__

#include "util.h"
#include "plugin.h"

struct TBDSubrecord {
    enum class RecFormat { Int8, Int16, Int32, Int64, Float, Double, Text, Empty };

    char      Name[128];                 // Record name
    RecFormat Format  = RecFormat::Int8; // Record format
    unsigned  Size    = 1;               // Record size (bytes)
    unsigned  Offset  = 0;               // Record offset (bytes)
    bool      Mutable = false;           // Record can be edited
    //char      Unit[32];                  // Record unit name ( TODO )

    // Used to display custom labels
    struct label {
        double value;    // Value to display label at
        char   text[32]; // Label text
    };
    label* Labels = nullptr; // Array of labels
    unsigned LabelCount = 0; // Number of labels. Leave at 0 if no custom labels are used.
};

class PluginTBDGetter : public Plugin {
public:
    // Returns an array of subrecords.
    // Must be in the same order they appear in the record
    virtual const TBDSubrecord* Format() = 0;
    // Returns the number of subrecords per record
    virtual unsigned SubrecordCount() = 0;
    // Returns the size of one record
    virtual unsigned RecordSize() = 0;
    // Returrns the name of the TOA field
    virtual const char* TOAName() = 0;

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
    // Render header / info display
    virtual void RenderInfo(void* gui) = 0;
    // Name in context menu
    virtual const char* SafeName() = 0;
    // Safe filename
    virtual const char* SafeFileName() = 0;
    // File path
    virtual const char* FilePath() = 0;
    // File extension filter (ex: "wav,mp3,ogg"). Return nullptr or "" for no filter
    virtual const char* FileExtensions() = 0;
    // Return true if the birch should provide a file path to the plugin (false disables file picker)
    virtual bool NoFilePath() = 0;

    // Call to reload data for current view
    void (*Update)(const PluginTBDGetter*) = nullptr;


    /***********************************************************************************
    * FILE EDITING FUNCTIONS
    * These functions are only called if SupportsEdit() returns true
    ************************************************************************************/
    
    // Return true if the plugin supports editing files
    //virtual bool SupportsEdit() = 0;

    // Add a field to the file. Return true on success
    //virtual bool AddRecord(const TBDSubrecord* record) = 0;
    // Remove a field from the file. Return true on success
    //virtual bool RemoveRecord(const TBDSubrecord* record) = 0;
    // Modify an element in a field in the file. Return true on success
    //virtual bool ModifyElement(const TBDSubrecord* record, size_t element, const char* value) = 0;
    // Modify multiple elements in a field in the file. Return true on success
    //virtual bool ModifyElements(const TBDSubrecord* record, size_t start, size_t count, const char* values) = 0;


    // Internal use, don't touch
    void* _bff = nullptr;
    void* _bfs = nullptr;
};

#endif

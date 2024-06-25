#ifndef PACKFILE_H
#define PACKFILE_H

#include <string>
#include <map>
#include <cstdio>
#include <vector>

class PackFile {
public:
    struct Entry {
        friend class PackFile;

    public:
        Entry(std::string name, size_t size, size_t offset, char* data, bool cache);
        ~Entry();

        const std::string name;
        const size_t size;
        const char* data() const;
        const bool cache;

        std::string toString() const;

    private:
        size_t offset;
        char *_data = nullptr;

        static bool compEntry(PackFile::Entry* a, PackFile::Entry* b);
    };

    explicit PackFile(std::string path);
    ~PackFile();

    const Entry* operator[](std::string name);

    const Entry* Find(std::string name);

    void Add(std::string name, size_t size, const char* data, bool cache);

    void Erase(std::string name);

    std::vector<std::string> Keys();

private:
    FILE* ioFile = NULL;
    std::map<std::string, Entry*> index;

    void initNew();
    void initExist();

    const Entry* fetch(Entry* entry);
    void cache(Entry* entry);

    size_t move(Entry* entry, size_t startOffset);
    void writeEntry(Entry* entry, const char* data);
};

#endif
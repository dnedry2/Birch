#include "packfile.hpp"
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <cstring>
#include <cinttypes>

using std::string;

PackFile::Entry::Entry(string name, size_t size, size_t offset, char *data, bool cache)
                : name(name), size(size), offset(offset), _data(data), cache(cache) {}

PackFile::Entry::~Entry() {
    delete[] _data;
}

const char* PackFile::Entry::data() const {
    return _data;
}

bool PackFile::Entry::compEntry(PackFile::Entry* a, PackFile::Entry* b) {
    return a->offset < b->offset;
}

string PackFile::Entry::toString() const {
    return string(_data, size);
}

PackFile::PackFile(string path)
{
    index = std::map<string, Entry*>();

    //if (access(path.c_str(), F_OK) == -1)
    //    throw("Failed to open file");

    if ((ioFile = fopen(path.c_str(), "r+")))
        initExist();
    else if ((ioFile = fopen(path.c_str(), "w+")))
        initNew();
    else
        throw("Failed to open file");
}
PackFile::~PackFile() {
    fclose(ioFile);

    for (auto pair : index)
        delete pair.second;
}
void PackFile::initExist() {
    char mn[2];
    fread(&mn, 2, 1, ioFile);

    if (memcmp(mn, "pf", 2))
        throw("Invalid format");
    
    uint16_t nameLen = 0;
    while (fread(&nameLen, sizeof(nameLen), 1, ioFile)) {
        char* nameBuf = new char[nameLen];
        fread(nameBuf, nameLen, 1, ioFile);

        bool doCache = false;
        fread(&doCache, sizeof(doCache), 1, ioFile);

        size_t dataLen = 0;
        fread(&dataLen, sizeof(dataLen), 1, ioFile);

        Entry* entry = new Entry(string(nameBuf), dataLen, ftell(ioFile), nullptr, doCache);

        if (entry->cache)
            cache(entry);

        index.insert(std::make_pair(string(nameBuf), entry));

        fseek(ioFile, entry->offset + entry->size, SEEK_SET);

        delete[] nameBuf;
    }
}

void PackFile::initNew() {
    uint32_t fcnt = 0;

    fseek(ioFile, 0, SEEK_SET);
    fwrite("pf", 2, 1, ioFile);
}

const PackFile::Entry* PackFile::fetch(Entry* entry) {
    if (entry->cache)
        return entry;
    
    Entry* out = new Entry(entry->name, entry->size, entry->offset, nullptr, false);

    cache(out);

    return out;
}
void PackFile::cache(Entry* entry) {
    entry->_data = new char[entry->size];

    fseek(ioFile, entry->offset, SEEK_SET);
    fread(entry->_data, entry->size, 1, ioFile);
}

const PackFile::Entry* PackFile::Find(string name) {
    Entry* e = index[name];
    return fetch(e);
}
const PackFile::Entry* PackFile::operator[](string name) {
    return Find(name);
}

void PackFile::Add(string name, size_t size, const char* data, bool doCache) {
    fseek(ioFile, 0, SEEK_END);

    Entry* entry = new Entry(name, size, 0, nullptr, doCache);

    writeEntry(entry, data);
    entry->offset = ftell(ioFile) - size;

    if (doCache)
        cache(entry);
    
    index.insert(std::make_pair(name, entry));
}

void PackFile::Erase(string name) {
    Entry* entry = index[name];

    size_t eCount = index.size();

    std::vector<Entry*> entries = std::vector<Entry*>();
    entries.reserve(eCount);

    for (auto pair : index)
        entries.push_back(pair.second);

    std::sort(entries.begin(), entries.end(), &PackFile::Entry::compEntry);
    int idx = std::distance(entries.begin(), std::lower_bound(entries.begin(), entries.end(), entry, &PackFile::Entry::compEntry));

    size_t off = entry->offset - (entry->name.length() + 1) - sizeof(entry->cache) - sizeof(entry->size) - sizeof(uint16_t);
    for (size_t i = idx + 1; i < eCount; i++)
        off = move(entries[i], off);

    ftruncate(fileno(ioFile), off);

    delete entry;
    index.erase(name);
}

size_t PackFile::move(Entry* entry, size_t startOffset) {
    if (!entry->cache)
        cache(entry);

    fseek(ioFile, startOffset, SEEK_SET);
    writeEntry(entry, entry->_data);

    if (!entry->cache) {
        delete[] entry->_data;
        entry->_data = nullptr;
    }

    return ftell(ioFile);
}

void PackFile::writeEntry(Entry* entry, const char* data) {
    uint16_t nameLen = entry->name.length() + 1;

    fwrite(&nameLen, sizeof(nameLen), 1, ioFile);
    fwrite(entry->name.c_str(), nameLen, 1, ioFile);
    fwrite(&entry->cache, sizeof(entry->cache), 1, ioFile);
    fwrite(&entry->size, sizeof(entry->size), 1, ioFile);
    fwrite(data, entry->size, 1, ioFile);
}

std::vector<string> PackFile::Keys() {
    std::vector<string> out = std::vector<string>();

    std::transform(index.begin(), index.end(), std::back_inserter(out), [](const auto& en) { return en.first; });

    return out;
}
// Utility for the bpk files

#include <iostream>
#include <string>
#include <filesystem>
#include <cstring>

#include "packfile.hpp"

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::vector;
namespace fs = std::filesystem;

void help() {
    cout << "Packfile utility" << endl;
    cout << "usage: " << endl;
    cout << "\t-p <outpath> [path 1] [path 2] ...\tcreate a new file or add to an existing" << endl;
    cout << "\t-u <packfile> <outpath>\t\t\tunpack a file" << endl;
    cout << "\t-l <packfile>\t\t\t\tlist file contents" << endl;
}

void list(string path) {
    PackFile* pf = nullptr;

    try
    {
        pf = new PackFile(path);
    }
    catch (char const *error)
    {
        std::cerr << error << endl;
        return;
    }

    auto keys = pf->Keys();

    cout << "Resources in file '" << path << "':" << endl;
    for (auto& key : keys)
        cout << "\t" << key << endl;

    delete pf;
}

void pack(string path, vector<string> infiles) {
    PackFile* pf = nullptr;

    try
    {
        pf = new PackFile(path);
    }
    catch (char const *error)
    {
        std::cerr << error << endl;
        return;
    }

    for (auto& file : infiles) {
        FILE* fp = NULL;

        if ((fp = fopen(file.c_str(), "r"))) {
            fseek(fp, 0, SEEK_END);
            size_t fLen = ftell(fp);
            rewind(fp);

            char* buf = new char[fLen];

            fread(buf, fLen, 1, fp);

            pf->Add(file, fLen, buf, false);

            delete[] buf;
            fclose(fp);

            cout << "Added '" << file << "'" << endl;
        } else {
            std::cerr << "Failed to open file '" << file << "'" << endl;
        }
    }

    delete pf;
}

void unpack(string path, string out) {
    PackFile* pf = nullptr;

    try
    {
        pf = new PackFile(path);
    }
    catch (char const *error)
    {
        std::cerr << error << endl;
        return;
    }

    auto keys = pf->Keys();
    for (auto& key : keys) {
        FILE* fp = NULL;
        auto entry = (*pf)[key];

        auto outPath = fs::path(out + "/" + entry->name);
        auto dir = fs::path(out + "/" + entry->name).remove_filename();
        fs::create_directories(dir);

        if ((fp = fopen(outPath.string().c_str(), "w+"))) {
            fwrite(entry->data(), entry->size, 1, fp);
            fclose(fp);

            cout << "Wrote '" << entry->name << "'" << endl;
        } else {
            std::cerr << "Failed to open file '" << outPath << "'" << endl;
        }
    }

    delete pf;
}

vector<string> getFiles(string dir) {
    vector<string> out = vector<string>();

    for (const auto &entry : fs::directory_iterator(dir))
    {
        if (fs::is_directory(entry))
        {
            auto files = getFiles(entry.path().string());
            out.insert(out.end(), files.begin(), files.end());
        }
        else if (fs::is_regular_file(entry))
        {
            out.push_back(entry.path().string());
        }
    }

    return out;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        help();
        return 0;
    }

    if (!strncmp("-l", argv[1], 2))
        list(argv[2]);
    
    if (!strncmp("-p", argv[1], 2)) {
        vector<string> inputs = vector<string>();

        for (int i = 3; i < argc; i++) {
            auto files = getFiles(argv[i]);
            inputs.insert(inputs.end(), files.begin(), files.end());
        }

        pack(argv[2], inputs);
    }

    if (!strncmp("-u", argv[1], 2))
        unpack(argv[2], argv[3]);

    return 0;
}
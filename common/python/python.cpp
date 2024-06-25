#include <iostream>
#include <string>
#include <filesystem>
#include <vector>

#include "python.hpp"
#include "python.hxx"

namespace Py = Python;

PythonIQProcessor::PythonIQProcessor(const char* file) {
    auto mod = Py::LoadModule(file);
    module = new Py::PyObj(mod);

    mod.Free = false;

    if (!Py::HasFunctions(module, api))
        throw std::runtime_error("API not implemented");

    std::string type = Py::Call(module, "Type");
    if (type != "IQProcessor")
        throw std::runtime_error("Not an IQProcessor");

    std::string ts = Py::Call(module, "ToolbarSVG");
    std::string cs = Py::Call(module, "CursorSVG");
    std::string n  = Py::Call(module, "Name");

    toolbarSVG = ts;
    cursorSVG = cs;
    name = n;
}
PythonIQProcessor::~PythonIQProcessor() {
    Py::Free(module);
}

const std::string& PythonIQProcessor::ToolbarSVG() const {
    return toolbarSVG;
}
const std::string& PythonIQProcessor::CursorSVG() const {
    return cursorSVG;
}
const std::string& PythonIQProcessor::Name() const {
    return name;
}
void PythonIQProcessor::Process(double* I, double* Q, double* TOA, uint len) {
    Py::PyObj si  = Py::MakeArrayObj<double>(I, len);
    Py::PyObj sq  = Py::MakeArrayObj<double>(Q, len);
    Py::PyObj toa = Py::MakeArrayObj<double>(TOA, len);

    Py::Call(module, "Process", si, sq, toa);
}


PythonTBDProcessor::PythonTBDProcessor(const char* file) {
    auto mod = Py::LoadModule(file);
    module = new Py::PyObj(mod);

    mod.Free = false;

    if (!Py::HasFunctions(module, api))
        throw std::runtime_error("API not implemented");

    std::string type = Py::Call(module, "Type");
    if (type != "TBDProcessor")
        throw std::runtime_error("Not an TBDProcessor");

    std::string ts = Py::Call(module, "ToolbarSVG");
    std::string cs = Py::Call(module, "CursorSVG");
    std::string n  = Py::Call(module, "Name");

    toolbarSVG = ts;
    cursorSVG = cs;
    name = n;
}
PythonTBDProcessor::~PythonTBDProcessor() {
    Py::Free(module);
}

const std::string& PythonTBDProcessor::ToolbarSVG() const {
    return toolbarSVG;
}
const std::string& PythonTBDProcessor::CursorSVG() const {
    return cursorSVG;
}
const std::string& PythonTBDProcessor::Name() const {
    return name;
}
void PythonTBDProcessor::Process(const std::map<std::string, double*>& data, uint len) {
    auto dict = Py::PyDict();

    for (auto [key, value] : data)
        dict.Add(key.c_str(), Py::MakeArrayObj<double>(value, len));

    Py::Call(module, "Process", dict.Get());
}


std::vector<PythonIQProcessor*> LoadPythonIQProcessors(const char* dir) {
    std::vector<PythonIQProcessor*> processors;

    for (auto& p : std::filesystem::directory_iterator(dir)) {
        if (p.path().extension() == ".py") {
            try {
                processors.push_back(new PythonIQProcessor(p.path().stem().string().c_str()));
            } catch (std::exception& e) {
                //std::cerr << "Error loading " << p.path() << ": " << e.what() << std::endl;
            }
        }
    }

    return processors;
}

std::vector<PythonTBDProcessor*> LoadPythonTBDProcessors(const char* dir) {
    std::vector<PythonTBDProcessor*> processors;

    for (auto& p : std::filesystem::directory_iterator(dir)) {
        if (p.path().extension() == ".py") {
            try {
                processors.push_back(new PythonTBDProcessor(p.path().stem().string().c_str()));
            } catch (std::exception& e) {
                //std::cerr << "Error loading " << p.path() << ": " << e.what() << std::endl;
            }
        }
    }

    return processors;
}

static bool pyInitialized = false;

void InitPython(const char* dir) {
    Py::Init(dir);
    pyInitialized = true;
}
void ShutdownPython() {
    if (pyInitialized)
        Py::Shutdown();
}
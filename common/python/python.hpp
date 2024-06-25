#ifndef __B_PYTHON_HPP__
#define __B_PYTHON_HPP__

#include <string>
#include <vector>
#include <map>

namespace Python {
	struct PyObj;
};

class PythonIQProcessor {
public:
	PythonIQProcessor(const char* file);
    ~PythonIQProcessor();

	const std::string& ToolbarSVG() const;
	const std::string& CursorSVG() const;
	const std::string& Name() const;
	void Process(double* I, double* Q, double* TOA, uint len);

private:
	Python::PyObj* module;
	std::string toolbarSVG;
	std::string cursorSVG;
	std::string name;

    std::vector<std::string> api = {
    	"ToolbarSVG",
    	"CursorSVG",
    	"Name",
    	"Type",
    	"Process"
    };
};

class PythonTBDProcessor {
public:
	PythonTBDProcessor(const char* file);
    ~PythonTBDProcessor();

	const std::string& ToolbarSVG() const;
	const std::string& CursorSVG() const;
	const std::string& Name() const;
	void Process(const std::map<std::string, double*>& data, uint len);

private:
	Python::PyObj* module;
	std::string toolbarSVG;
	std::string cursorSVG;
	std::string name;

    std::vector<std::string> api = {
    	"ToolbarSVG",
    	"CursorSVG",
    	"Name",
    	"Type",
    	"Process"
    };
};

std::vector<PythonIQProcessor*> LoadPythonIQProcessors(const char* dir);
std::vector<PythonTBDProcessor*> LoadPythonTBDProcessors(const char* dir);

void InitPython(const char* dir);
void ShutdownPython();

#endif
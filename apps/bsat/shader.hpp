#ifndef SHADER_HPP
#define SHADER_HPP

#include <GL/gl3w.h>
#include "imgui.h"
#include "TextEditor.h"

#include <string>
#include <map>

class Shader {
public:
    Shader(const std::string& vertexCode, const std::string& fragCode);
    ~Shader();

    void Use() const;

    void ShowEditor(bool* p_open = NULL);

    void SetBool(const std::string& name, bool value) const;
    void SetInt(const std::string& name, int value) const;
    void SetFloat(const std::string& name, float value) const;
    void SetVector2(const std::string& name, float x, float y) const;
    void SetVector2(const std::string& name, const ImVec2& vec) const;
    void SetVector3(const std::string& name, float x, float y, float z) const;
    void SetVector4(const std::string& name, float x, float y, float z, float w) const;
    void SetVector4(const std::string& name, const ImVec4& vec) const;
    void SetMatrix2(const std::string& name, const float* value) const;
    void SetMatrix3(const std::string& name, const float* value) const;
    void SetMatrix4(const std::string& name, const float* value) const;
    void SetSampler2D(const std::string& name, unsigned value) const;
private:
    GLuint shaderID;

    std::string vertexCode;
    std::string fragCode;

    // Debugging stuff

    TextEditor* vertEditor = nullptr;
    TextEditor* fragEditor = nullptr;

    std::string lastError;

    bool autoCompile = true;

    void compile();
};

#endif
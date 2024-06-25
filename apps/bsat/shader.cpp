#include "shader.hpp"

#include <stdexcept>
#include <cstdio>

Shader::Shader(const std::string& vertexCode, const std::string& fragCode)
       : vertexCode(vertexCode), fragCode(fragCode)
{
    compile();

    vertEditor = new TextEditor();
    fragEditor = new TextEditor();

    vertEditor->SetLanguageDefinition(TextEditor::LanguageDefinition::GLSL());
    fragEditor->SetLanguageDefinition(TextEditor::LanguageDefinition::GLSL());

    vertEditor->SetText(vertexCode);
    fragEditor->SetText(fragCode);

    vertEditor->SetShowWhitespaces(false);
    fragEditor->SetShowWhitespaces(false);

    vertEditor->SetTabSize(4);
    fragEditor->SetTabSize(4);
}

Shader::~Shader() {
    glDeleteProgram(shaderID);

    delete vertEditor;
    delete fragEditor;
}

void Shader::compile() {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragShader   = glCreateShader(GL_FRAGMENT_SHADER);

    const char* vertexCodePtr = vertexCode.c_str();
    const char* fragCodePtr   = fragCode.c_str();

    glShaderSource(vertexShader, 1, &vertexCodePtr, NULL);
    glShaderSource(fragShader,   1, &fragCodePtr,   NULL);

    glCompileShader(vertexShader);
    glCompileShader(fragShader);

    shaderID = glCreateProgram();

    glAttachShader(shaderID, vertexShader);
    glAttachShader(shaderID, fragShader);
    glLinkProgram(shaderID);

    glDeleteShader(vertexShader);
    glDeleteShader(fragShader);

    int status;
    glGetProgramiv(shaderID, GL_LINK_STATUS, &status);

    if (status == GL_FALSE) {
        // Display compilation errors
        int infoLogLength;
        glGetProgramiv(shaderID, GL_INFO_LOG_LENGTH, &infoLogLength);
        char* strInfoLog = new char[infoLogLength + 1];
        glGetProgramInfoLog(shaderID, infoLogLength, NULL, strInfoLog);

        printf("Shader linking error: %s\n", strInfoLog);

        lastError = strInfoLog;

        delete[] strInfoLog;

        throw std::runtime_error("Failed to compile shader");
    } else {
        lastError = "No errors";
    }
}

void Shader::Use() const {
    glUseProgram(shaderID);
}

void Shader::SetBool(const std::string& name, bool value) const {
    glUniform1i(glGetUniformLocation(shaderID, name.c_str()), (int)value);
}
void Shader::SetInt(const std::string& name, int value) const {
    glUniform1i(glGetUniformLocation(shaderID, name.c_str()), value);
}
void Shader::SetFloat(const std::string& name, float value) const {
    glUniform1f(glGetUniformLocation(shaderID, name.c_str()), value);
}
void Shader::SetVector2(const std::string& name, float x, float y) const {
    glUniform2f(glGetUniformLocation(shaderID, name.c_str()), x, y);
}
void Shader::SetVector2(const std::string& name, const ImVec2& vec) const {
    glUniform2f(glGetUniformLocation(shaderID, name.c_str()), vec.x, vec.y);
}
void Shader::SetVector3(const std::string& name, float x, float y, float z) const {
    glUniform3f(glGetUniformLocation(shaderID, name.c_str()), x, y, z);
}
void Shader::SetVector4(const std::string& name, float x, float y, float z, float w) const {
    glUniform4f(glGetUniformLocation(shaderID, name.c_str()), x, y, z, w);
}
void Shader::SetVector4(const std::string& name, const ImVec4& vec) const {
    glUniform4f(glGetUniformLocation(shaderID, name.c_str()), vec.x, vec.y, vec.z, vec.w);
}
void Shader::SetMatrix2(const std::string& name, const float* value) const {
    glUniformMatrix2fv(glGetUniformLocation(shaderID, name.c_str()), 1, GL_FALSE, value);
}
void Shader::SetMatrix3(const std::string& name, const float* value) const {
    glUniformMatrix3fv(glGetUniformLocation(shaderID, name.c_str()), 1, GL_FALSE, value);
}
void Shader::SetMatrix4(const std::string& name, const float* value) const {
    glUniformMatrix4fv(glGetUniformLocation(shaderID, name.c_str()), 1, GL_FALSE, value);
}
void Shader::SetSampler2D(const std::string& name, unsigned value) const {
    glUniform1i(glGetUniformLocation(shaderID, name.c_str()), value);
}

void Shader::ShowEditor(bool* p_open) {
    ImGui::PushID(this);

    if (ImGui::Begin("Shader Editor", p_open)) {
        if (ImGui::Button("Build Shader")) {
            vertexCode = vertEditor->GetText();
            fragCode   = fragEditor->GetText();

            glDeleteProgram(shaderID);

            try {
                compile();
            } catch (std::runtime_error& e) {
                ImGui::OpenPopup("Shader Compilation Error");
            }
        }

        ImGui::SameLine();

        ImGui::Checkbox("Auto Compile", &autoCompile);

        if (ImGui::BeginPopupModal("Shader Compilation Error", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Failed to compile shader");

            if (ImGui::Button("OK", ImVec2(120, 0))) {
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }

        bool modified = false;
        if (ImGui::BeginChild("editorTabs", ImVec2(0, -120))) {
            if (ImGui::BeginTabBar("Shader Editor Tabs")) {
                if (ImGui::BeginTabItem("Vertex")) {
                    vertEditor->Render("Vertex Shader Editor");

                    modified |= vertEditor->IsTextChanged();

                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Fragment")) {
                    fragEditor->Render("Fragment Shader Editor");

                    modified |= fragEditor->IsTextChanged();

                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }

            ImGui::EndChild();
        }

        if (autoCompile && modified) {
            vertexCode = vertEditor->GetText();
            fragCode   = fragEditor->GetText();

            glDeleteProgram(shaderID);

            try {
                compile();
            } catch (std::runtime_error& e) {
                
            }
        }

        char* error = (char*)lastError.c_str();
        int size    = lastError.size();

        ImGui::InputTextMultiline("##Last Error", error, size, ImVec2(-1, -1), ImGuiInputTextFlags_ReadOnly);
    }
    ImGui::End();

    ImGui::PopID();
}
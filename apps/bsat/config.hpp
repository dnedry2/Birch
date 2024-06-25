#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <variant>
#include <map>

#include "imgui.h"

class ConfigItem {
public:
    enum class ItemType {
        Int,
        Bool,
        Float,
        Vector2,
        Vector4,
        String
    };

    ConfigItem(const std::string& name, const std::variant<int, bool, float, ImVec2, ImVec4, std::string>& value);

    const std::string& Name() const;
    ItemType Type() const;
    
    int&         Int();
    bool&        Bool();
    float&       Float();
    ImVec2&      Vector2();
    ImVec4&      Vector4();
    std::string& String();

private:
    std::string name;
    ItemType    type;
    std::variant<int, bool, float, ImVec2, ImVec4, std::string> value;
};

class Config {
public:
    Config();
    ~Config();

    void Load(const std::string& path);
    void Save(const std::string& path);

    void AddItem(const std::string& name, const std::variant<int, bool, float, ImVec2, ImVec4, std::string>& value);
    void AddItem(const ConfigItem& item);
    void RemoveItem(const std::string& name);

    ConfigItem& Get(const std::string& name);
    ConfigItem& operator[](const std::string& name);

private:
    std::map<std::string, ConfigItem> items;
};

#endif
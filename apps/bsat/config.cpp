#include "config.hpp"

#include <cstdio>
#include <stdexcept>
#include "logger.hpp"

ConfigItem::ConfigItem(const std::string& name, const std::variant<int, bool, float, ImVec2, ImVec4, std::string>& value)
           : name(name), value(value)
{
    if (std::holds_alternative<int>(value)) {
        type = ItemType::Int;
    } else if (std::holds_alternative<bool>(value)) {
        type = ItemType::Bool;
    } else if (std::holds_alternative<float>(value)) {
        type = ItemType::Float;
    } else if (std::holds_alternative<ImVec2>(value)) {
        type = ItemType::Vector2;
    } else if (std::holds_alternative<ImVec4>(value)) {
        type = ItemType::Vector4;
    } else if (std::holds_alternative<std::string>(value)) {
        type = ItemType::String;
    }
}

const std::string& ConfigItem::Name() const {
    return name;
}
ConfigItem::ItemType ConfigItem::Type() const {
    return type;
}

int& ConfigItem::Int() {
    return std::get<int>(value);
}
bool& ConfigItem::Bool() {
    return std::get<bool>(value);
}
float& ConfigItem::Float() {
    return std::get<float>(value);
}
ImVec2& ConfigItem::Vector2() {
    return std::get<ImVec2>(value);
}
ImVec4& ConfigItem::Vector4() {
    return std::get<ImVec4>(value);
}
std::string& ConfigItem::String() {
    return std::get<std::string>(value);
}


Config::Config() { }
Config::~Config() { }

void Config::AddItem(const std::string& name, const std::variant<int, bool, float, ImVec2, ImVec4, std::string>& value) {
    items.emplace(name, ConfigItem(name, value));
}
void Config::AddItem(const ConfigItem& item) {
    items.emplace(item.Name(), item);
}
void Config::RemoveItem(const std::string& name) {
    items.erase(name);
}

ConfigItem& Config::Get(const std::string& name) {
    try {
        return items.at(name);
    } catch (std::out_of_range& e) {
        DispError("Config", "Failed to get config item: %s", name.c_str());
        throw e;
    }
}
ConfigItem& Config::operator[](const std::string& name) {
    return Get(name);
}

void Config::Save(const std::string& path) {
    FILE* file = fopen(path.c_str(), "w");

    if (!file) {
        throw std::runtime_error("Failed to open config file for writing");
    }

    std::string header = 
    "# This place is not a place of honor... no highly esteemed deed is commemorated here... nothing valued is here.\n"
    "# What is here was dangerous and repulsive to us. This message is a warning about danger.\n"
    "# The danger is unleashed only if you substantially disturb this place physically. This place is best shunned and left uninhabited.\n\n";

    fprintf(file, "%s", header.c_str());

    for (auto& [name, item] : items) {
        switch (item.Type()) {
            case ConfigItem::ItemType::Int:
                fprintf(file, "%s: int = %d\n", name.c_str(), item.Int());
                break;
            case ConfigItem::ItemType::Bool:
                fprintf(file, "%s: bool = %s\n", name.c_str(), item.Bool() ? "true" : "false");
                break;
            case ConfigItem::ItemType::Float:
                fprintf(file, "%s: float = %f\n", name.c_str(), item.Float());
                break;
            case ConfigItem::ItemType::Vector2:
                fprintf(file, "%s: vec2 = %f, %f\n", name.c_str(), item.Vector2().x, item.Vector2().y);
                break;
            case ConfigItem::ItemType::Vector4:
                fprintf(file, "%s: vec4 = %f, %f, %f, %f\n", name.c_str(), item.Vector4().x, item.Vector4().y, item.Vector4().z, item.Vector4().w);
                break;
            case ConfigItem::ItemType::String:
                fprintf(file, "%s: str = %s\n", name.c_str(), item.String().c_str());
                break;
        }
    }

    fclose(file);
}

void Config::Load(const std::string& path) {
    FILE* file = fopen(path.c_str(), "r");

    if (!file) {
        throw std::runtime_error("Failed to open config file for reading");
    }

    items.clear();

    char line[4096];
    while (fgets(line, sizeof(line), file)) {
        // Skip empty lines and comments
        if (line[0] == '\n' || line[0] == '#') {
            continue;
        }

        char* name = strtok(line, ":");
        char* type = strtok(NULL, "=");
        char* value = strtok(NULL, "\n");

        if (strcmp(type, " int ") == 0) {
            items.emplace(name, ConfigItem(name, atoi(value)));
        } else if (strcmp(type, " bool ") == 0) {
            items.emplace(name, ConfigItem(name, strcmp(value, " true") == 0));
        } else if (strcmp(type, " float ") == 0) {
            items.emplace(name, ConfigItem(name, (float)atof(value)));
        } else if (strcmp(type, " vec2 ") == 0) {
            float x = atof(strtok(value, ","));
            float y = atof(strtok(NULL, ","));
            items.emplace(name, ConfigItem(name, ImVec2(x, y)));
        } else if (strcmp(type, " vec4 ") == 0) {
            float x = atof(strtok(value, ","));
            float y = atof(strtok(NULL, ","));
            float z = atof(strtok(NULL, ","));
            float w = atof(strtok(NULL, ","));
            items.emplace(name, ConfigItem(name, ImVec4(x, y, z, w)));
        } else if (strcmp(type, " str ") == 0) {
            items.emplace(name, ConfigItem(name, value));
        }
    }

    fclose(file);
}
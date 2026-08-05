#include <iostream>
#include <optional>
#include <string>
#include <variant>

struct Describe {
    std::string operator()(std::monostate) const { return "empty"; }
    std::string operator()(int value) const { return std::to_string(value); }
    std::string operator()(const std::string& value) const { return value; }
};

int main() {
    std::optional<std::string> nickname = std::nullopt;
    nickname.emplace("Ada");
    std::string display = nickname.value_or("anonymous");
    std::variant<std::monostate, int, std::string> payload = std::monostate{};
    payload = std::string{"ready"};
    bool textual = std::holds_alternative<std::string>(payload);
    const std::string* text = std::get_if<std::string>(&payload);
    std::string rendered = std::visit(Describe{}, payload);
    std::cout << display << ':' << rendered << ':' << textual << ':' << (text ? text->size() : 0) << '\n';
}

#include <iostream>
#include <string>

enum class Access : unsigned {
    none = 0U,
    read = 1U << 0U,
    write = 1U << 1U,
    admin = 1U << 2U
};

constexpr Access operator|(Access left, Access right) {
    return static_cast<Access>(static_cast<unsigned>(left) | static_cast<unsigned>(right));
}

constexpr bool has(Access value, Access flag) {
    return (static_cast<unsigned>(value) & static_cast<unsigned>(flag)) != 0U;
}

int main() {
    Access granted = Access::read | Access::write;
    unsigned raw = static_cast<unsigned>(granted);
    int quota = static_cast<int>(raw) * 6 + 2;
    quota += has(granted, Access::admin) ? 20 : 5;
    bool may_publish = has(granted, Access::write) && quota >= 10;
    const std::string label = may_publish ? "publisher" : "reader";
    std::cout << label << ": " << quota << '\n';
}

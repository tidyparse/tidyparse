#include <algorithm>
#include <cctype>
#include <iostream>
#include <ranges>
#include <string>
#include <string_view>

int main() {
    std::string message{"  Mixed Case: alpha-beta  "};
    message.erase(0, message.find_first_not_of(' '));
    message.erase(message.find_last_not_of(' ') + 1);
    std::ranges::transform(message, message.begin(), [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    std::size_t dash = message.find('-');
    message.replace(dash, 1, " / ");
    std::string_view view{message};
    std::string prefix{view.substr(0, view.find(':'))};
    std::cout << prefix.append(" -> ").append(message) << '\n';
}

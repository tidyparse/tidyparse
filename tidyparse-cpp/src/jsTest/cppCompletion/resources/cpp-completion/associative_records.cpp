#include <iostream>
#include <map>
#include <set>
#include <string>
#include <tuple>

int main() {
    using Record = std::tuple<int, std::string, double>;
    std::map<int, Record> records;
    records.emplace(7, Record{7, "Noor", 88.5});
    records.try_emplace(3, 3, "Mira", 91.5);
    std::set<std::string> names;
    for (const auto& [key, record] : records) names.insert(std::get<1>(record));
    auto [id, name, score] = records.at(7);
    auto lower = records.lower_bound(4);
    std::cout << id << ':' << name << ':' << score << ':' << std::get<1>(lower->second) << ':' << names.size() << '\n';
}

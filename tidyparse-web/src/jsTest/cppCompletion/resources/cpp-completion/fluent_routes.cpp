#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

class Vehicle {
    std::string label_;
public:
    explicit Vehicle(std::string label) : label_(std::move(label)) {}
    virtual ~Vehicle() = default;
    const std::string& label() const { return label_; }
    virtual int range() const = 0;
};

class Bicycle final : public Vehicle {
    int gears_;
public:
    Bicycle(std::string label, int gears) : Vehicle(std::move(label)), gears_(gears) {}
    int range() const override { return gears_ * 6; }
};

class Route {
    std::string name_;
    std::vector<std::string> stops_;
public:
    Route(std::string name, std::vector<std::string> stops)
        : name_(std::move(name)), stops_(std::move(stops)) {}

    std::string summary() const {
        std::ostringstream out;
        out << name_ << " (" << stops_.size() << " stops)";
        return out.str();
    }
};

class RouteBuilder {
    std::string name_ = "untitled";
    std::vector<std::string> stops_;
    std::size_t limit_ = 8;
public:
    RouteBuilder& named(std::string name) {
        name_ = std::move(name);
        return *this;
    }

    RouteBuilder& add(std::string stop) {
        if (stops_.size() < limit_) stops_.push_back(std::move(stop));
        return *this;
    }

    RouteBuilder& with_limit(std::size_t limit) {
        limit_ = limit;
        return *this;
    }

    Route build() const { return Route{name_, stops_}; }
};

void print_trip(const Vehicle& vehicle, const Route& route, int repeats) {
    for (int i = 0; i < repeats; ++i)
        std::cout << vehicle.label() << ": " << route.summary() << '\n';
}

int main() {
    RouteBuilder builder;
    builder.named("Harbor Loop").with_limit(4).add("Museum").add("Market").add("Pier");
    Route route = builder.build();

    auto bicycle = std::make_unique<Bicycle>("Comet", 7);
    Vehicle* raw_vehicle = bicycle.get();
    const Vehicle& vehicle_ref = *raw_vehicle;
    std::vector<std::unique_ptr<Vehicle>> fleet;

    print_trip(vehicle_ref, route, 2);
    fleet.push_back(std::move(bicycle));
    fleet.push_back(std::make_unique<Bicycle>("Nova", 12));
    std::sort(fleet.begin(), fleet.end(), [](const auto& left, const auto& right) { return left->range() < right->range(); });

    int total_range = 0;
    for (const auto& vehicle : fleet)
        total_range += vehicle->range();
    std::cout << route.summary().append("; total range = ") << total_range << '\n';
}

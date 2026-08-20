#include <iostream>
#include <memory>

class Shape {
public:
    virtual ~Shape() = default;
    virtual double area() const = 0;
};

class Circle final : public Shape {
    double radius_;
public:
    explicit Circle(double radius) : radius_(radius) {}
    double radius() const { return radius_; }
    double area() const override { return radius_ * radius_ * 3.14159; }
};

int main() {
    Circle circle{3.0};
    Shape* base = &circle;
    Shape& alias = *base;
    Circle* recovered = dynamic_cast<Circle*>(base);
    const Shape* readonly = std::addressof(alias);
    double scaled = static_cast<double>(recovered->radius()) * 1.5;
    const void* erased = static_cast<const void*>(readonly);
    std::cout << alias.area() << ' ' << scaled << ' ' << (erased != nullptr) << '\n';
}

#include <cstdint>
#include <iostream>
#include <memory>
#include <typeinfo>

class Shape {
public:
    virtual ~Shape() = default;
    virtual double area() const = 0;
    virtual void scale(double factor) = 0;
};

class Circle final : public Shape {
    double radius_;
public:
    explicit Circle(double radius) : radius_(radius) {}
    double area() const override { return 3.14159 * radius_ * radius_; }
    void scale(double factor) override { radius_ *= factor; }
    double radius() const { return radius_; }
};

int main() {
    std::unique_ptr<Shape> shape = std::make_unique<Circle>(3.0);
    Shape* base = shape.get();
    Circle* circle = dynamic_cast<Circle*>(base);
    const Shape& view = *base;
    double radius = circle ? circle->radius() : 0.0;
    Shape& mutable_view = const_cast<Shape&>(view);
    mutable_view.scale(1.5);
    std::uintptr_t address = reinterpret_cast<std::uintptr_t>(base);
    std::cout << typeid(view).name() << ' ' << static_cast<int>(view.area()) << ' ' << radius << ' ' << address << '\n';
}

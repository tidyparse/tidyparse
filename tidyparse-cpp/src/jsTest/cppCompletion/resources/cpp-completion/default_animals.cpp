#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class Animal {
    std::string name_;
public:
    explicit Animal(std::string name) : name_(std::move(name)) {}
    virtual ~Animal() = default;
    const std::string& name() const { return name_; }
    virtual std::string speak() const = 0;
};

class Dog final : public Animal {
    int age_;
public:
    Dog(std::string name, int age) : Animal(std::move(name)), age_(age) {}
    std::string speak() const override { return age_ < 2 ? "yip" : "woof"; }
};

class Cat final : public Animal {
public:
    using Animal::Animal;
    std::string speak() const override { return "meow"; }
};

void introduce(const Animal& animal, int times = 1) {
    std::cout << animal.name() << ": ";
    for (int i = 0; i < times; ++i)
        std::cout << animal.speak() << (i + 1 == times ? '\n' : ' ');
}

int main() {
    Dog dog{"Rex", 4};
    Cat cat{"Luna"};

    introduce(dog);
    introduce(cat, 3);

    std::vector<std::unique_ptr<Animal>> animals;
    animals.push_back(std::make_unique<Dog>("Pip", 1));
    animals.push_back(std::make_unique<Cat>("Milo"));

    for (const auto& animal : animals)
        introduce(*animal, 2);
}

#include <functional>
#include <iostream>
#include <utility>
#include <vector>

class Transformer {
public:
    virtual ~Transformer() = default;
    virtual int apply(int value) const = 0;
};

class Pipeline final : public Transformer {
    std::vector<std::function<int(int)>> steps_;
public:
    Pipeline& then(std::function<int(int)> step) {
        steps_.push_back(std::move(step));
        return *this;
    }

    int apply(int value) const override {
        for (const auto& step : steps_) value = step(value);
        return value;
    }
};

int main() {
    Pipeline pipeline;
    int offset = 3;
    pipeline.then([offset](int value) { return value + offset; }).then([](int value) { return value * 2; });
    const Transformer& callable = pipeline;
    std::cout << callable.apply(5) << '\n';
}

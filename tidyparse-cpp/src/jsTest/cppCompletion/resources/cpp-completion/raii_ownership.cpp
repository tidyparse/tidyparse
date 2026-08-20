#include <iostream>
#include <memory>
#include <string>
#include <utility>

class Connection {
    std::string endpoint_;
public:
    explicit Connection(std::string endpoint) : endpoint_(std::move(endpoint)) {}
    const std::string& endpoint() const { return endpoint_; }
    void retarget(std::string endpoint) { endpoint_ = std::move(endpoint); }
};

int main() {
    auto primary = std::make_unique<Connection>("db-primary");
    std::unique_ptr<Connection> standby = std::move(primary);
    standby->retarget("db-standby");
    auto shared = std::make_shared<Connection>("cache");
    std::weak_ptr<Connection> observer = shared;
    std::shared_ptr<Connection> pinned = observer.lock();
    std::unique_ptr<int[]> counters = std::make_unique<int[]>(3);
    counters[1] = pinned ? 2 : 0;
    std::cout << standby->endpoint() << ':' << pinned->endpoint() << ':' << counters[1] << '\n';
}

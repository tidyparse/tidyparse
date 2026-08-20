#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class Node {
    std::string text_;
public:
    explicit Node(std::string text) : text_(std::move(text)) {}
    virtual ~Node() = default;
    virtual std::string render() const = 0;
    const std::string& text() const { return text_; }
};

class Heading final : public Node {
    int level_;
public:
    Heading(std::string text, int level) : Node(std::move(text)), level_(level) {}
    std::string render() const override { return std::string(static_cast<std::size_t>(level_), '#') + " " + text(); }
};

class Paragraph final : public Node {
public:
    using Node::Node;
    std::string render() const override { return text(); }
};

class Document {
    std::string title_ = "draft";
    std::vector<std::shared_ptr<Node>> nodes_;
public:
    Document& titled(std::string title) {
        title_ = std::move(title);
        return *this;
    }

    Document& append(std::shared_ptr<Node> node) {
        nodes_.push_back(std::move(node));
        return *this;
    }

    const std::shared_ptr<Node>& at(std::size_t index) const { return nodes_.at(index); }
    std::size_t size() const { return nodes_.size(); }
    const std::string& title() const { return title_; }
};

void inspect(const Document& document, const Node* selected, std::size_t index, bool verbose) {
    std::cout << document.title() << '[' << index << "] " << selected->render();
    if (verbose) std::cout << " (" << document.size() << " nodes)";
    std::cout << '\n';
}

int main() {
    auto heading = std::make_shared<Heading>("Pointers and References", 2);
    auto paragraph = std::make_shared<Paragraph>("Ownership remains explicit.");
    Document document;

    document.titled("Memory Notes").append(heading).append(paragraph).append(std::make_shared<Paragraph>("Chains stay readable."));
    const std::shared_ptr<Node>& selected_owner = document.at(1);
    Node* selected = selected_owner.get();
    const Node& selected_ref = *selected;

    inspect(document, &selected_ref, 1, true);
    std::weak_ptr<Node> observer = heading;
    std::shared_ptr<Node> locked = observer.lock();
    std::string preview = locked ? locked->render() : std::string{"expired"};
    std::cout << preview.append(" -> ").append(document.at(2)->render()) << '\n';
}

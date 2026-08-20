#include <algorithm>
#include <deque>
#include <iostream>
#include <iterator>
#include <list>
#include <numeric>
#include <vector>

int main() {
    std::vector<int> values{7, 2, 9, 4, 5, 8};
    std::deque<int> queue(values.begin(), values.end());
    std::transform(values.begin(), values.end(), values.begin(), [](int value) { return value * value; });
    auto first_even = std::find_if(values.begin(), values.end(), [](int value) { return value % 2 == 0; });
    std::rotate(values.begin(), first_even, values.end());
    std::list<int> ordered(values.cbegin(), values.cend());
    ordered.sort();
    std::copy_if(ordered.cbegin(), ordered.cend(), std::back_inserter(queue), [](int value) { return value > 20; });
    std::cout << std::accumulate(queue.cbegin(), queue.cend(), 0) << '\n';
}

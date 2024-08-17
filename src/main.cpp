#include "tensor/tensor_create.h"
#include <chrono>
#include <iostream>
#include <vector>

using namespace tensor;
int main() {
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();

  auto size = std::vector<int>{1000, 1000};
  auto x = rand_n(size);
  auto y = rand_n(size);
  for (int i = 0; i < 10; i++) {
    auto result = x + y;
    result = x - y;
    result = x / y;
    result = x * y;
    result = x & y;
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time = "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     begin)
                   .count()
            << "[ms]" << std::endl;
}

#include "tensor.h"
#include "variable/variable.h"
#include <numeric>
#include <random>

using namespace tensor;

namespace tensor {

Tensor zeros(std::vector<int> shape) {
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  auto data = std::vector<float>(size);
  return Tensor(data, shape, "zeros");
}

Tensor zeros_like(Tensor tensor) { return zeros(tensor.shape()); }

Tensor one_hot(int num, int num_classes) {
  auto data = std::vector<float>(num_classes, 0);
  data[num] = 1;

  return Tensor(data, {num_classes}, "");
}

Tensor rand_n(std::vector<int> shape) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> distr(0.0f, 1.0f);
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  auto data = std::vector<float>(size);
  for (int i = 0; i < size; i++) {
    data[i] = distr(gen);
  }
  return Tensor(data, shape, "rand_n");
}

Tensor uniform(std::vector<int> shape, float low, float high) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distr(low, high);
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  auto data = std::vector<float>(size);
  for (int i = 0; i < size; i++) {
    data[i] = distr(gen);
  }
  return Tensor(data, shape, "uniform");
}

} // namespace tensor

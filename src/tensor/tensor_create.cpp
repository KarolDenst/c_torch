#include "tensor.h"
#include <numeric>
#include <random>

using namespace tensor;

namespace tensor {

Tensor zeros(std::vector<int> shape, bool is_tmp) {
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  auto data = std::vector<float>(size);
  return Tensor(data, shape, "zeros", is_tmp);
}

Tensor zeros_like(const Tensor &tensor, bool is_tmp) {
  return zeros(tensor.shape, is_tmp);
}

Tensor one_hot(int num, int num_classes, bool is_tmp) {
  auto data = std::vector<float>(num_classes, 0);
  data[num] = 1;

  return Tensor(data, {num_classes}, "", is_tmp);
}

Tensor rand_n(std::vector<int> shape, bool is_tmp) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> distr(0.0f, 1.0f);
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  auto data = std::vector<float>(size);
  for (int i = 0; i < size; i++) {
    data[i] = distr(gen);
  }
  return Tensor(data, shape, "rand_n", is_tmp);
}

Tensor uniform(std::vector<int> shape, float low, float high,
               bool is_tmp = true) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distr(low, high);
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  auto data = std::vector<float>(size);
  for (int i = 0; i < size; i++) {
    data[i] = distr(gen);
  }
  return Tensor(data, shape, "uniform", is_tmp);
}

} // namespace tensor

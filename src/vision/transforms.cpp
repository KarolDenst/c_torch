#include "transforms.h"
#include "../tensor/tensor.h"
#include <cassert>
#include <random>

namespace vision {
namespace transforms {

tensor::Tensor resize(tensor::Tensor tensor, int height, int width) {
  throw std::runtime_error("Not implemented");
}

tensor::Tensor random_horizontal_flip(tensor::Tensor tensor, float p) {
  assert(tensor.shape().size() == 2);
  assert(p >= 0.0f && p <= 1.0f);

  std::random_device rd;
  std::minstd_rand gen(rd());
  std::uniform_real_distribution<> distr(0, 1);
  if (distr(gen) < p) {
    auto data = std::vector<float>(tensor.data().size());
    auto result = tensor::Tensor(data, tensor.shape());

    int width = tensor.shape(1);
    for (int i = 0; i < tensor.shape(0); i++) {
      for (int j = 0; j < tensor.shape(1); j++) {
        result.get({i, j}) = tensor.get({i, width - j - 1});
        result.get({i, width - j - 1}) = tensor.get({i, j});
      }
    }
    return result;
  } else {
    return tensor;
  }
}

tensor::Tensor random_vertical_flip(tensor::Tensor tensor, float p) {
  assert(tensor.shape().size() == 2);
  assert(p >= 0.0f && p <= 1.0f);

  std::random_device rd;
  std::minstd_rand gen(rd());
  std::uniform_real_distribution<> distr(0, 1);
  if (distr(gen) < p) {
    auto data = std::vector<float>(tensor.data().size());
    auto result = tensor::Tensor(data, tensor.shape());

    int height = tensor.shape(0);
    for (int i = 0; i < tensor.shape(0); i++) {
      for (int j = 0; j < tensor.shape(1); j++) {
        result.get({i, j}) = tensor.get({height - i - 1, j});
        result.get({height - i - 1, j}) = tensor.get({i, j});
      }
    }
    return result;
  } else {
    return tensor;
  }
}

tensor::Tensor random_rotation(tensor::Tensor tensor, float degrees) {
  assert(tensor.shape().size() == 2);

  auto data = std::vector<float>(tensor.data().size(), 0);
  auto result = tensor::Tensor(data, tensor.shape());

  std::random_device rd;
  std::minstd_rand gen(rd());
  std::uniform_real_distribution<> distr(-degrees, degrees);
  float rotation = distr(gen);

  int height = tensor.shape(0);
  int width = tensor.shape(1);
  int cos = std::cos(rotation);
  int sin = std::sin(rotation);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int x = i - height / 2;
      int y = j - width / 2;
      int nx = x * std::cos(rotation) - y * std::sin(rotation);
      int ny = x * std::sin(rotation) + y * std::cos(rotation);
      int x_result = nx + height / 2;
      int y_result = ny + width / 2;
      result.get({x_result, y_result}) = tensor.get({i, j});
    }
  }

  return result;
}

} // namespace transforms
} // namespace vision

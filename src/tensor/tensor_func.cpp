#include "tensor.h"
#include <cmath>
#include <numeric>
#include <optional>

namespace tensor {

Tensor tanh(Tensor *tensor) {
  std::vector<float> data;
  for (int i = 0; i < tensor->data.size(); i++) {
    data.push_back(std::tanh(tensor->data[i]));
  }
  auto prev = std::vector<Tensor *>{tensor};
  auto out =
      Tensor(data, tensor->shape, prev, "tanh(" + tensor->name + ")", true);
  auto backward = [tensor, &out]() {
    for (int i = 0; i < out.grad.size(); i++) {
      tensor->grad[i] += out.grad[i] * (1 - out.data[i] * out.data[i]);
    }
  };
  out.back = backward;
  return out;
}

Tensor exp(Tensor *tensor) {
  std::vector<float> data;
  for (int i = 0; i < tensor->data.size(); i++) {
    data.push_back(std::exp(tensor->data[i]));
  }
  auto prev = std::vector<Tensor *>{tensor};
  auto out =
      Tensor(data, tensor->shape, prev, "exp(" + tensor->name + ")", true);
  auto backward = [tensor, &out]() {
    for (int i = 0; i < out.grad.size(); i++) {
      tensor->grad[i] += out.grad[i] * out.data[i];
    }
  };
  out.back = backward;
  return out;
}

Tensor log(Tensor *tensor) {
  std::vector<float> data;
  for (int i = 0; i < tensor->data.size(); i++) {
    data.push_back(std::log(tensor->data[i]));
  }
  auto prev = std::vector<Tensor *>{tensor};
  auto out =
      Tensor(data, tensor->shape, prev, "log(" + tensor->name + ")", true);
  auto backward = [tensor, &out]() {
    for (int i = 0; i < out.grad.size(); i++) {
      tensor->grad[i] += out.grad[i] * 1.0 / (tensor->data[i] + EPS);
    }
  };
  out.back = backward;
  return out;
}

Tensor sum(Tensor *tensor, std::optional<int> dim) {
  std::vector<float> data = {
      std::accumulate(tensor->data.begin(), tensor->data.end(), 0.0f)};
  auto prev = std::vector<Tensor *>{tensor};
  auto out = Tensor(data, {1}, prev, "sum(" + tensor->name + ")", true);
  auto backward = [tensor, &out]() {
    for (int i = 0; i < out.prev[0]->data.size(); i++) {
      tensor->grad[i] += out.grad[0];
    }
  };
  out.back = backward;
  return out;
}

} // namespace tensor

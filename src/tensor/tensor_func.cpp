#include "tensor.h"
#include <cassert>
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
  assert(dim < tensor->shape.size());
  auto shape = std::vector<int>(3, 1);
  if (!dim.has_value()) {
    shape = {1, static_cast<int>(tensor->data.size()), 1};
  } else {
    for (int i = 0; i < tensor->shape.size(); i++) {
      if (i < dim.value())
        shape[0] *= tensor->shape[i];
      else if (i == dim.value())
        shape[1] = tensor->shape[i];
      else
        shape[2] *= tensor->shape[i];
    }
  }

  auto data = std::vector<float>(shape[0] * shape[2], 0);
  for (int i = 0; i < shape[0]; i++) {
    for (int j = 0; j < shape[1]; j++) {
      for (int k = 0; k < shape[2]; k++) {
        int index = i * shape[1] * shape[2] + j * shape[2] + k;
        data[i * shape[2] + k] += tensor->data[index];
      }
    }
  }
  // TODO: fix shape for tensors with 1D and 2D shapes
  auto prev = std::vector<Tensor *>{tensor};
  auto out = Tensor(data, shape, prev, "sum(" + tensor->name + ")", true);

  // float back_mul = 1;
  // if (dim.has_value()) {
  //   back_mul = tensor->shape[dim.value()];
  // }
  auto backward = [tensor, &out]() {
    for (int i = 0; i < out.prev[0]->data.size(); i++) {
      tensor->grad[i] += out.grad[0];
    }
  };
  out.back = backward;
  return out;
}

} // namespace tensor

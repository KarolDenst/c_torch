#ifndef VARIABLE_FUNC_H
#define VARIABLE_FUNC_H

#include "variable.h"
#include <optional>

namespace variable {

template <Numeric DType = float>
std::shared_ptr<Variable<DType>>
tanh(std::shared_ptr<Variable<DType>> variable) {
  std::vector<DType> data;
  for (int i = 0; i < variable->data.size(); i++) {
    data.push_back(std::tanh(variable->data[i]));
  }

  auto prev = std::vector<std::shared_ptr<Variable<DType>>>{variable};
  auto out = std::make_shared<Variable<DType>>(data, variable->shape, prev,
                                               "tanh(" + variable->name + ")");
  auto backward = [variable, out]() {
    for (int i = 0; i < out->grad.size(); i++) {
      variable->grad[i] += out->grad[i] * (1 - out->data[i] * out->data[i]);
    }
  };
  out->back = backward;
  return out;
}

template <Numeric DType = float>
std::shared_ptr<Variable<DType>>
relu(std::shared_ptr<Variable<DType>> variable) {
  auto data = std::vector<DType>(variable->data.size());
  for (int i = 0; i < variable->data.size(); i++) {
    data[i] = std::max(0.0f, variable->data[i]);
  }

  auto prev = std::vector<std::shared_ptr<Variable<DType>>>{variable};
  auto out = std::make_shared<Variable<DType>>(data, variable->shape, prev,
                                               "ReLU(" + variable->name + ")");
  auto backward = [variable, out]() {
    for (int i = 0; i < out->grad.size(); i++) {
      if (out->data[i] > 0) {
        variable->grad[i] += out->grad[i];
      }
    }
  };
  out->back = backward;
  return out;
}

template <Numeric DType = float>
std::shared_ptr<Variable<DType>>
exp(std::shared_ptr<Variable<DType>> variable) {
  std::vector<DType> data;
  for (int i = 0; i < variable->data.size(); i++) {
    data.push_back(std::exp(variable->data[i]));
  }

  auto prev = std::vector<std::shared_ptr<Variable<DType>>>{variable};
  auto out = std::make_shared<Variable<DType>>(data, variable->shape, prev,
                                               "exp(" + variable->name + ")");

  auto backward = [variable, out]() {
    for (int i = 0; i < out->grad.size(); i++) {
      variable->grad[i] += out->grad[i] * out->data[i];
    }
  };
  out->back = backward;
  return out;
}

template <Numeric DType = float>
std::shared_ptr<Variable<DType>>
log(std::shared_ptr<Variable<DType>> variable) {
  std::vector<DType> data;
  for (int i = 0; i < variable->data.size(); i++) {
    data.push_back(std::log(variable->data[i]));
  }

  auto prev = std::vector<std::shared_ptr<Variable<DType>>>{variable};
  auto out = std::make_shared<Variable<DType>>(data, variable->shape, prev,
                                               "log(" + variable->name + ")");

  auto backward = [variable, out]() {
    for (int i = 0; i < out->grad.size(); i++) {
      variable->grad[i] += out->grad[i] * 1.0 / (variable->data[i]);
    }
  };
  out->back = backward;
  return out;
}

template <Numeric DType = float>
std::shared_ptr<Variable<DType>> sum(std::shared_ptr<Variable<DType>> variable,
                                     std::optional<int> dim, bool keepdim) {
  assert(dim < variable->shape.size());
  auto shape = std::vector<int>(3, 1);
  if (!dim.has_value()) {
    shape = {1, static_cast<int>(variable->data.size()), 1};
  } else {
    for (int i = 0; i < variable->shape.size(); i++) {
      if (i < dim.value())
        shape[0] *= variable->shape[i];
      else if (i == dim.value())
        shape[1] = variable->shape[i];
      else
        shape[2] *= variable->shape[i];
    }
  }

  auto data = std::vector<DType>(shape[0] * shape[2], 0);
  for (int i = 0; i < shape[0]; i++) {
    for (int j = 0; j < shape[1]; j++) {
      for (int k = 0; k < shape[2]; k++) {
        int index = i * shape[1] * shape[2] + j * shape[2] + k;
        data[i * shape[2] + k] += variable->data[index];
      }
    }
  }

  std::vector<int> out_shape;
  if (dim.has_value()) {
    out_shape = variable->shape;
    if (!keepdim)
      out_shape.erase(out_shape.begin() + dim.value());
    else
      out_shape[dim.value()] = 1;
  } else {
    out_shape = {1};
  }

  auto prev = std::vector<std::shared_ptr<Variable<DType>>>{variable};
  auto out = std::make_shared<Variable<DType>>(data, out_shape, prev,
                                               "sum(" + variable->name + ")");

  auto backward = [variable, out]() {
    for (int i = 0; i < out->prev[0]->data.size(); i++) {
      variable->grad[i] += out->grad[0];
    }
  };
  out->back = backward;
  return out;
}

template <Numeric DType = float>
std::shared_ptr<Variable<DType>>
mean(std::shared_ptr<Variable<DType>> variable) {
  auto number = static_cast<DType>(variable->data.size());
  auto data = std::vector<DType>(
      1, std::accumulate(variable->data.begin(), variable->data.end(), 0.0) /
             number);

  auto prev = std::vector<std::shared_ptr<Variable<DType>>>{variable};
  auto out = std::make_shared<Variable<DType>>(data, std::vector<int>{1}, prev,
                                               "mean(" + variable->name + ")");

  auto backward = [variable, out]() {
    for (int i = 0; i < variable->grad.size(); i++) {
      variable->grad[i] += out->grad[0] / variable->data.size();
    }
  };
  out->back = backward;
  return out;
}

} // namespace variable

#endif // VARIABLE_FUNC_H

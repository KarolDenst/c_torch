#include "variable.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace variable {

Variable::Variable(std::vector<float> data, std::vector<int> shape,
                   std::string name)
    : data(data), shape(shape), strides(compute_strides(shape)), name(name),
      prev(std::vector<std::shared_ptr<Variable>>()),
      grad(std::vector<float>(data.size())), back([]() {}) {}

Variable::Variable(std::vector<float> data, std::vector<int> shape,
                   std::vector<std::shared_ptr<Variable>> prev,
                   std::string name)
    : data(data), shape(shape), strides(compute_strides(shape)), name(name),
      prev(prev), grad(std::vector<float>(data.size())), back([]() {}) {}

void Variable::print(bool print_prev) {
  std::cout << name;
  std::cout << std::endl << "Data: ";
  for (int i = 0; i < std::min(static_cast<int>(this->data.size()), 10); i++) {
    std::cout << this->data[i] << " ";
  }
  if (this->grad.size() > 10) {
    std::cout << "...";
  }
  std::cout << std::endl << "Shape: ";
  for (auto i : this->shape) {
    std::cout << i << " ";
  }
  std::cout << std::endl << "Grad: ";
  for (int i = 0; i < std::min(static_cast<int>(this->grad.size()), 10); i++) {
    std::cout << this->grad[i] << " ";
  }
  if (this->grad.size() > 10) {
    std::cout << "...";
  }
  if (print_prev) {
    std::cout << "\nPrev: \n";
    for (auto &t : this->prev) {
      t->print(print_prev);
    }
  }
  std::cout << std::endl;
  std::cout << std::endl;
}

std::shared_ptr<Variable> Variable::add(std::shared_ptr<Variable> first,
                                        std::shared_ptr<Variable> second) {
  auto front = [](Variable *first, Variable *second, Variable *out, int i,
                  int j,
                  int k) { out->data[k] = first->data[i] + second->data[j]; };
  auto back = [](Variable *first, Variable *second, Variable *out, int i, int j,
                 int k) {
    first->grad[i] += out->grad[k];
    second->grad[j] += out->grad[k];
  };
  return transform(first, second, front, back, "+");
}

std::shared_ptr<Variable> Variable::sub(std::shared_ptr<Variable> first,
                                        std::shared_ptr<Variable> second) {
  auto front = [](Variable *first, Variable *second, Variable *out, int i,
                  int j,
                  int k) { out->data[k] = first->data[i] - second->data[j]; };
  auto back = [](Variable *first, Variable *second, Variable *out, int i, int j,
                 int k) {
    first->grad[i] += out->grad[k];
    second->grad[j] -= out->grad[k];
  };
  return transform(first, second, front, back, "-");
}

std::shared_ptr<Variable> Variable::mul(std::shared_ptr<Variable> first,
                                        std::shared_ptr<Variable> second) {
  auto front = [](Variable *first, Variable *second, Variable *out, int i,
                  int j,
                  int k) { out->data[k] = first->data[i] * second->data[j]; };
  auto back = [](Variable *first, Variable *second, Variable *out, int i, int j,
                 int k) {
    first->grad[i] += second->data[j] * out->grad[k];
    second->grad[j] += first->data[i] * out->grad[k];
  };
  return transform(first, second, front, back, "*");
}

std::shared_ptr<Variable> Variable::div(std::shared_ptr<Variable> first,
                                        std::shared_ptr<Variable> second) {
  auto front = [](Variable *first, Variable *second, Variable *out, int i,
                  int j,
                  int k) { out->data[k] = first->data[i] / second->data[j]; };
  auto back = [](Variable *first, Variable *second, Variable *out, int i, int j,
                 int k) {
    first->grad[i] += 1.0 / (second->data[j]) * out->grad[k];
    second->grad[j] +=
        -first->data[i] / (second->data[j] * second->data[j]) * out->grad[k];
  };
  return transform(first, second, front, back, "/");
}

std::shared_ptr<Variable> Variable::mat_mul(std::shared_ptr<Variable> first,
                                            std::shared_ptr<Variable> second) {
  assert(first->shape.back() == second->shape.front());
  auto shape1 = std::vector<int>{
      static_cast<int>(first->data.size() / first->shape.back()),
      first->shape.back()};
  auto shape2 = std::vector<int>{
      second->shape.front(),
      static_cast<int>(second->data.size() / second->shape.front())};

  auto shape = std::vector<int>(first->shape.size() + second->shape.size() - 2);
  for (int i = 0; i < shape.size(); i++) {
    if (i < first->shape.size() - 1)
      shape[i] = first->shape[i];
    else {
      int j = i - first->shape.size() + 2;
      shape[i] = second->shape[j];
    }
  }
  auto data = std::vector<float>(shape1[0] * shape2[1], 0);
  for (int i = 0; i < shape1[0]; i++) {
    for (int k = 0; k < shape1[1]; k++) {
      for (int j = 0; j < shape2[1]; j++) {
        data[i * shape2[1] + j] +=
            first->data[i * shape1[1] + k] * second->data[k * shape2[1] + j];
      }
    }
  }
  auto prev = std::vector<std::shared_ptr<Variable>>{first, second};
  auto out = std::make_shared<Variable>(data, shape, prev,
                                        first->name + " & " + second->name);

  auto backward = [out, first, second, shape1, shape2]() {
    for (int i = 0; i < shape1[0]; i++) {
      for (int k = 0; k < shape1[1]; k++) {
        for (int j = 0; j < shape2[1]; j++) {
          first->grad[i * shape1[1] + k] +=
              out->grad[i * shape2[1] + j] * second->data[k * shape2[1] + j];
        }
      }
    }

    for (int i = 0; i < shape1[0]; i++) {
      for (int k = 0; k < shape1[1]; k++) {
        for (int j = 0; j < shape2[1]; j++) {
          second->grad[k * shape2[1] + j] +=
              first->data[i * shape1[1] + k] * out->grad[i * shape2[1] + j];
        }
      }
    }
  };
  out->back = backward;

  return out;
}

void Variable::view(std::vector<int> shape) {
  auto dim1 =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  auto dim2 = std::accumulate(this->shape.begin(), this->shape.end(), 1,
                              std::multiplies<int>());
  assert(dim1 == dim2);
  this->shape = shape;
}

void Variable::backward() {
  auto topo = std::vector<Variable *>();
  auto visited = std::unordered_set<Variable *>();
  std::function<void(Variable *)> build_topo = [&](Variable *t) {
    if (visited.find(t) != visited.end()) {
      return;
    }
    visited.insert(t);
    for (const auto &p : t->prev) {
      build_topo(p.get());
    }
    topo.push_back(t);
  };

  for (int i = 0; i < this->grad.size(); i++) {
    this->grad[i] = 1;
  }
  build_topo(this);
  std::reverse(topo.begin(), topo.end());
  for (Variable *t : topo) {
    t->back();
  }
}

std::vector<int> Variable::compute_strides(std::vector<int> shape) {
  std::vector<int> strides(shape.size());
  strides[shape.size() - 1] = 1;
  for (int i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return strides;
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
Variable::compute_broadcast_strides(Variable &first, Variable &second) {
  std::vector<int> out_shape = std::vector<int>();
  std::vector<int> shape1 = first.shape;
  std::vector<int> shape2 = second.shape;
  while (shape1.size() < shape2.size()) {
    shape1.insert(shape1.begin(), 1);
  }
  while (shape2.size() < shape1.size()) {
    shape2.insert(shape2.begin(), 1);
  }
  for (int i = 0; i < shape1.size(); i++) {
    if (shape1[i] != shape2[i] && shape1[i] != 1 && shape2[i] != 1) {
      for (int i = 0; i < first.shape.size(); i++)
        std::cout << first.shape[i] << " ";
      std::cout << std::endl;
      for (int i = 0; i < second.shape.size(); i++)
        std::cout << second.shape[i] << " ";
      std::cout << std::endl;
      throw std::runtime_error("Shape missmatch");
    }
    if (shape1[i] == 1) {
      out_shape.push_back(shape2[i]);
    } else {
      out_shape.push_back(shape1[i]);
    }
  }

  std::vector<int> s1 = compute_strides(shape1);
  std::vector<int> s2 = compute_strides(shape2);
  std::vector<int> stride1 = std::vector<int>(s1.size());
  std::vector<int> stride2 = std::vector<int>(s2.size());
  for (int i = 0; i < s1.size(); i++) {
    if (shape1[i] == shape2[i]) {
      stride1[i] = s1[i];
      stride2[i] = s2[i];
    } else if (shape1[i] == 1) {
      stride1[i] = 0;
      stride2[i] = s2[i];
    } else if (shape2[i] == 1) {
      stride1[i] = s1[i];
      stride2[i] = 0;
    } else {
      throw std::runtime_error("Invalid shape");
    }
  }

  return std::make_tuple(out_shape, stride1, stride2);
}

std::shared_ptr<Variable> Variable::transform(
    std::shared_ptr<Variable> first, std::shared_ptr<Variable> second,
    void (*front)(Variable *, Variable *, Variable *, int, int, int),
    void (*back)(Variable *, Variable *, Variable *, int, int, int),
    std::string name) {
  auto tuple = compute_broadcast_strides(*first, *second);
  auto out_shape = std::get<0>(tuple);
  auto stride1 = std::get<1>(tuple);
  auto stride2 = std::get<2>(tuple);

  auto data_size = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                   std::multiplies<int>());
  auto prev = std::vector<std::shared_ptr<Variable>>{first, second};
  auto out =
      std::make_shared<Variable>(std::vector<float>(data_size), out_shape, prev,
                                 first->name + name + second->name);

  transform_rec(0, 0, 0, 0, out.get(), first.get(), second.get(), stride1,
                stride2, front);

  auto backward = [first, second, out, back]() {
    auto tuple = compute_broadcast_strides(*first, *second);
    auto stride1 = std::get<1>(tuple);
    auto stride2 = std::get<2>(tuple);
    transform_rec(0, 0, 0, 0, out.get(), first.get(), second.get(), stride1,
                  stride2, back);
  };
  out->back = backward;

  return out;
}

void Variable::transform_rec(int dim, int offset1, int offset2, int index,
                             Variable *out, Variable *first, Variable *second,
                             std::vector<int> &stride1,
                             std::vector<int> &stride2,
                             void (*func)(Variable *, Variable *, Variable *,
                                          int, int, int)) {
  if (dim >= out->shape.size()) {
    func(first, second, out, offset1, offset2, index);
    return;
  }
  int s1 = 0, s2 = 0;
  for (int i = 0; i < out->shape[dim]; i++) {
    transform_rec(dim + 1, offset1 + s1, offset2 + s2, index, out, first,
                  second, stride1, stride2, func);
    index += out->strides[dim];
    s1 += stride1[dim];
    s2 += stride2[dim];
  }
};
} // namespace variable

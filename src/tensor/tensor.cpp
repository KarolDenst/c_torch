#include "tensor.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace tensor {

Tensor::Tensor(std::vector<float> data, std::vector<int> shape,
               std::string name, bool is_tmp)
    : data(data), shape(shape), name(name), is_tmp(is_tmp),
      prev(std::vector<Tensor *>()), grad(std::vector<float>(data.size())),
      back([]() {}) {}

Tensor::Tensor(std::vector<float> data, std::vector<int> shape,
               std::vector<Tensor *> prev, std::string name, bool is_tmp)
    : data(data), shape(shape), name(name), is_tmp(is_tmp), prev(prev),
      grad(std::vector<float>(data.size())), back([]() {}) {}

void Tensor::print(bool print_prev) {
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

template <std::size_t N> float &Tensor::get_data(const int (&indices)[N]) {
  static_assert(N == this->shape.size(),
                "Index array size must match the tensor shape size.");
  int index = 0;
  int stride = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    index += indices[i] * stride;
    stride *= shape[i];
  }
  return this->data[index];
}

template <std::size_t N> float &Tensor::get_grad(const int (&indices)[N]) {
  static_assert(N == this->shape.size(),
                "Index array size must match the tensor shape size.");
  int index = 0;
  int stride = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    index += indices[i] * stride;
    stride *= shape[i];
  }
  return this->grad[index];
}

Tensor Tensor::operator+(Tensor &other) {
  auto front = [](Tensor *first, Tensor *second, Tensor *out, int i, int j,
                  int k) { out->data[k] = first->data[i] + second->data[j]; };
  auto back = [](Tensor *first, Tensor *second, Tensor *out, int i, int j,
                 int k) {
    first->grad[i] += out->grad[k];
    second->grad[j] += out->grad[k];
  };
  return transform(this, &other, front, back, "+");
}

Tensor Tensor::operator-(Tensor &other) {
  auto front = [](Tensor *first, Tensor *second, Tensor *out, int i, int j,
                  int k) { out->data[k] = first->data[i] - second->data[j]; };
  auto back = [](Tensor *first, Tensor *second, Tensor *out, int i, int j,
                 int k) {
    first->grad[i] += out->grad[k];
    second->grad[j] -= out->grad[k];
  };
  return transform(this, &other, front, back, "-");
}

Tensor Tensor::operator*(Tensor &other) {
  auto front = [](Tensor *first, Tensor *second, Tensor *out, int i, int j,
                  int k) { out->data[k] = first->data[i] * second->data[j]; };
  auto back = [](Tensor *first, Tensor *second, Tensor *out, int i, int j,
                 int k) {
    first->grad[i] += second->data[j] * out->grad[k];
    second->grad[j] += first->data[i] * out->grad[k];
  };
  return transform(this, &other, front, back, "*");
}

Tensor Tensor::operator/(Tensor &other) {
  auto front = [](Tensor *first, Tensor *second, Tensor *out, int i, int j,
                  int k) { out->data[k] = first->data[i] / second->data[j]; };
  auto back = [](Tensor *first, Tensor *second, Tensor *out, int i, int j,
                 int k) {
    first->grad[i] += 1.0 / (second->data[j]) * out->grad[k];
    second->grad[j] +=
        -first->data[i] / (second->data[j] * second->data[j]) * out->grad[k];
  };
  return transform(this, &other, front, back, "/");
}

Tensor Tensor::operator&(Tensor &other) {
  if (this->shape.back() != other.shape.front()) {
    throw std::invalid_argument("Shape mismatch. & requires last dim to match "
                                "first dim of the other tensor.");
  }
  if (this->shape.size() != 2 || other.shape.size() != 2) {
    throw std::invalid_argument("Only 2D tensors are supported");
  }

  std::vector<int> shape = {this->shape[0], other.shape[1]};
  std::vector<float> data;
  for (int i = 0; i < this->shape[0]; i++) {
    for (int j = 0; j < other.shape[1]; j++) {
      float sum = 0;
      for (int k = 0; k < this->shape[1]; k++) {
        sum += this->data[i * this->shape[1] + k] *
               other.data[k * other.shape[1] + j];
      }
      data.push_back(sum);
    }
  }
  auto prev = std::vector<Tensor *>{this, &other};
  auto out = Tensor(data, shape, prev, this->name + " & " + other.name, true);

  auto backward = [&out, this, &other]() {
    auto dC = out.grad;

    std::vector<float> dA(this->shape[0] * this->shape[1], 0);
    for (int i = 0; i < this->shape[0]; i++) {
      for (int k = 0; k < other.shape[0]; k++) {
        for (int j = 0; j < other.shape[1]; j++) {
          dA[i * this->shape[1] + k] +=
              dC[i * other.shape[1] + j] * other.data[k * other.shape[1] + j];
        }
      }
    }

    std::vector<float> dB(other.shape[0] * other.shape[1], 0);
    for (int k = 0; k < this->shape[1]; k++) {
      for (int i = 0; i < this->shape[0]; i++) {
        for (int j = 0; j < other.shape[1]; j++) {
          dB[k * other.shape[1] + j] +=
              this->data[i * this->shape[1] + k] * dC[i * other.shape[1] + j];
        }
      }
    }

    this->grad = dA;
    other.grad = dB;
  };
  out.back = backward;

  return out;
}

void Tensor::view(std::vector<int> shape) {
  auto dim1 =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  auto dim2 = std::accumulate(this->shape.begin(), this->shape.end(), 1,
                              std::multiplies<int>());
  assert(dim1 == dim2);
  this->shape = shape;
}

void Tensor::backward(bool clear_tmp) {
  auto topo = std::vector<Tensor *>();
  auto visited = std::unordered_set<Tensor *>();
  std::function<void(Tensor *)> build_topo = [&](Tensor *t) {
    if (visited.find(t) != visited.end()) {
      return;
    }
    visited.insert(t);
    for (Tensor *p : t->prev) {
      build_topo(p);
    }
    topo.push_back(t);
  };

  for (int i = 0; i < this->grad.size(); i++) {
    this->grad[i] = 1;
  }
  build_topo(this);
  std::reverse(topo.begin(), topo.end());
  for (Tensor *t : topo) {
    t->back();
    if (clear_tmp && t->is_tmp) {
      delete t;
      t = nullptr;
    }
  }
}

void Tensor::clear_tmp() {
  auto topo = std::vector<Tensor *>();
  auto visited = std::unordered_set<Tensor *>();
  std::function<void(Tensor *)> build_topo = [&](Tensor *t) {
    if (visited.find(t) != visited.end()) {
      return;
    }
    visited.insert(t);
    for (Tensor *p : t->prev) {
      build_topo(p);
    }
    topo.push_back(t);
  };

  build_topo(this);
  std::reverse(topo.begin(), topo.end());
  for (Tensor *t : topo) {
    if (t->is_tmp) {
      delete t;
      t = nullptr;
    }
  }
}

Tensor
Tensor::transform(Tensor *first, Tensor *second,
                  void (*front)(Tensor *, Tensor *, Tensor *, int, int, int),
                  void (*back)(Tensor *, Tensor *, Tensor *, int, int, int),
                  std::string name) {
  Tensor *larger;
  if (first->data.size() >= second->data.size()) {
    assert(first->data.size() % second->data.size() == 0);
    larger = first;
  } else {
    assert(second->data.size() % first->data.size() == 0);
    larger = second;
  }

  auto out = Tensor(std::vector<float>(larger->data.size()), larger->shape,
                    std::vector<Tensor *>{first, second},
                    first->name + name + second->name, true);
  int i = 0, j = 0;
  for (int k = 0; k < out.data.size(); k++) {
    if (i == first->data.size())
      i = 0;
    if (j == second->data.size())
      j = 0;
    front(first, second, &out, i, j, k);
    i++;
    j++;
  }

  auto backward = [first, second, &out, back]() {
    int i = 0, j = 0;
    for (int k = 0; k < out.data.size(); k++) {
      if (i == first->data.size())
        i = 0;
      if (j == second->data.size())
        j = 0;
      back(first, second, &out, i, j, k);
      i++;
      j++;
    }
  };
  out.back = backward;

  return out;
}

} // namespace tensor

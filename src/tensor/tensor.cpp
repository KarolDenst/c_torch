#include "tensor.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
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

Tensor Tensor::zeros(std::vector<int> shape, bool is_tmp) {
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  auto data = std::vector<float>(size);
  return Tensor(data, shape, "zeros", is_tmp);
}

Tensor Tensor::zeros_like(const Tensor &tensor, bool is_tmp) {
  return zeros(tensor.shape, is_tmp);
}

Tensor Tensor::rand_n(std::vector<int> shape, bool is_tmp) {
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

void Tensor::print(bool print_prev) {
  std::cout << name;
  std::cout << std::endl << "Data: ";
  for (auto i : this->data) {
    std::cout << i << " ";
  }
  std::cout << std::endl << "Shape: ";
  for (auto i : this->shape) {
    std::cout << i << " ";
  }
  std::cout << std::endl << "Grad: ";
  for (auto i : this->grad) {
    std::cout << i << " ";
  }
  if (print_prev) {
    std::cout << "\nPrev: \n";
    for (auto &t : this->prev) {
      t->print();
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
  if (!equal(this->shape.begin(), this->shape.end(), other.shape.begin())) {
    throw std::invalid_argument("Shape mismatch. + requires the same shape.");
  }
  std::vector<float> data;
  for (int i = 0; i < this->data.size(); i++) {
    data.push_back(this->data[i] + other.data[i]);
  }
  auto prev = std::vector<Tensor *>{this, &other};
  auto out =
      Tensor(data, this->shape, prev, this->name + " + " + other.name, true);

  auto backward = [this, &other, &out]() {
    for (int i = 0; i < out.grad.size(); i++) {
      this->grad[i] += out.grad[i];
      other.grad[i] += out.grad[i];
    }
  };
  out.back = backward;

  return out;
}

Tensor Tensor::operator-(Tensor &other) {
  if (!equal(this->shape.begin(), this->shape.end(), other.shape.begin())) {
    throw std::invalid_argument("Shape mismatch. - requires the same shape.");
  }
  std::vector<float> data;
  for (int i = 0; i < this->data.size(); i++) {
    data.push_back(this->data[i] - other.data[i]);
  }
  auto prev = std::vector<Tensor *>{this, &other};
  auto out =
      Tensor(data, this->shape, prev, this->name + " - " + other.name, true);

  auto backward = [this, &other, &out]() {
    for (int i = 0; i < out.grad.size(); i++) {
      this->grad[i] += out.grad[i];
      other.grad[i] -= out.grad[i];
    }
  };
  out.back = backward;

  return out;
}

Tensor Tensor::operator*(Tensor &other) {
  if (!equal(this->shape.begin(), this->shape.end(), other.shape.begin())) {
    throw std::invalid_argument("Shape mismatch. * requires the same shape.");
  }
  std::vector<float> data;
  for (int i = 0; i < this->data.size(); i++) {
    data.push_back(this->data[i] * other.data[i]);
  }
  auto prev = std::vector<Tensor *>{this, &other};
  auto out =
      Tensor(data, this->shape, prev, this->name + " * " + other.name, true);

  auto backward = [this, &other, &out]() {
    for (int i = 0; i < out.grad.size(); i++) {
      this->grad[i] += other.data[i] * out.grad[i];
      other.grad[i] += this->data[i] * out.grad[i];
    }
  };
  out.back = backward;

  return out;
}

Tensor Tensor::operator/(Tensor &other) {
  if (this->data.size() % other.data.size() != 0) {
    throw std::invalid_argument("Shape mismatch for / operator.");
  }

  std::vector<float> data;
  for (int i = 0; i < this->data.size(); i++) {
    data.push_back(this->data[i] / other.data[i % other.data.size()]);
  }
  auto prev = std::vector<Tensor *>{this, &other};
  auto out =
      Tensor(data, this->shape, prev, this->name + " / " + other.name, true);

  auto backward = [&out, this, &other]() {
    for (int i = 0; i < out.grad.size(); i++) {
      this->grad[i] += 1.0 / (other.data[i % other.data.size()]) * out.grad[i];
      other.grad[i % other.data.size()] +=
          -this->data[i] /
          (other.data[i % other.data.size()] *
               other.data[i % other.data.size()] +
           EPS) *
          out.grad[i];
    }
  };
  out.back = backward;

  return out;
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

Tensor Tensor::tanh() {
  std::vector<float> data;
  for (int i = 0; i < this->data.size(); i++) {
    data.push_back(std::tanh(this->data[i]));
  }
  auto prev = std::vector<Tensor *>{this};
  auto out = Tensor(data, this->shape, prev, "tanh(" + this->name + ")", true);
  auto backward = [this, &out]() {
    for (int i = 0; i < out.grad.size(); i++) {
      this->grad[i] += out.grad[i] * (1 - out.data[i] * out.data[i]);
    }
  };
  out.back = backward;
  return out;
}

Tensor Tensor::exp() {
  std::vector<float> data;
  for (int i = 0; i < this->data.size(); i++) {
    data.push_back(std::exp(this->data[i]));
  }
  auto prev = std::vector<Tensor *>{this};
  auto out = Tensor(data, this->shape, prev, "exp(" + this->name + ")", true);
  auto backward = [this, &out]() {
    for (int i = 0; i < out.grad.size(); i++) {
      this->grad[i] += out.grad[i] * out.data[i];
    }
  };
  out.back = backward;
  return out;
}

Tensor Tensor::log() {
  std::vector<float> data;
  for (int i = 0; i < this->data.size(); i++) {
    data.push_back(std::log(this->data[i]));
  }
  auto prev = std::vector<Tensor *>{this};
  auto out = Tensor(data, this->shape, prev, "log(" + this->name + ")", true);
  auto backward = [this, &out]() {
    for (int i = 0; i < out.grad.size(); i++) {
      this->grad[i] += out.grad[i] * 1.0 / (this->data[i] + EPS);
    }
  };
  out.back = backward;
  return out;
}

Tensor Tensor::sum() {
  std::vector<float> data = {
      std::accumulate(this->data.begin(), this->data.end(), 0.0f)};
  auto prev = std::vector<Tensor *>{this};
  auto out = Tensor(data, {1}, prev, "sum(" + this->name + ")", true);
  auto backward = [this, &out]() {
    for (int i = 0; i < out.prev[0]->data.size(); i++) {
      this->grad[i] += out.grad[0];
    }
  };
  out.back = backward;
  return out;
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

} // namespace tensor

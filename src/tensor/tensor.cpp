#include "tensor.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

Tensor::Tensor(std::vector<float> data, std::vector<int> shape)
    : data(data), shape(shape), prev(std::vector<Tensor *>()),
      grad(std::vector<float>(data.size())), backward([]() {}) {}

Tensor::Tensor(std::vector<float> data, std::vector<int> shape,
               std::vector<Tensor *> prev)
    : data(data), shape(shape), prev(prev),
      grad(std::vector<float>(data.size())), backward([]() {}) {}

Tensor Tensor::zeros(std::vector<int> shape) {
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  auto data = std::vector<float>(size);
  return Tensor(data, shape);
}

Tensor Tensor::rand_n(std::vector<int> shape) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> distr(0.0f, 1.0f);
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  auto data = std::vector<float>(size);
  for (int i = 0; i < size; i++) {
    data[i] = distr(gen);
  }
  return Tensor(data, shape);
}

void Tensor::print(bool print_prev) {
  std::cout << "Data: ";
  for (auto i : this->data) {
    std::cout << i << " ";
  }
  std::cout << "\nShape: ";
  for (auto i : this->shape) {
    std::cout << i << " ";
  }
  std::cout << "Grad: ";
  for (auto i : this->grad) {
    std::cout << i << " ";
  }
  std::cout << "\nPrev: \n";
  if (print_prev) {
    for (auto &t : this->prev) {
      t->print();
    }
  }
  std::cout << "\n";
}

Tensor Tensor::operator+(Tensor &other) {
  if (!equal(this->shape.begin(), this->shape.end(), other.shape.begin())) {
    throw std::invalid_argument("Shape mismatch");
  }
  std::vector<float> data;
  for (int i = 0; i < this->data.size(); i++) {
    data.push_back(this->data[i] + other.data[i]);
  }
  auto prev = std::vector<Tensor *>{this, &other};
  auto out = Tensor(data, this->shape, prev);

  auto backward = [&out]() {
    for (int i = 0; i < out.grad.size(); i++) {
      out.prev[0]->grad[i] += out.grad[i];
      out.prev[1]->grad[i] += out.grad[i];
    }
  };
  out.backward = backward;

  return out;
}

Tensor Tensor::operator*(Tensor &other) {
  if (!equal(this->shape.begin(), this->shape.end(), other.shape.begin())) {
    throw std::invalid_argument("Shape mismatch");
  }
  std::vector<float> data;
  for (int i = 0; i < this->data.size(); i++) {
    data.push_back(this->data[i] * other.data[i]);
  }
  auto prev = std::vector<Tensor *>{this, &other};
  auto out = Tensor(data, this->shape, prev);

  auto backward = [&out, this, &other]() {
    for (int i = 0; i < out.grad.size(); i++) {
      out.prev[0]->grad[i] += other.data[i] * out.grad[i];
      out.prev[1]->grad[i] += this->data[i] * out.grad[i];
    }
  };
  out.backward = backward;

  return out;
}

Tensor Tensor::operator&(Tensor &other) {
  if (this->shape.back() != other.shape.front()) {
    throw std::invalid_argument("Shape mismatch");
  }
  if (this->shape.size() != 2 || other.shape.size() != 2) {
    throw std::invalid_argument("Only 2D tensors are supported");
  }

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
  auto out = Tensor(data, this->shape, prev);

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
  out.backward = backward;

  return out;
}

Tensor Tensor::tanh() {
  std::vector<float> data;
  for (int i = 0; i < this->data.size(); i++) {
    data.push_back(std::tanh(this->data[i]));
  }
  auto prev = std::vector<Tensor *>{this};
  auto out = Tensor(data, this->shape, prev);
  auto backward = [&out]() {
    for (int i = 0; i < out.grad.size(); i++) {
      out.prev[0]->grad[i] += out.grad[i] * (1 - out.data[i] * out.data[i]);
    }
  };
  out.backward = backward;
  return out;
}

void Tensor::backwards() {
  for (int i = 0; i < this->grad.size(); i++) {
    this->grad[i] = 1;
  }
  this->backwards_no_set_grad();
}

void Tensor::backwards_no_set_grad() {
  this->backward();
  for (auto &t : this->prev) {
    t->backwards_no_set_grad();
  }
}

void Tensor::clear_grad_recursive() {
  for (int i = 0; i < this->grad.size(); i++) {
    this->grad[i] = 0;
  }
  for (auto &t : this->prev) {
    t->clear_grad_recursive();
  }
}

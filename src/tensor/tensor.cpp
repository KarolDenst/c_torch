#include "tensor.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

Tensor::Tensor(std::vector<float> data, std::vector<int> shape) {
  this->data = data;
  this->shape = shape;
  this->prev = std::vector<Tensor *>();
  this->grad = 0;
  this->backward = []() {};
}

Tensor::Tensor(std::vector<float> data, std::vector<int> shape,
               std::vector<Tensor *> prev) {
  this->data = data;
  this->shape = shape;
  this->prev = prev;
  this->grad = 0;
  this->backward = []() {};
}

Tensor Tensor::Zeros(std::vector<int> shape) {
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  auto data = std::vector<float>(size);
  for (int i = 0; i < shape[0] * shape[1]; i++) {
    data[i] = 0;
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
  std::cout << "\nGrad: " << this->grad;
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
    out.prev[0]->grad += out.grad;
    out.prev[1]->grad += out.grad;
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
    out.prev[0]->grad += other.data[0] * out.grad;
    out.prev[1]->grad += this->data[0] * out.grad;
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
    out.prev[0]->grad += out.grad * (1 - out.data[0] * out.data[0]);
  };
  out.backward = backward;
  return out;
}

void Tensor::backwards() {
  this->grad = 1;
  this->backwards_no_set_grad();
}

void Tensor::backwards_no_set_grad() {
  this->backward();
  for (auto &t : this->prev) {
    t->backwards_no_set_grad();
  }
}

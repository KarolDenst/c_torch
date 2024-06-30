#include "tensor.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

Tensor::Tensor(std::vector<float> data, std::vector<int> shape) {
  this->data = data;
  this->shape = shape;
  this->prev = std::vector<Tensor *>();
  this->grad = std::vector<float>(data.size());
  this->backward = []() {};
}

Tensor::Tensor(std::vector<float> data, std::vector<int> shape,
               std::vector<Tensor *> prev) {
  this->data = data;
  this->shape = shape;
  this->prev = prev;
  this->grad = std::vector<float>(data.size());
  this->backward = []() {};
}

Tensor Tensor::Zeros(std::vector<int> shape) {
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  auto data = std::vector<float>(size);
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

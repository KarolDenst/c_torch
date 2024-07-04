#include "activation/softmax.h"
#include "activation/tanh.h"
#include "containers/sequential.h"
#include "functional/loss.h"
#include "linear/linear.h"
#include "optim/sgd.h"
#include "tensor/tensor.h"
#include <iostream>
#include <ostream>
#include <random>
#include <vector>

Tensor get_tensor(int num) {
  auto t = Tensor::zeros({1, 10});
  t.data[num] = 1.0f;

  return t;
}

int main() {
  int iterations = 10000;
  float step = 0.01f;
  std::mt19937 gen(0);
  std::uniform_int_distribution<> distr(0, 9);

  auto model = Sequential(
      {new Linear(10, 5), new Tanh(), new Linear(5, 10), new Softmax()});
  auto optimizer = SGD(model.parameters());

  for (int i = 0; i < iterations; i++) {
    int num = distr(gen);
    auto data = get_tensor(num);
    data.name = "data";
    auto expected = get_tensor(num);
    expected.name = "expected";

    optimizer.zero_grad();
    auto result = model.forward(new Tensor(data));
    auto loss = cross_entropy(*result, expected);
    loss.backwards();

    if (i % 100 == 0) {
      std::cout << "Iteration " << i << " Loss: " << loss.data[0] << "\n";
      result->print();
      expected.print();
    }

    optimizer.step();
  }

  return 0;
}

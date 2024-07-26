#include "nn/activation/tanh.h"
#include "nn/containers/sequential.h"
#include "nn/functional/loss.h"
#include "nn/linear/linear.h"
#include "nn/optim/sgd.h"
#include "tensor.h"
#include "tensor/tensor_create.h"
#include "tensor_utils.h"
#include <iostream>
#include <ostream>
#include <random>
#include <vector>

tensor::Tensor get_tensor(int num) {
  auto t = tensor::zeros({1, 10});
  t.data[num] = 1.0f;

  return t;
}

int main() {
  int iterations = 1000000;
  float step = 0.01f;
  std::mt19937 gen(0);
  std::uniform_int_distribution<> distr(0, 9);

  auto model = nn::container::Sequential({new nn::linear::Linear(10, 5),
                                          new nn::activation::Tanh(),
                                          new nn::linear::Linear(5, 10)});
  auto optimizer = nn::optim::SGD(model.parameters());

  for (int i = 0; i < iterations; i++) {
    auto x = std::vector<Tensor *>();
    auto y = std::vector<Tensor *>();
    for (int j = 0; j < 10; j++) {
      int num = distr(gen);
      auto data = new Tensor(get_tensor(num));
      x.push_back(data);
      auto expected = new Tensor(get_tensor(num));
      y.push_back(expected);
    }
    auto data = tensor::stack(x);
    data.name = "data";
    auto expected = tensor::stack(y);
    expected.is_tmp = false;
    expected.name = "expected";

    optimizer.zero_grad();
    auto result = model.forward(new Tensor(data));
    auto loss = nn::functional::mse_loss(*result, expected);

    if (i % 10000 == 0) {
      std::cout << "Iteration " << i << " Loss: " << loss->data[0] << "\n";
    }

    loss->backward();
    optimizer.step();
  }

  return 0;
}

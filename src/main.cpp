#include "nn/activation/softmax.h"
#include "nn/activation/tanh.h"
#include "nn/containers/sequential.h"
#include "nn/functional/loss.h"
#include "nn/linear/linear.h"
#include "nn/optim/sgd.h"
#include "tensor.h"
#include "tensor_create.h"
#include <iostream>
#include <ostream>
#include <random>
#include <vector>

tensor::Tensor get_tensor(int num) {
  int batch_size = 10;
  int num_classes = 10;
  auto t = tensor::zeros({1, num_classes});
  t.data[num] = 1.0f;

  return t;
}

int main() {
  int iterations = 1000000;
  float step = 0.01f;
  std::mt19937 gen(0);
  std::uniform_int_distribution<> distr(0, 9);

  auto model = nn::container::Sequential(
      {new nn::linear::Linear(10, 5), new nn::activation::Tanh(),
       new nn::linear::Linear(5, 10), new nn::activation::Softmax()});
  auto optimizer = nn::optim::SGD(model.parameters());

  for (int i = 0; i < iterations; i++) {
    int num = distr(gen);
    auto data = get_tensor(num);
    data.name = "data";
    auto expected = get_tensor(num);
    expected.is_tmp = false;
    expected.name = "expected";

    optimizer.zero_grad();
    auto result = model.forward(new Tensor(data));
    auto loss = nn::functional::cross_entropy(*result, expected);

    if (i % 10000 == 0) {
      std::cout << "Iteration " << i << " Loss: " << loss->data[0] << "\n";
    }

    loss->backward();
    optimizer.step();
  }

  return 0;
}

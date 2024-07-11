#include "nn/activation/softmax.h"
#include "nn/activation/tanh.h"
#include "nn/containers/sequential.h"
#include "nn/functional/loss.h"
#include "nn/functional/tensor_func.h"
#include "nn/linear/linear.h"
#include "nn/optim/sgd.h"
#include "tensor.h"
#include "utils/data/csv_reader.h"
#include <iostream>
#include <ostream>
#include <random>
#include <string>
#include <vector>
#include <windows.h>

std::string getExecutablePath() {
  char path[MAX_PATH];
  GetModuleFileNameA(NULL, path, MAX_PATH);
  std::string::size_type pos = std::string(path).find_last_of("\\/");
  return std::string(path).substr(0, pos + 1);
}

int main() {
  std::string exe_path = getExecutablePath();
  std::string train_path = exe_path + "..\\data\\train.csv";
  auto train_reader = utils::data::CSVReader(train_path);
  std::vector<tensor::Tensor> y_train;
  auto labels = train_reader.pop("label");

  std::transform(labels.begin(), labels.end(), std::back_inserter(y_train),
                 [](const std::string &label) {
                   auto result =
                       nn::functional::one_hot(std::stoi(label), 10, false);
                   result.view({1, 10});
                   return result;
                 });

  std::vector<tensor::Tensor> x_train;
  std::transform(train_reader.data.begin(), train_reader.data.end(),
                 std::back_inserter(x_train),
                 [](const std::vector<std::string> &row) {
                   std::vector<float> result(row.size());
                   for (int i = 0; i < row.size(); i++) {
                     result[i] = std::stof(row[i]) / 255.0f;
                   }
                   return Tensor(result, {1, 28 * 28});
                 });

  auto model = nn::container::Sequential(
      {new nn::linear::Linear(28 * 28, 512), new nn::activation::Tanh(),
       new nn::linear::Linear(512, 512), new nn::activation::Tanh(),
       new nn::linear::Linear(512, 10), new nn::activation::Tanh(),
       new nn::activation::Softmax()});

  auto optimizer = nn::optim::SGD(model.parameters(), 0.01f);

  for (int i = 0; i < y_train.size(); i++) {
    auto x = x_train[i];
    x.name = "data";
    auto y = y_train[i];
    y.name = "expected";
    y.is_tmp = false;

    optimizer.zero_grad();
    auto result = model.forward(new Tensor(x));
    auto loss = nn::functional::cross_entropy(*result, y);

    if (i % 1000 == 0) {
      std::cout << "Iteration " << i << " Loss: " << loss->data[0] << "\n";
      result->print();
      y.print();
    }

    loss->backward();
    optimizer.step();
  }

  return 0;
}

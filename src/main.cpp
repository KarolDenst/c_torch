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

std::string get_executable_path() {
  char path[MAX_PATH];
  GetModuleFileNameA(NULL, path, MAX_PATH);
  std::string::size_type pos = std::string(path).find_last_of("\\/");
  return std::string(path).substr(0, pos + 1);
}

int get_max_index(tensor::Tensor tensor) {
  auto max_element = std::max_element(tensor.data.begin(), tensor.data.end());
  return std::distance(tensor.data.begin(), max_element);
}

int main() {
  // Setup Training Data
  std::string exe_path = get_executable_path();
  std::string train_path = exe_path + "..\\data\\train.csv";
  auto csv_reader = utils::data::CSVReader(train_path);
  auto labels = csv_reader.pop("label");

  auto y_transform = [](const std::string &label) {
    auto result = nn::functional::one_hot(std::stoi(label), 10, false);
    result.view({1, 10});
    return result;
  };
  auto x_transform = [](const std::vector<std::string> &row) {
    std::vector<float> result(row.size());
    for (int i = 0; i < row.size(); i++) {
      result[i] = std::stof(row[i]) / 255.0f;
    }
    return Tensor(result, {1, 28 * 28});
  };

  int train_size = 0.8 * csv_reader.data.size();

  std::vector<tensor::Tensor> x_train;
  std::transform(csv_reader.data.begin(), csv_reader.data.begin() + train_size,
                 std::back_inserter(x_train), x_transform);
  std::vector<tensor::Tensor> y_train;
  std::transform(labels.begin(), labels.begin() + train_size,
                 std::back_inserter(y_train), y_transform);

  std::vector<tensor::Tensor> x_val;
  std::transform(csv_reader.data.begin() + train_size, csv_reader.data.end(),
                 std::back_inserter(x_val), x_transform);
  std::vector<tensor::Tensor> y_val;
  std::transform(labels.begin() + train_size, labels.end(),
                 std::back_inserter(y_val), y_transform);
  // csv_reader.~CSVReader();

  // Define Model and parameters
  auto model = nn::container::Sequential(
      {new nn::linear::Linear(28 * 28, 512), new nn::activation::Tanh(),
       new nn::linear::Linear(512, 128), new nn::activation::Tanh(),
       new nn::linear::Linear(128, 10), new nn::activation::Tanh(),
       new nn::activation::Softmax()});

  auto optimizer = nn::optim::SGD(model.parameters(), 0.01f);
  const int epochs = 1;
  const int batch_size = 32;

  // Train model
  for (int epoch = 0; epoch < epochs; epoch++) {
    for (int batch = 0; batch + batch_size < y_train.size();
         batch += batch_size) {

      for (int i = batch; i < batch + batch_size; i++) {
        optimizer.zero_grad();
        auto x = x_train[i];
        x.name = "data";
        auto y = y_train[i];
        y.name = "expected";

        auto result = model.forward(new Tensor(x));
        auto loss = nn::functional::cross_entropy(*result, y);

        if (i % 3000 == 0) {
          std::cout << "Epoch: " << epoch << ", Iteration " << i
                    << " Loss: " << loss->data[0] << "\n";
          result->print();
          y.print();
        }
        loss->backward();
        optimizer.step();
      }
    }
  }

  // Validate model
  float accuracy = 0;
  float avg_loss = 0;
  float size = y_val.size();
  for (int i = 0; i < y_val.size(); i++) {
    optimizer.zero_grad();
    auto x = x_val[i];
    auto y = y_val[i];

    auto result = model.forward(new Tensor(x));

    int result_index = get_max_index(*result);
    int y_index = get_max_index(y);
    if (result_index == y_index) {
      accuracy++;
    }

    auto loss = nn::functional::cross_entropy(*result, y);
    avg_loss += loss->data[0];
    loss->clear_tmp();
  }
  accuracy /= size;
  avg_loss /= size;

  std::cout << "Validation Accuracy: " << accuracy
            << ", Average Loss: " << avg_loss << "\n";

  return 0;
}

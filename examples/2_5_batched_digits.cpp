#include "nn/activation/relu.h"
#include "nn/activation/softmax.h"
#include "nn/activation/tanh.h"
#include "nn/containers/sequential.h"
#include "nn/dropout/dropout.h"
#include "nn/functional/loss.h"
#include "nn/linear/linear.h"
#include "nn/optim/adam.h"
#include "nn/optim/sgd.h"
#include "tensor.h"
#include "tensor/tensor_create.h"
#include "tensor_utils.h"
#include "utils/data/csv_reader.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <ostream>
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
  auto max_element =
      std::max_element(tensor.data().begin(), tensor.data().end());
  return std::distance(tensor.data().begin(), max_element);
}

int main() {
  // Setup Training Data
  std::string exe_path = get_executable_path();
  std::string train_path = exe_path + "..\\..\\data\\train.csv";
  auto csv_reader = utils::data::CSVReader(train_path);
  auto labels = csv_reader.pop("label");

  auto y_transform = [](const std::string &label) {
    auto result = tensor::one_hot(std::stoi(label), 10);
    result.view({10});
    return result;
  };
  auto x_transform = [](const std::vector<std::string> &row) {
    std::vector<float> result(row.size());
    for (int i = 0; i < row.size(); i++) {
      result[i] = std::stof(row[i]) / 255.0f;
    }
    return Tensor(result, {28 * 28});
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
  csv_reader.~CSVReader();

  // Define Model and parameters
  int input_size = 28 * 28;
  int num_classes = 10;
  auto model = nn::container::Sequential({
      new nn::linear::Linear(input_size, 512),
      new nn::activation::Tanh(),
      new nn::linear::Linear(512, 128),
      new nn::activation::Tanh(),
      new nn::linear::Linear(128, num_classes),
  });

  auto optimizer = nn::optim::Adam(model.parameters(), 0.001f);
  auto criterion = [](Tensor &x, Tensor &y) {
    return nn::functional::cross_entropy(x, y, "mean");
  };
  const int epochs = 1;
  const int batch_size = 32;

  // Train model
  model.train();
  for (int epoch = 0; epoch < epochs; epoch++) {

    float total_loss = 0;
    for (int batch = 0; batch + batch_size < y_train.size();
         batch += batch_size) {
      auto x_tensors = std::vector<Tensor>();
      auto y_tensors = std::vector<Tensor>();
      for (int i = batch; i < batch + batch_size; i++) {
        x_tensors.push_back(x_train[i]);
        y_tensors.push_back(y_train[i]);
      }
      auto x = tensor::stack(x_tensors);
      x.name() = "data";
      auto y = tensor::stack(y_tensors);
      y.name() = "expected";

      auto result = model.forward(x);
      auto loss = criterion(result, y);

      total_loss += loss.data(0);
      int iteration = (batch + batch_size) / batch_size;
      if (iteration % 100 == 0) {
        total_loss /= 100;
        std::cout << "Epoch: " << epoch << ", Iteration " << iteration
                  << " Loss: " << total_loss << "\n";
        total_loss = 0;
      }
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
    }
  }

  std::string model_path = exe_path + "..\\model.ct";
  model.save(model_path);

  // Validate model
  model.eval();
  float accuracy = 0;
  float avg_loss = 0;
  float size = y_val.size();
  for (int i = 0; i < y_val.size(); i++) {
    optimizer.zero_grad();
    auto x = x_val[i];
    auto y = y_val[i];

    auto result = model.forward(x);

    int result_index = get_max_index(result);
    int y_index = get_max_index(y);
    if (result_index == y_index) {
      accuracy++;
    }

    auto loss = criterion(result, y);
    avg_loss += loss.data(0);
  }
  accuracy /= size;
  avg_loss /= size;

  std::cout << "Validation Accuracy: " << accuracy
            << ", Average Loss: " << avg_loss << "\n";

  return 0;
}

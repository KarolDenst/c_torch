// #include "tensor/tensor_create.h"
// #include <chrono>
// #include <iostream>
// #include <vector>
//
// using namespace tensor;
// int main() {
//   std::chrono::steady_clock::time_point begin =
//       std::chrono::steady_clock::now();
//
//   auto result = zeros({10, 10});
//   auto size = std::vector<int>{1000, 1000};
//   auto x = rand_n(size);
//   auto y = rand_n(size);
//   for (int i = 0; i < 10; i++) {
//     result = x & y;
//     result = x & y;
//     result = x & y;
//     result = x & y;
//     result = x & y;
//   }
//   std::chrono::steady_clock::time_point end =
//   std::chrono::steady_clock::now(); std::cout << "Time = "
//             << std::chrono::duration_cast<std::chrono::milliseconds>(end -
//                                                                      begin)
//                    .count()
//             << "[ms]" << std::endl;
// }
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
#include "vision/transforms.h"
#include "vision/utils.h"
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
    return tensor::Tensor({static_cast<float>(std::stoi(label))}, {1});
    auto result = tensor::one_hot(std::stoi(label), 10);
    result.view({10});
    return result;
  };
  auto x_transform = [](const std::vector<std::string> &row) {
    std::vector<float> result(row.size());
    for (int i = 0; i < row.size(); i++) {
      result[i] = std::stof(row[i]) / 255.0f;
    }
    return Tensor(result, {28, 28});
  };

  int train_size = 10;
  std::vector<tensor::Tensor> x_train;
  std::transform(csv_reader.data.begin(), csv_reader.data.begin() + train_size,
                 std::back_inserter(x_train), x_transform);
  std::vector<tensor::Tensor> y_train;
  std::transform(labels.begin(), labels.begin() + train_size,
                 std::back_inserter(y_train), y_transform);
  csv_reader.~CSVReader();

  for (int i = 0; i < train_size; i++) {
    std::cout << "Label: " << y_train[i].data(0) << std::endl;
    vision::print(x_train[i], 0.1);
    vision::print(vision::transforms::random_rotation(x_train[i], 1), 0.1);
    // vision::print(vision::transforms::random_vertical_flip(x_train[i], 1),
    // 0.1);
  }

  return 0;
}

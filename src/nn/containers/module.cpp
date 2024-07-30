#include "module.h"
#include <cassert>
#include <fstream>
#include <sstream>

namespace nn {
std::vector<tensor::Tensor *> Module::parameters() { return {}; }

void Module::save(std::string filename) {
  std::ofstream file(filename);
  assert(file.is_open());

  for (auto tensor : parameters()) {
    for (auto shape : tensor->shape()) {
      file << shape << " ";
    }
    file << std::endl;

    for (auto value : tensor->data()) {
      file << value << " ";
    }
    file << std::endl;
  }
}

void Module::load(std::string filename) {
  std::ifstream file(filename);
  assert(file.is_open());

  std::string line;
  for (auto tensor : parameters()) {
    float value;

    std::getline(file, line);
    std::istringstream iss(line);
    for (int i = 0; i < tensor->shape().size(); i++) {
      iss >> value;
      tensor->shape()[i] = value;
    }

    std::getline(file, line);
    for (int i = 0; i < tensor->data().size(); i++) {
      iss >> value;
      tensor->data()[i] = value;
    }
  }
}
} // namespace nn

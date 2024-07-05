#include "sequential.h"
#include "../tensor/tensor.h"
#include <vector>


using namespace tensor;

namespace nn {
namespace container {

Sequential::Sequential(std::vector<Module *> modules) : modules(modules) {}

Tensor *Sequential::forward(Tensor *data) {
  auto result = data;
  for (auto &module : modules) {
    result = module->forward(result);
  }
  return result;
}

std::vector<Tensor *> Sequential::parameters() {
  std::vector<Tensor *> params;
  for (auto &module : modules) {
    auto module_params = module->parameters();
    params.insert(params.end(), module_params.begin(), module_params.end());
  }
  return params;
}

void Sequential::append(Module *module) { modules.push_back(module); }

} // namespace container
} // namespace nn

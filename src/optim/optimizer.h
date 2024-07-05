#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "../tensor/tensor.h"

namespace nn {
namespace optim {

class Optimizer {
public:
  Optimizer(std::vector<tensor::Tensor *> parameters)
      : parameters(parameters) {}
  virtual void step() {}
  virtual void zero_grad() {
    for (tensor::Tensor *parameter : parameters) {
      std::fill(parameter->grad.begin(), parameter->grad.end(), 0.0f);
    }
  }

protected:
  std::vector<tensor::Tensor *> parameters;
};

} // namespace optim
} // namespace nn

#endif // OPTIMIZER_H

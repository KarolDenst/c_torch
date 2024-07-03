#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "../tensor/tensor.h"

class Optimizer {
public:
  Optimizer(std::vector<Tensor *> parameters) : parameters(parameters) {}
  virtual void step() {}
  virtual void zero_grad() {
    for (Tensor *parameter : parameters) {
      std::fill(parameter->grad.begin(), parameter->grad.end(), 0.0f);
    }
  }

protected:
  std::vector<Tensor *> parameters;
};

#endif // OPTIMIZER_H

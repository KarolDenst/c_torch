#ifndef SGD_H
#define SGD_H

#include "../tensor/tensor.h"
#include "optimizer.h"


class SGD : public Optimizer {
public:
  SGD(std::vector<Tensor *> parameters, float learning_rate = 1e-3)
      : Optimizer(parameters), learning_rate(learning_rate) {}
  virtual void step() {
    for (Tensor *parameter : parameters) {
      for (int i = 0; i < parameter->grad.size(); i++) {
        parameter->data[i] -= learning_rate * parameter->grad[i];
      }
    }
  }

private:
  float learning_rate;
};

#endif // SGD_H

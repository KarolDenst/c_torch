#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "../containers/module.h"

class Softmax : public Module {
public:
  Tensor forward(Tensor &data) {
    Tensor exps = data.exp();
    Tensor sum = exps.sum();
    return exps / sum;
  }
};

#endif // SOFTMAX_H

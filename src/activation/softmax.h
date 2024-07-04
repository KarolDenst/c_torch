#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "../containers/module.h"

class Softmax : public Module {
public:
  Tensor *forward(Tensor *data) {
    auto exps = new Tensor(data->exp());
    auto sum = new Tensor(exps->sum());
    return new Tensor(*exps / *sum);
  }
};

#endif // SOFTMAX_H

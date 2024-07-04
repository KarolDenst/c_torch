#ifndef TANH_H
#define TANH_H

#include "../containers/module.h"

class Tanh : public Module {
public:
  Tensor *forward(Tensor *data) { return new Tensor(data->tanh()); }
};

#endif // TANH_H

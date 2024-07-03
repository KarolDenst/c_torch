#ifndef MODULE_H
#define MODULE_H

#include "../tensor/tensor.h"

class Module {
public:
  virtual Tensor forward(Tensor &data) = 0;
  virtual std::vector<Tensor *> parameters() { return {}; }
};

#endif // MODULE_H

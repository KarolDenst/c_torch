#ifndef MODULE_H
#define MODULE_H

#include "../tensor/tensor.h"

namespace nn {

class Module {
public:
  virtual tensor::Tensor *forward(tensor::Tensor *data) = 0;
  virtual std::vector<tensor::Tensor *> parameters() { return {}; }
};

} // namespace nn

#endif // MODULE_H

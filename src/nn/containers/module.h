#ifndef MODULE_H
#define MODULE_H

#include "../../tensor/tensor.h"

namespace nn {

class Module {
public:
  virtual tensor::Tensor forward(tensor::Tensor data) = 0;
  virtual std::vector<tensor::Tensor *> parameters();
  void save(std::string filename);
  void load(std::string filename);
};

} // namespace nn

#endif // MODULE_H

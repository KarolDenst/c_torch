#ifndef MODULE_H
#define MODULE_H

#include "../../tensor/tensor.h"

namespace nn {

class Module {
public:
  virtual tensor::Tensor forward(tensor::Tensor data) = 0;
  virtual std::vector<tensor::Tensor *> parameters();
  virtual void train() { training = true; }
  virtual void eval() { training = false; }
  void save(std::string filename);
  void load(std::string filename);

protected:
  bool training = true;
};

} // namespace nn

#endif // MODULE_H

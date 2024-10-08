#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "../../tensor/tensor.h"
#include "module.h"

namespace nn {
namespace container {

class Sequential : public Module {
public:
  Sequential(std::vector<Module *> modules);

  tensor::Tensor forward(tensor::Tensor data) override;
  std::vector<tensor::Tensor *> parameters() override;
  void train() override;
  void eval() override;

  void append(Module *module);

private:
  std::vector<Module *> modules;
};

} // namespace container
} // namespace nn

#endif // SEQUENTIAL_H

#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "../tensor/tensor.h"
#include "module.h"

class Sequential : public Module {
public:
  Sequential(std::vector<Module *> modules);

  Tensor *forward(Tensor *data);
  std::vector<Tensor *> parameters();

  void append(Module *module);

private:
  std::vector<Module *> modules;
};

#endif // SEQUENTIAL_H

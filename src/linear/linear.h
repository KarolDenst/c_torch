#ifndef LINEAR_H
#define LINEAR_H

#include "../containers/module.h"
#include "../tensor/tensor.h"
#include <optional>

class Linear : public Module {
public:
  Linear(int in_features, int out_features, bool has_bias = true);
  Tensor forward(Tensor &data);
  std::vector<Tensor *> parameters();

private:
  int in_features;
  int out_features;
  bool has_bias;
  Tensor weights;
  std::optional<Tensor> bias;
};

#endif // LINEAR_H

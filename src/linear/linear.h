#ifndef LINEAR_H
#define LINEAR_H

#include "../tensor/tensor.h"
#include <optional>

class Linear {
private:
  int in_features;
  int out_features;
  bool has_bias;
  Tensor weights;
  std::optional<Tensor> bias;

  Linear(int in_features, int out_features, bool has_bias = true);
};

#endif // LINEAR_H

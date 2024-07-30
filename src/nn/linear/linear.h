#ifndef LINEAR_H
#define LINEAR_H

#include "../../tensor/tensor.h"
#include "../containers/module.h"
#include <optional>

namespace nn {
namespace linear {

class Linear : public Module {
public:
  Linear(int in_features, int out_features, bool has_bias = true);
  tensor::Tensor forward(tensor::Tensor data);
  std::vector<tensor::Tensor *> parameters();

private:
  int in_features;
  int out_features;
  bool has_bias;
  tensor::Tensor weights;
  std::optional<tensor::Tensor> bias;
};

} // namespace linear
} // namespace nn

#endif // LINEAR_H

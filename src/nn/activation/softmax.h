#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "../../tensor/tensor.h"
#include "../../tensor/tensor_func.h"
#include "../containers/module.h"

namespace nn {
namespace activation {

class Softmax : public Module {
public:
  Softmax(std::optional<int> dim = std::nullopt) { this->dim = dim; }

  tensor::Tensor forward(tensor::Tensor data) override {
    bool keepdim = false;
    if (dim.has_value())
      keepdim = true;
    auto exps = tensor::exp(data);
    auto sum = tensor::sum(exps, dim, keepdim);
    return exps / sum;
  }

private:
  std::optional<int> dim;
};

} // namespace activation
} // namespace nn

#endif // SOFTMAX_H

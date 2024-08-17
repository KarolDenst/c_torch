#ifndef TANH_H
#define TANH_H

#include "../../tensor/tensor_func.h"
#include "../containers/module.h"

namespace nn {
namespace activation {

class Tanh : public Module {
public:
  tensor::Tensor forward(tensor::Tensor data) override {
    return tensor::tanh(data);
  }
};

} // namespace activation
} // namespace nn
#endif // TANH_H

#ifndef RELU_H
#define RELU_H

#include "../../tensor/tensor_func.h"
#include "../containers/module.h"

namespace nn {
namespace activation {

class ReLU : public Module {
public:
  tensor::Tensor forward(tensor::Tensor data) { return tensor::relu(data); }
};

} // namespace activation
} // namespace nn
#endif // RELU_H

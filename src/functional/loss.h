#ifndef LOSS_H
#define LOSS_H

#include "../tensor/tensor.h"

namespace nn {
namespace functional {

tensor::Tensor *cross_entropy(tensor::Tensor &output, tensor::Tensor &target);

} // namespace functional
} // namespace nn

#endif // LOSS_H

#ifndef LOSS_H
#define LOSS_H

#include "../../tensor/tensor.h"

namespace nn {
namespace functional {

tensor::Tensor binary_cross_entropy(tensor::Tensor &output,
                                    tensor::Tensor &target);
tensor::Tensor cross_entropy(tensor::Tensor &output, tensor::Tensor &target,
                             std::string reduction = "mean");
tensor::Tensor mse_loss(tensor::Tensor &output, tensor::Tensor &target,
                        std::string reduction = "mean");

} // namespace functional
} // namespace nn

#endif // LOSS_H

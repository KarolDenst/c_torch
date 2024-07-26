#ifndef TENSOR_FUNC_H
#define TENSOR_FUNC_H

#include "tensor.h"
#include <optional>

namespace tensor {

Tensor tanh(Tensor *tensor);
Tensor relu(Tensor *tensor);
Tensor log(Tensor *tensor);
Tensor exp(Tensor *tensor);
Tensor sum(Tensor *tensor, std::optional<int> dim = std::nullopt,
           bool keepdim = false);
Tensor mean(Tensor *tensor);

} // namespace tensor

#endif // TENSOR_FUNC_H

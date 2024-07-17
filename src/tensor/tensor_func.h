#ifndef TENSOR_FUNC_H
#define TENSOR_FUNC_H

#include "tensor.h"
#include <optional>

namespace tensor {

Tensor tanh(Tensor *tensor);
Tensor log(Tensor *tensor);
Tensor exp(Tensor *tensor);
Tensor sum(Tensor *tensor, std::optional<int> dim = std::nullopt);

} // namespace tensor

#endif // TENSOR_FUNC_H

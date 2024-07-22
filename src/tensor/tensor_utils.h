#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include "tensor.h"
#include <vector>

namespace tensor {

Tensor stack(std::vector<Tensor *> tensors);

} // namespace tensor

#endif // TENSOR_UTILS_H

#ifndef TENSOR_FUNC_H
#define TENSOR_FUNC_H

#include "tensor.h"

namespace tensor {

Tensor tanh(Tensor *tensor);
Tensor log(Tensor *tensor);
Tensor exp(Tensor *tensor);
Tensor sum(Tensor *tensor);

} // namespace tensor

#endif // TENSOR_FUNC_H

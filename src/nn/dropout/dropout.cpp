#include "dropout.h"
#include "../../tensor/tensor.h"
#include "../../tensor/tensor_create.h"
#include <cassert>

using namespace tensor;

namespace nn {
namespace dropout {

Dropout::Dropout(float p) : p(p) { assert(p >= 0 && p <= 1); }

Tensor Dropout::forward(Tensor data) {
  if (training) {
    auto mask = tensor::uniform(data.shape(), 0.0, 1.0) > p;
    return data * mask;
  } else {
    auto mul = Tensor({1 - p}, {1});
    return data * mul;
  }
}

} // namespace dropout
} // namespace nn

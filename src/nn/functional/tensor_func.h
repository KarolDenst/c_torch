#ifndef TENSOR_FUNC_H
#define TENSOR_FUNC_H

#include "../../tensor/tensor.h"
#include <vector>

namespace nn {
namespace functional {

tensor::Tensor one_hot(int num, int num_classes, bool is_tmp = true);
tensor::Tensor uniform(std::vector<int> shape, float low, float high,
                       bool is_tmp = true);

} // namespace functional
} // namespace nn

#endif // TENSOR_FUNC_H

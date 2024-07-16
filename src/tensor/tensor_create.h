#ifndef TENSOR_CREATE_H
#define TENSOR_CREATE_H

#include "tensor.h"
#include <vector>

namespace tensor {

tensor::Tensor one_hot(int num, int num_classes, bool is_tmp = true);
tensor::Tensor uniform(std::vector<int> shape, float low, float high,
                       bool is_tmp = true);
tensor::Tensor zeros(std::vector<int> shape, bool is_tmp = true);
tensor::Tensor zeros_like(const Tensor &tensor, bool is_tmp = true);
tensor::Tensor rand_n(std::vector<int> shape, bool is_tmp = true);

} // namespace tensor

#endif // TENSOR_CREATE_H

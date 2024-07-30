#ifndef TENSOR_CREATE_H
#define TENSOR_CREATE_H

#include "tensor.h"
#include <vector>

namespace tensor {

tensor::Tensor one_hot(int num, int num_classes);
tensor::Tensor uniform(std::vector<int> shape, float low, float high);
tensor::Tensor zeros(std::vector<int> shape);
tensor::Tensor zeros_like(const Tensor &tensor);
tensor::Tensor rand_n(std::vector<int> shape);

} // namespace tensor

#endif // TENSOR_CREATE_H

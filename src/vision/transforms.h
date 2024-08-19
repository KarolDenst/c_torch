#ifndef TRANSFORMS_H
#define TRANSFORMS_H

#include "../tensor/tensor.h"

namespace vision {
namespace transforms {

tensor::Tensor resize(tensor::Tensor tensor, int height, int width);
tensor::Tensor random_horizontal_flip(tensor::Tensor tensor, float p);
tensor::Tensor random_vertical_flip(tensor::Tensor tensor, float p);
tensor::Tensor random_rotation(tensor::Tensor tensor, float degrees);

} // namespace transforms
} // namespace vision

#endif // TRANSFORMS_H

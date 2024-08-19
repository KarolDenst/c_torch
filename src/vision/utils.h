#ifndef VISION_UTILS_H
#define VISION_UTILS_H

#include "../tensor/tensor.h"

namespace vision {

void print(tensor::Tensor tensor, float threshold = 0.1);

} // namespace vision

#endif // VISION_UTILS_H

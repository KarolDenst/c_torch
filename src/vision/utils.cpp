#include "utils.h"
#include "../tensor/tensor.h"
#include <cassert>
#include <iostream>
#include <ostream>

using namespace tensor;

namespace vision {
void print(Tensor tensor, float threshold) {
  assert(tensor.shape().size() == 2);
  for (int i = 0; i < tensor.shape()[0]; i++) {
    for (int j = 0; j < tensor.shape()[1]; j++) {
      float val = tensor.data(tensor.shape(1) * i + j);
      if (val > threshold)
        std::cout << "#";
      else
        std::cout << " ";
    }
    std::cout << std::endl;
  }
}
} // namespace vision

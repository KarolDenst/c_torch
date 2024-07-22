#include "tensor_utils.h"
#include "tensor.h"
#include <cassert>
#include <vector>

namespace tensor {

Tensor stack(std::vector<Tensor *> tensors) {
  auto shape = std::vector<int>{static_cast<int>(tensors.size())};
  for (auto size : tensors[0]->shape) {
    shape.push_back(size);
  }
  auto data = std::vector<float>(tensors[0]->data.size() * tensors.size());
  for (int i = 0; i < tensors.size(); i++) {
    assert(tensors[i]->shape == tensors[0]->shape);
    for (int j = 0; j < tensors[0]->data.size(); j++) {
      data[i * tensors[0]->data.size() + j] = tensors[i]->data[j];
    }
  }
  return Tensor(data, shape);
}

} // namespace tensor

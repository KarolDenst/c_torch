#ifndef TANH_H
#define TANH_H

#include "../containers/module.h"

namespace nn {
namespace activation {

class Tanh : public Module {
public:
  tensor::Tensor *forward(tensor::Tensor *data) {
    return new tensor::Tensor(data->tanh());
  }
};

} // namespace activation
} // namespace nn
#endif // TANH_H

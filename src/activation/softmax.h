#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "../containers/module.h"
#include "../tensor/tensor.h"

namespace nn {
namespace activation {

class Softmax : public Module {
public:
  tensor::Tensor *forward(tensor::Tensor *data) {
    auto exps = new tensor::Tensor(data->exp());
    auto sum = new tensor::Tensor(exps->sum());
    return new tensor::Tensor(*exps / *sum);
  }
};

} // namespace activation
} // namespace nn

#endif // SOFTMAX_H

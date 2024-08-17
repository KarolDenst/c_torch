#ifndef DROPOUT_H
#define DROPOUT_H

#include "../containers/module.h"

namespace nn {
namespace dropout {

class Dropout : public Module {
public:
  Dropout(float p);
  tensor::Tensor forward(tensor::Tensor data) override;

private:
  float p; // probability of an element to be zeroed
};

} // namespace dropout
} // namespace nn
#endif // DROPOUT_H

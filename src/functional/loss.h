#ifndef LOSS_H
#define LOSS_H

#include "../tensor/tensor.h"
#include <memory>

// Tensor cross_entropy(Tensor &output, Tensor &target);
std::shared_ptr<Tensor> cross_entropy(const std::shared_ptr<Tensor> &output,
                                      const std::shared_ptr<Tensor> &target);

#endif // LOSS_H

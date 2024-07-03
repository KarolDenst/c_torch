#ifndef LOSS_H
#define LOSS_H

#include "../tensor/tensor.h"

Tensor cross_entropy(Tensor &output, Tensor &target);

#endif // LOSS_H

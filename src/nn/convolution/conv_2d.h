#ifndef CONV_2D_H
#define CONV_2D_H

#include "../../tensor/tensor.h"
#include "../containers/module.h"
#include <optional>

namespace nn {
namespace conv {

// class Conv2d : public Module {
// public:
//   Conv2d(int in_channels, int out_channels, int kernel_size,
//          bool has_bias = true);
//   tensor::Tensor forward(tensor::Tensor data) override;
//   std::vector<tensor::Tensor *> parameters() override;
//
// private:
//   bool has_bias;
//   tensor::Tensor weights;
//   std::optional<tensor::Tensor> bias;
//   int in_channels;
//   int out_channels;
//   int kernel_size;
// };

} // namespace conv
} // namespace nn

#endif // CONV_2D_H

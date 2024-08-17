#include "conv_2d.h"
#include "../../tensor/tensor.h"
#include "../../tensor/tensor_create.h"
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace tensor;

namespace nn {
namespace conv {

// Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size,
//                bool has_bias)
//     : in_channels(in_channels), out_channels(out_channels),
//       kernel_size(kernel_size), has_bias(has_bias),
//       weights(tensor::rand_n(
//           {kernel_size, kernel_size, in_channels, out_channels})),
//       bias(has_bias ? std::make_optional(tensor::zeros({out_channels}))
//                     : std::nullopt) {
//   weights.name() = "weights";
//   if (this->has_bias)
//     bias.value().name() = "bias";
// }
//
// Tensor Conv2d::forward(Tensor data) {
//   // Data in format (batch, channels, height, width)
//   if (data.shape().size() != 4) {
//     throw std::invalid_argument("Input data must be 4D");
//   }
//   auto result_shape = std::vector<int>{data.shape(0), out_channels,
//                                        data.shape(2) + 1 - kernel_size,
//                                        data.shape(3) + 1 - kernel_size};
//   auto result_data = std::vector<float>(
//       std::accumulate(result_shape.begin(), result_shape.end(), 1,
//                       std::multiplies<float>()),
//       0);
//
//   for (int b = 0; b < result_shape[0]; b++) {
//     for (int c = 0; c < result_shape[1]; c++) {
//       for (int y = 0; y < result_shape[2]; y++) {
//         for (int x = 0; x < result_shape[3]; x++) {
//           for (int m = 0; m < kernel_size; m++) {
//             for (int n = 0; n < kernel_size; n++) {
//               for (int o = 0; o < in_channels; o++) {
//                 result_data[b * result_shape[1] * result_shape[2] *
//                                 result_shape[3] +
//                             c * result_shape[2] * result_shape[3] +
//                             y * result_shape[3] + x] +=
//                     data.data(b * data.shape(1) * data.shape(2) *
//                                   data.shape(3) +
//                               o * data.shape(2) * data.shape(3) +
//                               (y + m) * data.shape(3) + (x + n)) *
//                     weights.data(m * kernel_size * in_channels * out_channels
//                     +
//                                  n * in_channels * out_channels +
//                                  o * out_channels + c);
//               }
//             }
//           }
//         }
//       }
//     }
//   }
//
//   auto prev = std::vector<Tensor>{data};
//   auto output =
//       Tensor(result_data, result_shape, prev, "conv(" + data.name() + ")");
//   auto backwards = [this, data, output]() {};
//   output.back = backwards;
//
//   if (this->has_bias) {
//     return Tensor(output + bias.value());
//   }
//   return output;
// }
//
// std::vector<tensor::Tensor *> Conv2d::parameters() {
//   std::vector<Tensor *> params;
//   params.push_back(&weights);
//   if (this->has_bias)
//     params.push_back(&bias.value());
//   return params;
// }

} // namespace conv
} // namespace nn

#include "conv_2d.h"
#include "../../tensor/tensor.h"
#include <stdexcept>
#include <vector>

using namespace tensor;

namespace nn {
namespace conv {

// Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size,
//                bool has_bias)
//     : in_channels(in_channels), out_channels(out_channels),
//       kernel_size(kernel_size), has_bias(has_bias),
//       weights(Tensor::rand_n(
//           {kernel_size, kernel_size, in_channels, out_channels}, false)),
//       bias(has_bias ? std::make_optional(Tensor::zeros({out_channels}, false))
//                     : std::nullopt) {
//   weights.name = "weights";
//   if (this->has_bias)
//     bias.value().name = "bias";
// }

// tensor::Tensor *Conv2d::forward(tensor::Tensor *data) {
//   if (data->shape.size() != 3) {
//     throw std::invalid_argument("Input data must be 3D");
//   }
//   auto result_shape =
//       std::vector<int>{data->shape[0] - kernel_size + 1,
//                        data->shape[1] - kernel_size + 1, out_channels};
//   auto result_data = std::vector<float>();

//   for (int i = 0; i < data->shape[0] - kernel_size + 1; i++) {
//     for (int j = 0; j < data->shape[1] - kernel_size + 1; j++) {
//       for (int out = 0; out < out_channels; out++) {

//         float sum = 0;
//         for (int in = 0; in < in_channels; in++) {
//           for (int k = 0; k < kernel_size; k++) {
//             for (int l = 0; l < kernel_size; l++) {
//               sum += data->get_data({i + k, j + l, in}) *
//                      weights.get_data({k, l, in, out});
//             }
//           }
//         }
//         result_data.push_back(sum);
//       }
//     }
//   }
//   auto prev = std::vector<Tensor *>{data};
//   auto output = new Tensor(result_data, result_shape, prev,
//                            "conv(" + data->name + ")", true);
//   auto backwards = [this, data, output]() {
//     for (int i = 0; i < data->shape[0] - kernel_size + 1; i++) {
//       for (int j = 0; j < data->shape[1] - kernel_size + 1; j++) {
//         for (int out = 0; out < out_channels; out++) {
//           for (int in = 0; in < in_channels; in++) {
//             for (int k = 0; k < kernel_size; k++) {
//               for (int l = 0; l < kernel_size; l++) {
//                 float grad = output->get_grad({i, j, out});
//                 data->get_grad({i + k, j + l, in}) +=
//                     grad * weights.get_data({k, l, in, out});
//                 weights.get_grad({k, l, in, out}) +=
//                     grad * data->get_data({i + k, j + l, in});
//               }
//             }
//           }
//         }
//       }
//     }
//   };
//   output->back = backwards;

//   if (this->has_bias) {
//     return new Tensor(*output + bias.value());
//   }
//   return output;
// }

// std::vector<tensor::Tensor *> Conv2d::parameters() {
//   std::vector<Tensor *> params;
//   params.push_back(&weights);
//   if (this->has_bias)
//     params.push_back(&bias.value());
//   return params;
// }

} // namespace conv
} // namespace nn

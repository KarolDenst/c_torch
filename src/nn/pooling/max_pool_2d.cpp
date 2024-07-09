#include "max_pool_2d.h"
#include "../../tensor/tensor.h"
#include <limits>
#include <vector>

using namespace tensor;

namespace nn {
namespace pooling {

// MaxPool2d::MaxPool2d(int kernel_size) : kernel_size(kernel_size) {}
// Tensor *MaxPool2d::forward(tensor::Tensor *data) {
//   auto result_shape =
//       std::vector<int>{data->shape[0] - kernel_size + 1,
//                        data->shape[1] - kernel_size + 1, data->shape[2]};
//   auto result_data = std::vector<float>();

//   for (int i = 0; i < data->shape[0]; i += kernel_size) {
//     for (int j = 0; j < data->shape[1]; j += kernel_size) {
//       for (int out_channels = 0; out_channels < data->shape[1];
//            out_channels += kernel_size) {
//         float max = std::numeric_limits<float>::min();
//         for (int x = 0; x < kernel_size; x++) {
//           for (int y = 0; y < kernel_size; y++) {
//             max = std::max(max, data->get_data({i, j, out_channels}));
//           }
//         }
//         result_data.push_back(max);
//       }
//     }
//   }

//   auto prev = std::vector<Tensor *>{data};
//   auto output = new Tensor(result_data, result_shape, prev,
//                            "max_pool(" + data->name + ")", true);
//   auto backwards = [this, data, output]() {
//     for (int i = 0; i < data->shape[0]; i += kernel_size) {
//       for (int j = 0; j < data->shape[1]; j += kernel_size) {
//         for (int out_channels = 0; out_channels < data->shape[1];
//              out_channels += kernel_size) {
//           for (int x = 0; x < kernel_size; x++) {
//             for (int y = 0; y < kernel_size; y++) {
//               if (data->get_data({i, j, out_channels}) ==
//                   output->get_data(
//                       {i / kernel_size, j / kernel_size, out_channels})) {
//                 data->get_grad({i, j, out_channels}) = output->get_grad(
//                     {i / kernel_size, j / kernel_size, out_channels});
//               }
//             }
//           }
//         }
//       }
//     }
//   };
//   output->back = backwards;

//   return output;
// }
// std::vector<tensor::Tensor *> MaxPool2d::parameters() {
//   return std::vector<Tensor *>();
// }

} // namespace pooling
} // namespace nn

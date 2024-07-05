#include "loss.h"
#include "../../tensor/tensor.h"
#include <stdexcept>

using namespace tensor;

namespace nn {
namespace functional {

Tensor *cross_entropy(Tensor &output, Tensor &target) {
  if (output.shape[0] != target.shape[0]) {
    throw std::invalid_argument(
        "output and target should have the same batch size");
  }
  auto number = static_cast<int>(output.data.size());
  auto one =
      new Tensor(std::vector<float>(number, 1.0f), {1, number}, "one", true);
  auto inverse =
      new Tensor(std::vector<float>(1, -1.0f / number), {1}, "inverse", true);

  auto output_log = new Tensor(output.log());
  auto one_minus_output = new Tensor(*one - output);
  auto one_minus_output_log = new Tensor(one_minus_output->log());
  auto left = new Tensor(target * *output_log);
  auto one_minus_target = new Tensor(*one - target);
  auto right = new Tensor(*one_minus_target * *one_minus_output_log);

  auto sum_vec = new Tensor(*left + *right);
  auto sum = new Tensor(sum_vec->sum());
  auto loss = new Tensor(*inverse * *sum);

  return loss;
}

} // namespace functional
} // namespace nn

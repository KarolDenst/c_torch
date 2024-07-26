#include "loss.h"
#include "../../tensor/tensor.h"
#include "../../tensor/tensor_func.h"
#include <cassert>
#include <stdexcept>

using namespace tensor;

namespace nn {
namespace functional {

Tensor *cross_entropy(Tensor &output, Tensor &target) {
  assert(output.data.size() == target.data.size());
  auto number = static_cast<int>(output.data.size());
  auto one =
      new Tensor(std::vector<float>(number, 1.0f), output.shape, "one", true);
  auto inverse =
      new Tensor(std::vector<float>(1, -1.0f / number), {1}, "inverse", true);

  auto output_log = new Tensor(log(&output));
  auto one_minus_output = new Tensor(*one - output);
  auto one_minus_output_log = new Tensor(log(one_minus_output));
  auto left = new Tensor(target * *output_log);
  auto one_minus_target = new Tensor(*one - target);
  auto right = new Tensor(*one_minus_target * *one_minus_output_log);

  auto sum_vec = new Tensor(*left + *right);
  auto sum = new Tensor(tensor::sum(sum_vec));
  auto loss = new Tensor(*inverse * *sum);

  return loss;
}

Tensor *mse_loss(Tensor &output, Tensor &target, std::string reduction) {
  auto diff = new Tensor(output - target);
  auto square = new Tensor(*diff * *diff);
  auto sum = new Tensor(tensor::sum(square));
  if (reduction == "sum") {
    return sum;
  } else if (reduction == "mean") {
    auto number = static_cast<float>(output.data.size());
    auto inverse =
        new Tensor(std::vector<float>(1, 1.0f / number), {1}, "inverse", true);
    return new Tensor(*inverse * *sum);
  } else {
    throw std::invalid_argument("Invalid reduction");
  }
}

} // namespace functional
} // namespace nn

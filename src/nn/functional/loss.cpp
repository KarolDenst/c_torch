#include "loss.h"
#include "../../tensor/tensor.h"
#include "../../tensor/tensor_func.h"
#include <cassert>
#include <stdexcept>

using namespace tensor;

namespace nn {
namespace functional {

Tensor *binary_cross_entropy(Tensor &output, Tensor &target) {
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

tensor::Tensor *cross_entropy(tensor::Tensor &output, tensor::Tensor &target,
                              std::string reduction) {
  if (output.shape.size() == 1)
    output.view({1, output.shape[0]});
  auto exp = new Tensor(tensor::exp(&output));
  auto sum = new Tensor(tensor::sum(exp, 1));
  auto log = new Tensor(tensor::log(sum));
  auto mul = new Tensor(
      output * target); // -input[i, target[i]] for one hot encoded target
  auto mul_sum = new Tensor(tensor::sum(mul, 1));
  auto result = new Tensor(*log - *mul_sum);

  if (reduction == "mean")
    return new Tensor(tensor::mean(result));
  else if (reduction == "sum")
    return new Tensor(tensor::mean(result));
  else if (reduction == "")
    return result;
  else
    throw std::invalid_argument("Invalid reduction");
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

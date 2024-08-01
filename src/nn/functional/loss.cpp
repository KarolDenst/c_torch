#include "loss.h"
#include "../../tensor/tensor.h"
#include "../../tensor/tensor_func.h"
#include <cassert>
#include <stdexcept>

using namespace tensor;

namespace nn {
namespace functional {

Tensor binary_cross_entropy(Tensor &output, Tensor &target) {
  assert(output.data().size() == target.data().size());
  auto number = static_cast<int>(output.data().size());
  auto one = Tensor(std::vector<float>(number, 1.0f), output.shape(), "one");
  auto inverse = Tensor(std::vector<float>(1, -1.0f / number), {1}, "inverse");

  auto output_log = log(output);
  auto one_minus_output = one - output;
  auto one_minus_output_log = log(one_minus_output);
  auto left = target * output_log;
  auto one_minus_target = one - target;
  auto right = one_minus_target * one_minus_output_log;

  auto sum_vec = left + right;
  auto sum = tensor::sum(sum_vec);
  auto loss = inverse * sum;

  return loss;
}

tensor::Tensor cross_entropy(tensor::Tensor &output, tensor::Tensor &target,
                             std::string reduction) {
  if (output.shape().size() == 1)
    output.view({1, output.shape()[0]});
  auto exp = tensor::exp(output);
  auto sum = tensor::sum(exp, 1);
  auto log = tensor::log(sum);
  auto mul = output * target; // -input[i, target[i]] for one hot encoded target
  auto mul_sum = tensor::sum(mul, 1);
  auto result = log - mul_sum;

  if (reduction == "mean")
    return tensor::mean(result);
  else if (reduction == "sum")
    return tensor::sum(result);
  else if (reduction == "")
    return result;
  else
    throw std::invalid_argument("Invalid reduction");
}

Tensor mse_loss(Tensor &output, Tensor &target, std::string reduction) {
  auto diff = output - target;
  auto square = diff * diff;
  auto sum = tensor::sum(square);
  if (reduction == "sum") {
    return sum;
  } else if (reduction == "mean") {
    auto number = static_cast<float>(output.data().size());
    auto inverse = Tensor(std::vector<float>(1, 1.0f / number), {1}, "inverse");
    return inverse * sum;
  } else {
    throw std::invalid_argument("Invalid reduction");
  }
}

} // namespace functional
} // namespace nn

#include "loss.h"
#include "../tensor/tensor.h"
#include <memory>
#include <stdexcept>

// Tensor cross_entropy(Tensor &output, Tensor &target) {
//   if (output.shape[0] != target.shape[0]) {
//     throw std::invalid_argument(
//         "output and target should have the same batch size");
//   }
//   auto number = static_cast<int>(output.data.size());
//   auto one = Tensor(std::vector<float>(number, 1.0f), {1, number});
//   auto inverse = Tensor(std::vector<float>(1, -1.0f / number), {1});
//
//   auto output_log = output.log();
//   auto one_minus_output_log = (one - output).log();
//   auto left = target * output_log;
//   auto right = (one - target) * one_minus_output_log;
//
//   auto sum_vec = left + right;
//   auto sum = sum_vec.sum();
//   auto loss = inverse * sum;
//
//   return loss;
// }

std::shared_ptr<Tensor> cross_entropy(const std::shared_ptr<Tensor> &output,
                                      const std::shared_ptr<Tensor> &target) {
  if (output->shape[0] != target->shape[0]) {
    throw std::invalid_argument(
        "output and target should have the same batch size");
  }

  auto number = static_cast<int>(output->data.size());
  auto one = std::make_shared<Tensor>(std::vector<float>(number, 1.0f),
                                      std::vector<int>{1, number});
  auto inverse = std::make_shared<Tensor>(std::vector<float>(1, -1.0f / number),
                                          std::vector<int>{1});

  auto output_log = std::make_shared<Tensor>(output->log());
  auto one_minus_output_log =
      std::make_shared<Tensor>(((*one) - (*output)).log());
  auto left = std::make_shared<Tensor>((*target) * (*output_log));
  auto right =
      std::make_shared<Tensor>((*one - *target) * (*one_minus_output_log));

  auto sum_vec = std::make_shared<Tensor>((*left) + (*right));
  auto sum = std::make_shared<Tensor>(sum_vec->sum());
  auto loss = std::make_shared<Tensor>((*inverse) * (*sum));

  return loss;
}

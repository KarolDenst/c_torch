#include "linear.h"
#include "../tensor/tensor.h"
#include <vector>


using namespace tensor;

namespace nn {
namespace linear {

Linear::Linear(int in_features, int out_features, bool has_bias)
    : in_features(in_features), out_features(out_features), has_bias(has_bias),
      weights(Tensor::rand_n({in_features, out_features}, false)),
      bias(has_bias
               ? std::make_optional(Tensor::zeros({1, out_features}, false))
               : std::nullopt) {
  weights.name = "weights";
  if (this->has_bias)
    bias.value().name = "bias";
}

Tensor *Linear::forward(Tensor *data) {
  auto result = new Tensor(*data & weights);
  if (this->has_bias)
    result = new Tensor(*result + bias.value());
  return result;
}

std::vector<Tensor *> Linear::parameters() {
  std::vector<Tensor *> params;
  params.push_back(&weights);
  if (this->has_bias)
    params.push_back(&bias.value());
  return params;
}

} // namespace linear
} // namespace nn

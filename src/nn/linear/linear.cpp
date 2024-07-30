#include "linear.h"
#include "../../tensor/tensor.h"
#include "../../tensor/tensor_create.h"
#include <cmath>
#include <vector>

using namespace tensor;

namespace nn {
namespace linear {

Linear::Linear(int in_features, int out_features, bool has_bias)
    : in_features(in_features), out_features(out_features), has_bias(has_bias),
      weights(tensor::uniform({in_features, out_features},
                              -1.0f / std::sqrt(in_features),
                              1.0f / std::sqrt(in_features))),
      bias(has_bias ? std::make_optional(tensor::zeros({out_features}))
                    : std::nullopt) {
  weights.var->name = "weights";
  if (this->has_bias)
    bias.value().var->name = "bias";
}

Tensor Linear::forward(Tensor data) {
  auto result = data & weights;
  if (this->has_bias)
    result = result + bias.value();
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

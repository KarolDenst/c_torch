#include "linear.h"
#include <vector>

Linear::Linear(int in_features, int out_features, bool has_bias)
    : in_features(in_features), out_features(out_features), has_bias(has_bias),
      weights(Tensor::rand_n({in_features, out_features})),
      bias(has_bias ? std::make_optional(Tensor::zeros({1, out_features}))
                    : std::nullopt) {}

Tensor Linear::forward(Tensor &data) {
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

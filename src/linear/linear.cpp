#include "linear.h"

Linear::Linear(int in_features, int out_features, bool has_bias)
    : in_features(in_features), out_features(out_features), has_bias(has_bias),
      weights(Tensor::rand_n({in_features, out_features})),
      bias(has_bias ? std::make_optional(Tensor::rand_n({out_features}))
                    : std::nullopt) {}

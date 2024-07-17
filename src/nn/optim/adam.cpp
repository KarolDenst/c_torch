#include "adam.h"
#include "optimizer.h"
#include <cmath>

namespace nn {
namespace optim {

Adam::Adam(std::vector<tensor::Tensor *> parameters, float learning_rate,
           float beta_1, float beta_2, float eps)
    : Optimizer(parameters), learning_rate(learning_rate), beta_1(beta_1),
      beta_2(beta_2), eps(eps) {
  t = 0;
  for (tensor::Tensor *parameter : parameters) {
    m.push_back(std::vector<float>(parameter->data.size(), 0));
    v.push_back(std::vector<float>(parameter->data.size(), 0));
  }
}

void Adam::step() {
  t++;
  beta_1_to_t_power = beta_1_to_t_power * beta_1;
  beta_2_to_t_power = beta_2_to_t_power * beta_2;

  for (int i = 0; i < parameters.size(); i++) {
    for (int j = 0; j < parameters[i]->data.size(); j++) {
      m[i][j] = beta_1 * m[i][j] + (1 - beta_1) * parameters[i]->grad[j];
      v[i][j] =
          beta_2 * v[i][j] + (1 - beta_2) * std::pow(parameters[i]->grad[j], 2);

      float m_hat = m[i][j] / (1 - beta_1_to_t_power);
      float v_hat = v[i][j] / (1 - beta_2_to_t_power);

      parameters[i]->data[j] -=
          learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
  }
}

} // namespace optim
} // namespace nn

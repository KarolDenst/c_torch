#ifndef ADAM_H
#define ADAM_H

#include "../../tensor/tensor.h"
#include "optimizer.h"

namespace nn {
namespace optim {

class Adam : public Optimizer {
public:
  Adam(std::vector<tensor::Tensor *> parameters, float learning_rate = 1e-3,
       float beta_1 = 0.9, float beta_2 = 0.999, float eps = 1e-8);
  virtual void step();

private:
  float learning_rate;
  float beta_1;
  float beta_2;
  float eps;

  int t = 0;
  std::vector<std::vector<float>> m;
  std::vector<std::vector<float>> v;

  float beta_1_to_t_power;
  float beta_2_to_t_power;
};

} // namespace optim
} // namespace nn

#endif // ADAM_H

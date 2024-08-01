#ifndef TENSOR_H
#define TENSOR_H

#include "variable/variable.h"
#include <memory>
#include <string>
#include <vector>

namespace tensor {

class Tensor {
public:
  Tensor(std::vector<float> data, std::vector<int> shape,
         std::string name = "");
  Tensor(std::vector<float> data, std::vector<int> shape,
         std::vector<Tensor> prev, std::string name = "");
  Tensor(std::shared_ptr<variable::Variable> var);

  std::vector<float> &data() { return var->data; }
  std::vector<float> &grad() { return var->grad; }
  std::vector<int> &shape() { return var->shape; }

  float &data(int index) { return var->data[index]; }
  float &grad(int index) { return var->grad[index]; }
  int &shape(int index) { return var->shape[index]; }

  std::string &name() { return var->name; }

  Tensor operator+(Tensor &other);
  Tensor operator-(Tensor &other);
  Tensor operator*(Tensor &other);
  Tensor operator/(Tensor &other);
  Tensor operator&(Tensor &other);

  void print(bool print_prev = false);
  void view(std::vector<int> shape);
  void backward();

private:
  std::shared_ptr<variable::Variable> var;
};

} // namespace tensor

#endif // TENSOR_H

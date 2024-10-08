#ifndef TENSOR_H
#define TENSOR_H

#include "variable/variable.h"
#include <memory>
#include <string>
#include <vector>

namespace tensor {

class Tensor {
public:
  std::shared_ptr<variable::Variable<>> var;
  Tensor(std::vector<float> data, std::vector<int> shape,
         std::string name = "");
  Tensor(std::vector<float> data, std::vector<int> shape,
         std::vector<Tensor> prev, std::string name = "");
  Tensor(std::shared_ptr<variable::Variable<>> var);

  float &get(std::initializer_list<int> args);

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
  Tensor operator>(float value);

  void print(bool print_prev = false);
  void view(std::vector<int> shape);
  void backward();

private:
};

} // namespace tensor

#endif // TENSOR_H

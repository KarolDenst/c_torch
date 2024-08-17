#include "tensor.h"
#include "variable/variable.h"
#include <cassert>
#include <memory>
#include <vector>

using namespace variable;

namespace tensor {

Tensor::Tensor(std::vector<float> data, std::vector<int> shape,
               std::string name)
    : var(std::make_shared<Variable>(data, shape, name)) {}

Tensor::Tensor(std::vector<float> data, std::vector<int> shape,
               std::vector<Tensor> prev, std::string name)
    : var(std::make_shared<Variable>(data, shape, name)) {
  for (auto &tensor : prev) {
    var->prev.push_back(tensor.var);
  }
}

Tensor::Tensor(std::shared_ptr<Variable> var) : var(var) {}

void Tensor::print(bool print_prev) { var->print(print_prev); }

Tensor Tensor::operator+(Tensor &other) {
  return Tensor(Variable::add(var, other.var));
}

Tensor Tensor::operator-(Tensor &other) {
  return Tensor(Variable::sub(var, other.var));
}

Tensor Tensor::operator*(Tensor &other) {
  return Tensor(Variable::mul(var, other.var));
}

Tensor Tensor::operator/(Tensor &other) {
  return Tensor(Variable::div(var, other.var));
}

Tensor Tensor::operator&(Tensor &other) {
  return Tensor(Variable::mat_mul(var, other.var));
}

Tensor Tensor::operator>(float value) {
  return Tensor(Variable::greater(var, value));
}

void Tensor::view(std::vector<int> shape) { var->view(shape); }

void Tensor::backward() { var->backward(); }

} // namespace tensor

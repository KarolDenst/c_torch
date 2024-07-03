#ifndef TENSOR_H
#define TENSOR_H

#include <functional>
#include <string>
#include <vector>

const float EPS = 1e-7;

class Tensor {
public:
  std::vector<float> data;
  std::vector<int> shape;
  std::vector<float> grad;
  std::string name;

  Tensor(std::vector<float> data, std::vector<int> shape,
         std::string name = "");
  static Tensor zeros(std::vector<int> shape);
  static Tensor rand_n(std::vector<int> shape);
  void print(bool print_prev = false);
  Tensor operator+(Tensor &other);
  Tensor operator-(Tensor &other);
  Tensor operator*(Tensor &other);
  Tensor operator/(Tensor &other);
  Tensor operator&(Tensor &other);
  Tensor tanh();
  Tensor log();
  Tensor exp();
  Tensor sum();
  void backwards();
  void clear_grad_recursive();

private:
  std::function<void(void)> backward;
  std::vector<Tensor *> prev;

  Tensor(std::vector<float> data, std::vector<int> shape,
         std::vector<Tensor *> prev, std::string name = "");
  void backwards_no_set_grad();
};

#endif // TENSOR_H

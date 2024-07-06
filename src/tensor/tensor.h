#ifndef TENSOR_H
#define TENSOR_H

#include <functional>
#include <string>
#include <vector>

namespace tensor {

const float EPS = 1e-7;

class Tensor {
public:
  std::vector<float> data;
  std::vector<int> shape;
  std::vector<float> grad;
  std::string name;
  bool is_tmp;

  Tensor(std::vector<float> data, std::vector<int> shape, std::string name = "",
         bool is_tmp = true);
  static Tensor zeros(std::vector<int> shape, bool is_tmp = true);
  static Tensor rand_n(std::vector<int> shape, bool is_tmp = true);
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
  void backward(bool clear_tmp = true);

private:
  std::function<void(void)> back;
  std::vector<Tensor *> prev;

  Tensor(std::vector<float> data, std::vector<int> shape,
         std::vector<Tensor *> prev, std::string name = "", bool is_tmp = true);
};

} // namespace tensor

#endif // TENSOR_H

#ifndef TENSOR_H
#define TENSOR_H

#include <functional>
#include <vector>

class Tensor {
public:
  Tensor(std::vector<float> data, std::vector<int> shape);
  static Tensor Zeros(std::vector<int> shape);
  void print(bool print_prev = false);
  Tensor operator+(Tensor &other);
  Tensor operator*(Tensor &other);
  Tensor tanh();
  void backwards();

private:
  std::vector<float> data;
  std::vector<int> shape;
  float grad;
  std::function<void(void)> backward;
  std::vector<Tensor *> prev;

  Tensor(std::vector<float> data, std::vector<int> shape,
         std::vector<Tensor *> prev);
  void backwards_no_set_grad();
};

#endif // TENSOR_H

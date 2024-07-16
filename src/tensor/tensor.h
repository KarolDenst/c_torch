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
  std::string name = "";
  bool is_tmp;
  std::function<void(void)> back;
  std::vector<Tensor *> prev;

  Tensor(std::vector<float> data, std::vector<int> shape, std::string name = "",
         bool is_tmp = true);
  Tensor(std::vector<float> data, std::vector<int> shape,
         std::vector<Tensor *> prev, std::string name = "", bool is_tmp = true);
  void print(bool print_prev = false);
  template <std::size_t N> float &get_data(const int (&indices)[N]);
  template <std::size_t N> float &get_grad(const int (&indices)[N]);
  Tensor operator+(Tensor &other);
  Tensor operator-(Tensor &other);
  Tensor operator*(Tensor &other);
  Tensor operator/(Tensor &other);
  Tensor operator&(Tensor &other);
  void view(std::vector<int> shape);
  void backward(bool clear_tmp = true);
  void clear_tmp();

private:
};

} // namespace tensor

#endif // TENSOR_H

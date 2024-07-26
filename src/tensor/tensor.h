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
  std::vector<int> strides;
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
  static std::vector<int> compute_strides(std::vector<int> shape);
  static std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
  compute_broadcast_strides(Tensor &first, Tensor &second);
  static Tensor
  transform(Tensor *first, Tensor *second,
            void (*front)(Tensor *, Tensor *, Tensor *, int, int, int),
            void (*back)(Tensor *, Tensor *, Tensor *, int, int, int),
            std::string name = "");
  static void transform_rec(int, int, int, int, Tensor &, Tensor *, Tensor *,
                            std::vector<int> &, std::vector<int> &,
                            void (*front)(Tensor *, Tensor *, Tensor *, int,
                                          int, int));
};

} // namespace tensor

#endif // TENSOR_H

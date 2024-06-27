#include "tensor/tensor.h"

int main() {
  auto x1 = Tensor({2.0}, {1});
  auto x2 = Tensor({0.0}, {1});
  auto w1 = Tensor({-3.0}, {1});
  auto w2 = Tensor({1.0}, {1});
  auto b = Tensor({6.88137}, {1});
  auto x1w1 = x1 * w1;
  auto x2w2 = x2 * w2;
  auto x1w1_x2w2 = x1w1 + x2w2;
  auto n = x1w1_x2w2 + b;
  auto o = n.tanh();
  o.backwards();

  x1.print();
  w1.print();
  x2.print();
  w2.print();
  x1w1.print();
  x2w2.print();
  x1w1_x2w2.print();
  b.print();
  n.print();
  o.print();

  return 0;
}

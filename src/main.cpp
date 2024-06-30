#include "tensor/tensor.h"

int main() {
  auto t1 = Tensor({1.0, 2.0}, {1, 2});
  auto t2 = Tensor({8.0, 6.0, 4.0, 2.0}, {2, 2});

  auto sum = t1 & t2;
  sum.backwards();
  sum.print();
  t1.print();
  t2.print();

  return 0;
}

#include "../../../../src/nn/activation/relu.h"
#include "../../../../src/tensor/tensor.h"
#include "../../tensor_tests/tensor_utils.h"
#include <gtest/gtest.h>

TEST(ReluTest, ReluWorks) {
  // arrange
  auto t1 = tensor::Tensor(std::vector<float>({-1, -2, 3, 4}),
                           std::vector<int>({2, 2}));
  auto relu = nn::activation::ReLU();

  // act
  auto result = relu.forward(&t1);
  result->backward(false);

  // assert
  ExpectVectorsNear(result->data, {0, 0, 3, 4});
  ExpectVectorsNear(t1.grad, {0, 0, 1, 1});
}

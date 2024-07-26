#include "../../../../src/nn/functional/loss.h"
#include "../../../../src/tensor/tensor.h"
#include "../../tensor_tests/tensor_utils.h"
#include <gtest/gtest.h>

TEST(LossTests, MSE_WithMeanReduction_Works) {
  // arrange
  auto t1 = tensor::Tensor({1, 2, 3, 4}, {2, 2});
  auto t2 = tensor::Tensor({2, 4, 6, 8}, {2, 2});

  // act
  auto result = nn::functional::mse_loss(t1, t2, "mean");
  result->backward(false);

  // assert
  EXPECT_EQ(result->data[0], 7.5);
  ExpectVectorsNear(t1.grad, {-0.5, -1, -1.5, -2});
  ExpectVectorsNear(t2.grad, {0.5, 1, 1.5, 2});
}

TEST(LossTests, MSE_WithSumReduction_Works) {
  // arrange
  auto t1 = tensor::Tensor({1, 2, 3, 4}, {2, 2});
  auto t2 = tensor::Tensor({2, 4, 6, 8}, {2, 2});

  // act
  auto result = nn::functional::mse_loss(t1, t2, "sum");
  result->backward(false);

  // assert
  EXPECT_EQ(result->data[0], 30);
  ExpectVectorsNear(t1.grad, {-2, -4, -6, -8});
  ExpectVectorsNear(t2.grad, {2, 4, 6, 8});
}

TEST(LossTests, CrossEntropy_Works) {
  // arrange
  auto t1 = tensor::Tensor({1, 2, 3, 1, 2, 3}, {2, 3});
  auto t2 = tensor::Tensor({1, 0, 0, 0, 0, 1}, {2, 3});

  // act
  auto result = nn::functional::cross_entropy(t1, t2);
  result->backward(false);

  // assert
  ExpectVectorsNear(result->data, {1.4076});
  ExpectVectorsNear(t1.grad,
                    {-0.4550, 0.1224, 0.3326, 0.0450, 0.1224, -0.1674});
}

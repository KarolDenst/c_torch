#include "../../src/tensor/tensor.h"
#include "../../src/tensor/tensor_func.h"
#include "tensor_utils.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace tensor;

TEST(TensorFunTest, Tanh_Works) {
  // arrange
  auto t = Tensor({1.0, 2.0, 3.0, 4.0}, {2, 2});

  // act
  auto result = tanh(t);
  result.backward();

  // assert
  ExpectVectorsNear(result.data(),
                    std::vector<float>({std::tanh(1.0f), std::tanh(2.0f),
                                        std::tanh(3.0f), std::tanh(4.0f)}));
  ExpectVectorsNear(t.grad(),
                    std::vector<float>({0.4200, 0.0707, 0.0099, 0.0013}));
}

TEST(TensorFunTest, Exp_Works) {
  // arrange
  auto t = Tensor({1.0, 2.0, 3.0, 4.0}, {2, 2});

  // act
  auto result = exp(t);
  result.backward();

  // assert
  ExpectVectorsNear(result.data(),
                    std::vector<float>({std::exp(1.0f), std::exp(2.0f),
                                        std::exp(3.0f), std::exp(4.0f)}));
  ExpectVectorsNear(t.grad(), result.data());
}

TEST(TensorFunTest, Log_Works) {
  // arrange
  auto t = Tensor({1.0, 2.0, 3.0, 4.0}, {2, 2});

  // act
  auto result = log(t);
  result.backward();

  // assert
  ExpectVectorsNear(result.data(),
                    std::vector<float>({std::log(1.0f), std::log(2.0f),
                                        std::log(3.0f), std::log(4.0f)}));
  ExpectVectorsNear(t.grad(),
                    std::vector<float>({1.0, 1.0 / 2, 1.0 / 3, 1.0 / 4}));
}

TEST(TensorFunTest, Sum_Works) {
  // arrange
  auto t = Tensor({1.0, 2.0, 3.0, 4.0}, {2, 2});

  // act
  auto result = sum(t);
  result.backward();

  // assert
  ExpectVectorsNear(result.data(), std::vector<float>({10.0}));
  ExpectVectorsNear(t.grad(), std::vector<float>({1.0, 1.0, 1.0, 1.0}));
  EXPECT_EQ(result.shape(), std::vector<int>({1}));
}

TEST(TensorFunTest, Sum_ForSpecifiedDim_Works) {
  // arrange
  auto t = Tensor({1.0, 2.0, 3.0, 4.0}, {2, 2});

  // act
  auto result = sum(t, 1);
  result.backward();

  // assert
  ExpectVectorsNear(result.data(), std::vector<float>({3, 7}));
  ExpectVectorsNear(t.grad(), std::vector<float>({1.0, 1.0, 1.0, 1.0}));
  EXPECT_EQ(result.shape(), std::vector<int>({2}));
}

TEST(TensorFunTest, Sum_ForSpecifiedDimWithKeepdim_Works) {
  // arrange
  auto t = Tensor({1.0, 2.0, 3.0, 4.0}, {2, 2});

  // act
  auto result = sum(t, 1, true);
  result.backward();

  // assert
  ExpectVectorsNear(result.data(), std::vector<float>({3, 7}));
  ExpectVectorsNear(t.grad(), std::vector<float>({1.0, 1.0, 1.0, 1.0}));
  EXPECT_EQ(result.shape(), std::vector<int>({2, 1}));
}

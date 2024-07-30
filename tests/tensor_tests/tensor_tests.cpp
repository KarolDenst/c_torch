#include "../../src/nn/functional/loss.h"
#include "../../src/tensor/tensor.h"
#include "../../src/tensor/tensor_create.h"
#include "../../src/tensor/tensor_func.h"
#include "tensor_utils.h"
#include <gtest/gtest.h>

using namespace tensor;

TEST(TensorTest, Addition_ForMatchingShapes_Works) {
  // arrange
  auto t1 = Tensor(std::vector<float>({1, 2, 3, 4}), std::vector<int>({2, 2}));
  auto t2 = Tensor(std::vector<float>({2, 4, 6, 8}), std::vector<int>({2, 2}));

  // act
  auto result = t1 + t2;
  result.backward();

  // assert
  ExpectVectorsNear(result.data(), std::vector<float>({3, 6, 9, 12}));
  ExpectVectorsNear(t1.grad(), std::vector<float>({1, 1, 1, 1}));
  ExpectVectorsNear(t2.grad(), std::vector<float>({1, 1, 1, 1}));
}

TEST(TensorTest, Addition_ForFirstLargerThanSecond_BoardcastingWorks) {
  // arrange
  auto t1 = Tensor(std::vector<float>({1, 2, 3, 4}), std::vector<int>({2, 2}));
  auto t2 = Tensor(std::vector<float>({5, 6}), std::vector<int>({2}));

  // act
  auto result = t1 + t2;
  auto result2 = t2 + t1;
  result.backward();

  // assert
  ExpectVectorsNear(result.data(), std::vector<float>({6, 8, 8, 10}));
  ExpectVectorsNear(result.data(), result2.data());
  ExpectVectorsNear(t1.grad(), std::vector<float>({1, 1, 1, 1}));
  ExpectVectorsNear(t2.grad(), std::vector<float>({2, 2}));
}

TEST(TensorTest, Subtraction_ForMatchingShapes_Works) {
  // arrange
  auto t1 = Tensor(std::vector<float>({1, 2, 3, 4}), std::vector<int>({2, 2}));
  auto t2 = Tensor(std::vector<float>({2, 4, 6, 8}), std::vector<int>({2, 2}));

  // act
  auto result = t1 - t2;
  result.backward();

  // assert
  ExpectVectorsNear(result.data(), std::vector<float>({-1, -2, -3, -4}));
  ExpectVectorsNear(t1.grad(), std::vector<float>({1, 1, 1, 1}));
  ExpectVectorsNear(t2.grad(), std::vector<float>({-1, -1, -1, -1}));
}

TEST(TensorTest, Multiplication_ForMatchingShapes_Works) {
  // arrange
  auto t1 = Tensor(std::vector<float>({1, 2, 3, 4}), std::vector<int>({2, 2}));
  auto t2 = Tensor(std::vector<float>({2, 4, 6, 8}), std::vector<int>({2, 2}));

  // act
  auto result = t1 * t2;
  result.backward();

  // assert
  ExpectVectorsNear(result.data(), std::vector<float>({2, 8, 18, 32}));
  ExpectVectorsNear(t1.grad(), std::vector<float>({2, 4, 6, 8}));
  ExpectVectorsNear(t2.grad(), std::vector<float>({1, 2, 3, 4}));
}

TEST(TensorTest, Division_ForMatchingShapes_Works) {
  // arrange
  auto t1 = Tensor(std::vector<float>({1, 2, 3, 4}), std::vector<int>({2, 2}));
  auto t2 = Tensor(std::vector<float>({2, 4, 6, 8}), std::vector<int>({2, 2}));

  // act
  auto result = t1 / t2;
  result.backward();

  // assert
  ExpectVectorsNear(result.data(), std::vector<float>({0.5, 0.5, 0.5, 0.5}));
  ExpectVectorsNear(t1.grad(),
                    std::vector<float>({1.0 / 2, 1.0 / 4, 1.0 / 6, 1.0 / 8}));
  ExpectVectorsNear(t2.grad(), std::vector<float>(
                                   {-1.0 / 4, -1.0 / 8, -1.0 / 12, -1.0 / 16}));
}

TEST(TensorTest, Division_ForCorrectShapes_Works) {
  // arrange
  auto t1 = Tensor(std::vector<float>({1, 2, 3, 4}), std::vector<int>({2, 2}));
  auto t2 = Tensor(std::vector<float>({10}), std::vector<int>({1}));

  // act
  auto result = t1 / t2;
  result.backward();

  // assert
  ExpectVectorsNear(result.data(), std::vector<float>({0.1, 0.2, 0.3, 0.4}));
  ExpectVectorsNear(t1.grad(), std::vector<float>({0.1, 0.1, 0.1, 0.1}));
  ExpectVectorsNear(t2.grad(), std::vector<float>({-0.1}));
}

TEST(TensorTest, MatrixMultiplication_ForMatchingShapes_Works) {
  // arrange
  auto t1 = Tensor({1.0, 2.0}, {1, 2});
  auto t2 = Tensor({8.0, 6.0, 4.0, 2.0}, {2, 2});

  // act
  auto result = t1 & t2;

  // assert
  ExpectVectorsNear(result.data(), std::vector<float>({16.0, 10.0}));
  EXPECT_EQ(result.shape(), std::vector<int>({1, 2}));
}

TEST(TensorTest, MatrixMultiplication_ForVectorAndVector_Works) {
  // arrange
  auto t1 = Tensor({1.0, 2.0}, {1, 2});
  auto t2 = Tensor({1.0, 2.0}, {2, 1});

  // act
  auto result = t1 & t2;

  // assert
  ExpectVectorsNear(result.data(), std::vector<float>({5}));
  EXPECT_EQ(result.shape(), std::vector<int>({1, 1}));
}

TEST(TensorTest, MatrixMultiplication_ForMatchingShapes_AssignsCorrectShape) {
  // arrange
  auto t1 = zeros({1, 10});
  auto t2 = zeros({10, 5});

  // act
  auto result = t1 & t2;

  // assert
  EXPECT_EQ(result.shape(), std::vector<int>({1, 5}));
}

TEST(TensorTest, MatrixMultiplicationBackward_ForMatchingShapes_Works) {
  // arrange
  auto t1 = Tensor({1.0, 2.0}, {1, 2});
  auto t2 = Tensor({8.0, 6.0, 4.0, 2.0}, {2, 2});

  // act
  auto result = t1 & t2;
  result.backward();

  // assert
  ExpectVectorsNear(t1.grad(), std::vector<float>({14.0, 6.0}));
  ExpectVectorsNear(t2.grad(), std::vector<float>({1.0, 1.0, 2.0, 2.0}));
}

TEST(TensorTest, MatrixMultiplication_ForTensors_Works) {
  throw std::runtime_error("Test not working");
  // arrange
  auto t1 = Tensor({8, 6, 4, 2, 0, -2, -4, -6}, {2, 2, 2});
  auto t2 = Tensor({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});

  // act
  auto result = t1 & t2;
  result.backward();

  // assert
  ExpectVectorsNear(result.data(),
                    std::vector<float>({26, 40, 10, 16, -14, -16, -62, -72}));
  EXPECT_EQ(result.shape(), std::vector<int>({2, 2, 2}));
  ExpectVectorsNear(t1.grad(),
                    std::vector<float>({12, 12, 8, 8, -4, -4, -8, -8}));
  ExpectVectorsNear(t2.grad(),
                    std::vector<float>({3, 7, 3, 7, 11, 15, 11, 15}));
}

TEST(TensorTest, MatrixMultiplication_ForTensorAndMatrix_Works) {
  // arrange
  auto t1 = Tensor({8, 6, 4, 2, 0, -2, -4, -6}, {2, 2, 2});
  auto t2 = Tensor({1, 2, 3, 4}, {2, 2});

  // act
  auto result = t1 & t2;
  result.backward();

  // assert
  ExpectVectorsNear(result.data(),
                    std::vector<float>({26, 40, 10, 16, -6, -8, -22., -32}));
  EXPECT_EQ(result.shape(), std::vector<int>({2, 2, 2}));
  ExpectVectorsNear(t1.grad(), std::vector<float>({3, 7, 3, 7, 3, 7, 3, 7}));
  ExpectVectorsNear(t2.grad(), std::vector<float>({8, 8, 0, 0}));
}

TEST(TensorTest, backward_ForSingleValueTensors_Works) {
  // arrange
  auto x1 = Tensor({2.0}, {1});
  auto x2 = Tensor({0.0}, {1});
  auto w1 = Tensor({-3.0}, {1});
  auto w2 = Tensor({1.0}, {1});
  auto b = Tensor({6.88137}, {1});
  auto x1w1 = x1 * w1;
  auto x2w2 = x2 * w2;
  auto x1w1_x2w2 = x1w1 + x2w2;
  auto n = x1w1_x2w2 + b;
  auto o = tanh(n);

  auto eps = 1.0e-5;

  // act
  o.backward();

  // assert
  EXPECT_NEAR(o.grad()[0], 1.0, eps);
  EXPECT_NEAR(n.grad()[0], 0.5, eps);
  EXPECT_NEAR(b.grad()[0], 0.5, eps);
  EXPECT_NEAR(x1w1_x2w2.grad()[0], 0.5, eps);
  EXPECT_NEAR(x1w1.grad()[0], 0.5, eps);
  EXPECT_NEAR(x2w2.grad()[0], 0.5, eps);
  EXPECT_NEAR(w2.grad()[0], 0.0, eps);
  EXPECT_NEAR(x2.grad()[0], 0.5, eps);
  EXPECT_NEAR(w1.grad()[0], 1.0, eps);
  EXPECT_NEAR(x1.grad()[0], -1.5, eps);
}

TEST(TensorTest, backward_ForBranchingModel_Works) {
  // arrange
  auto x0 = Tensor({-2.0}, {1});
  auto w1 = Tensor({-3.0}, {1});
  auto x1 = x0 * w1;
  auto w2 = Tensor({4.0}, {1});
  auto x1w1 = x1 * w1;
  auto x1w2 = x1 * w2;
  auto x1w1_x1w2 = x1w1 + x1w2;

  auto eps = 1.0e-5;

  // act
  x1w1_x1w2.backward();

  // assert
  EXPECT_NEAR(x1w1_x1w2.grad()[0], 1, eps);
  EXPECT_NEAR(x1w2.grad()[0], 1, eps);
  EXPECT_NEAR(x1w1.grad()[0], 1, eps);
  EXPECT_NEAR(w2.grad()[0], 6, eps);
  EXPECT_NEAR(w1.grad()[0], 4, eps);
  EXPECT_NEAR(x1.grad()[0], 1, eps);
  EXPECT_NEAR(x0.grad()[0], -3, eps);
}

TEST(TensorTest, Backward_ForLinearLayerBatchedData_Works) {
  // arrange
  auto w = Tensor({0.1, 0.2, 0.3, 0.4}, {2, 2});
  auto b = Tensor({0.1, 0.2}, {2});
  auto x = Tensor({0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, {3, 2});
  auto y = Tensor({-0.1, 0.2, -0.3, 0.4, -0.5, 0.6}, {3, 2});

  // act
  auto wx = x & w;
  auto wx_b = wx + b;
  auto o = tanh(wx_b);
  auto loss = nn::functional::mse_loss(o, y, "sum");
  loss.backward();

  // assert
  ExpectVectorsNear(wx.data(), {0.07, 0.1, 0.15, 0.22, 0.23, 0.34});
  ExpectVectorsNear(wx_b.data(), {0.17, 0.30, 0.25, 0.42, 0.33, 0.54});
  ExpectVectorsNear(o.data(), {0.1684, 0.2913, 0.2449, 0.3969, 0.3185, 0.4930});
  ExpectVectorsNear(loss.data(), {1.0587});

  ExpectVectorsNear(w.grad(), {1.0950, -0.0658, 1.3967, -0.0658});
  ExpectVectorsNear(b.grad(), {3.0169615, -5.3018e-5}, 1.0e-7);
}

TEST(TensorTest, CopyConstructor_CopiesData_1to1) {
  // arrange
  auto t = Tensor({1.0, 2.0, 3.0, 4.0}, {2, 2});
  t.grad() = {4.0, 3.0, 2.0, 1.0};

  // act
  auto copy = new Tensor(t);

  // assert
  ExpectVectorsNear(t.data(), copy->data());
  ExpectVectorsNear(t.grad(), copy->grad());
}

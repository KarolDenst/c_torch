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
  result.backward(false);

  // assert
  ExpectVectorsNear(result.data, std::vector<float>({3, 6, 9, 12}));
  ExpectVectorsNear(t1.grad, std::vector<float>({1, 1, 1, 1}));
  ExpectVectorsNear(t2.grad, std::vector<float>({1, 1, 1, 1}));
}

TEST(TensorTest, Addition_ForFirstLargerThanSecond_BoardcastingWorks) {
  // arrange
  auto t1 = Tensor(std::vector<float>({1, 2, 3, 4}), std::vector<int>({2, 2}));
  auto t2 = Tensor(std::vector<float>({5, 6}), std::vector<int>({2}));

  // act
  auto result = t1 + t2;
  auto result2 = t2 + t1;
  result.backward(false);

  // assert
  ExpectVectorsNear(result.data, std::vector<float>({6, 8, 8, 10}));
  ExpectVectorsNear(result.data, result2.data);
  ExpectVectorsNear(t1.grad, std::vector<float>({1, 1, 1, 1}));
  ExpectVectorsNear(t2.grad, std::vector<float>({2, 2}));
}

TEST(TensorTest, Subtraction_ForMatchingShapes_Works) {
  // arrange
  auto t1 = Tensor(std::vector<float>({1, 2, 3, 4}), std::vector<int>({2, 2}));
  auto t2 = Tensor(std::vector<float>({2, 4, 6, 8}), std::vector<int>({2, 2}));

  // act
  auto result = t1 - t2;
  result.backward(false);

  // assert
  ExpectVectorsNear(result.data, std::vector<float>({-1, -2, -3, -4}));
  ExpectVectorsNear(t1.grad, std::vector<float>({1, 1, 1, 1}));
  ExpectVectorsNear(t2.grad, std::vector<float>({-1, -1, -1, -1}));
}

TEST(TensorTest, Multiplication_ForMatchingShapes_Works) {
  // arrange
  auto t1 = Tensor(std::vector<float>({1, 2, 3, 4}), std::vector<int>({2, 2}));
  auto t2 = Tensor(std::vector<float>({2, 4, 6, 8}), std::vector<int>({2, 2}));

  // act
  auto result = t1 * t2;
  result.backward(false);

  // assert
  ExpectVectorsNear(result.data, std::vector<float>({2, 8, 18, 32}));
  ExpectVectorsNear(t1.grad, std::vector<float>({2, 4, 6, 8}));
  ExpectVectorsNear(t2.grad, std::vector<float>({1, 2, 3, 4}));
}

TEST(TensorTest, Division_ForMatchingShapes_Works) {
  // arrange
  auto t1 = Tensor(std::vector<float>({1, 2, 3, 4}), std::vector<int>({2, 2}));
  auto t2 = Tensor(std::vector<float>({2, 4, 6, 8}), std::vector<int>({2, 2}));

  // act
  auto result = t1 / t2;
  result.backward(false);

  // assert
  ExpectVectorsNear(result.data, std::vector<float>({0.5, 0.5, 0.5, 0.5}));
  ExpectVectorsNear(t1.grad,
                    std::vector<float>({1.0 / 2, 1.0 / 4, 1.0 / 6, 1.0 / 8}));
  ExpectVectorsNear(
      t2.grad, std::vector<float>({-1.0 / 4, -1.0 / 8, -1.0 / 12, -1.0 / 16}));
}

TEST(TensorTest, Division_ForCorrectShapes_Works) {
  // arrange
  auto t1 = Tensor(std::vector<float>({1, 2, 3, 4}), std::vector<int>({2, 2}));
  auto t2 = Tensor(std::vector<float>({10}), std::vector<int>({1}));

  // act
  auto result = t1 / t2;
  result.backward(false);

  // assert
  ExpectVectorsNear(result.data, std::vector<float>({0.1, 0.2, 0.3, 0.4}));
  ExpectVectorsNear(t1.grad, std::vector<float>({0.1, 0.1, 0.1, 0.1}));
  ExpectVectorsNear(t2.grad, std::vector<float>({-0.1}));
}

TEST(TensorTest, MatrixMultiplication_ForMatchingShapes_Works) {
  // arrange
  auto t1 = Tensor({1.0, 2.0}, {1, 2});
  auto t2 = Tensor({8.0, 6.0, 4.0, 2.0}, {2, 2});

  // act
  auto result = t1 & t2;

  // assert
  ExpectVectorsNear(result.data, std::vector<float>({16.0, 10.0}));
  EXPECT_EQ(result.shape, std::vector<int>({1, 2}));
}

TEST(TensorTest, MatrixMultiplication_ForVectorAndVector_Works) {
  // arrange
  auto t1 = Tensor({1.0, 2.0}, {1, 2});
  auto t2 = Tensor({1.0, 2.0}, {2, 1});

  // act
  auto result = t1 & t2;

  // assert
  ExpectVectorsNear(result.data, std::vector<float>({5}));
  EXPECT_EQ(result.shape, std::vector<int>({1, 1}));
}

TEST(TensorTest, MatrixMultiplication_ForMatchingShapes_AssignsCorrectShape) {
  // arrange
  auto t1 = zeros({1, 10});
  auto t2 = zeros({10, 5});

  // act
  auto result = t1 & t2;

  // assert
  EXPECT_EQ(result.shape, std::vector<int>({1, 5}));
}

TEST(TensorTest, MatrixMultiplicationBackward_ForMatchingShapes_Works) {
  // arrange
  auto t1 = Tensor({1.0, 2.0}, {1, 2});
  auto t2 = Tensor({8.0, 6.0, 4.0, 2.0}, {2, 2});

  // act
  auto result = t1 & t2;
  result.backward(false);

  // assert
  ExpectVectorsNear(t1.grad, std::vector<float>({14.0, 6.0}));
  ExpectVectorsNear(t2.grad, std::vector<float>({1.0, 1.0, 2.0, 2.0}));
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
  auto o = tanh(&n);

  auto eps = 1.0e-5;

  // act
  o.backward(false);

  // assert
  EXPECT_NEAR(o.grad[0], 1.0, eps);
  EXPECT_NEAR(n.grad[0], 0.5, eps);
  EXPECT_NEAR(b.grad[0], 0.5, eps);
  EXPECT_NEAR(x1w1_x2w2.grad[0], 0.5, eps);
  EXPECT_NEAR(x1w1.grad[0], 0.5, eps);
  EXPECT_NEAR(x2w2.grad[0], 0.5, eps);
  EXPECT_NEAR(w2.grad[0], 0.0, eps);
  EXPECT_NEAR(x2.grad[0], 0.5, eps);
  EXPECT_NEAR(w1.grad[0], 1.0, eps);
  EXPECT_NEAR(x1.grad[0], -1.5, eps);
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
  x1w1_x1w2.backward(false);

  // assert
  EXPECT_NEAR(x1w1_x1w2.grad[0], 1, eps);
  EXPECT_NEAR(x1w2.grad[0], 1, eps);
  EXPECT_NEAR(x1w1.grad[0], 1, eps);
  EXPECT_NEAR(w2.grad[0], 6, eps);
  EXPECT_NEAR(w1.grad[0], 4, eps);
  EXPECT_NEAR(x1.grad[0], 1, eps);
  EXPECT_NEAR(x0.grad[0], -3, eps);
}

TEST(TensorTest, CopyConstructor_CopiesData_1to1) {
  // arrange
  auto t = Tensor({1.0, 2.0, 3.0, 4.0}, {2, 2});
  t.grad = {4.0, 3.0, 2.0, 1.0};

  // act
  auto copy = new Tensor(t);

  // assert
  ExpectVectorsNear(t.data, copy->data);
  ExpectVectorsNear(t.grad, copy->grad);
  EXPECT_EQ(t.is_tmp, copy->is_tmp);
}

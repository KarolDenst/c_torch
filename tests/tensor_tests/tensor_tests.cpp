#include <cmath>
#include <gtest/gtest.h>
#include <tensor/tensor.h>

void ExpectVectorsNear(const std::vector<float> &expected,
                       const std::vector<float> &actual) {
  auto tolerance = 1.0e-4f;
  ASSERT_EQ(expected.size(), actual.size()) << "Vectors are of unequal length";

  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(expected[i], actual[i], tolerance)
        << "Vectors differ at index " << i;
  }
}

TEST(TensorTest, Addition_ForMatchingShapes_Works) {
  // arrange
  auto t1 = Tensor(std::vector<float>({1, 2, 3, 4}), std::vector<int>({2, 2}));
  auto t2 = Tensor(std::vector<float>({2, 4, 6, 8}), std::vector<int>({2, 2}));

  // act
  auto result = t1 + t2;
  result.backwards();

  // assert
  ExpectVectorsNear(result.data, std::vector<float>({3, 6, 9, 12}));
  ExpectVectorsNear(t1.grad, std::vector<float>({1, 1, 1, 1}));
  ExpectVectorsNear(t2.grad, std::vector<float>({1, 1, 1, 1}));
}

TEST(TensorTest, Subtraction_ForMatchingShapes_Works) {
  // arrange
  auto t1 = Tensor(std::vector<float>({1, 2, 3, 4}), std::vector<int>({2, 2}));
  auto t2 = Tensor(std::vector<float>({2, 4, 6, 8}), std::vector<int>({2, 2}));

  // act
  auto result = t1 - t2;
  result.backwards();

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
  result.backwards();

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
  result.backwards();

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
  result.backwards();

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
}

TEST(TensorTest, MatrixMultiplicationBackward_ForMatchingShapes_Works) {
  // arrange
  auto t1 = Tensor({1.0, 2.0}, {1, 2});
  auto t2 = Tensor({8.0, 6.0, 4.0, 2.0}, {2, 2});

  // act
  auto result = t1 & t2;
  result.backwards();

  // assert
  ExpectVectorsNear(t1.grad, std::vector<float>({14.0, 6.0}));
  ExpectVectorsNear(t2.grad, std::vector<float>({1.0, 1.0, 2.0, 2.0}));
}

TEST(TensorTest, Tanh_Works) {
  // arrange
  auto t = Tensor({1.0, 2.0, 3.0, 4.0}, {2, 2});

  // act
  auto result = t.tanh();
  result.backwards();

  // assert
  ExpectVectorsNear(result.data,
                    std::vector<float>({std::tanh(1.0f), std::tanh(2.0f),
                                        std::tanh(3.0f), std::tanh(4.0f)}));
  ExpectVectorsNear(t.grad,
                    std::vector<float>({0.4200, 0.0707, 0.0099, 0.0013}));
}

TEST(TensorTest, Exp_Works) {
  // arrange
  auto t = Tensor({1.0, 2.0, 3.0, 4.0}, {2, 2});

  // act
  auto result = t.exp();
  result.backwards();

  // assert
  ExpectVectorsNear(result.data,
                    std::vector<float>({std::exp(1.0f), std::exp(2.0f),
                                        std::exp(3.0f), std::exp(4.0f)}));
  ExpectVectorsNear(t.grad, result.data);
}

TEST(TensorTest, Log_Works) {
  // arrange
  auto t = Tensor({1.0, 2.0, 3.0, 4.0}, {2, 2});

  // act
  auto result = t.log();
  result.backwards();

  // assert
  ExpectVectorsNear(result.data,
                    std::vector<float>({std::log(1.0f), std::log(2.0f),
                                        std::log(3.0f), std::log(4.0f)}));
  ExpectVectorsNear(t.grad,
                    std::vector<float>({1.0, 1.0 / 2, 1.0 / 3, 1.0 / 4}));
}

TEST(TensorTest, Sum_Works) {
  // arrange
  auto t = Tensor({1.0, 2.0, 3.0, 4.0}, {2, 2});

  // act
  auto result = t.sum();
  result.backwards();

  // assert
  ExpectVectorsNear(result.data, std::vector<float>({10.0}));
  ExpectVectorsNear(t.grad, std::vector<float>({1.0, 1.0, 1.0, 1.0}));
}

TEST(TensorTest, Backwards_ForSingleValueTensors_Works) {
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
  auto o = n.tanh();

  auto eps = 1.0e-5;

  // act
  o.backwards();

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

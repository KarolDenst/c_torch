#include <gtest/gtest.h>
#include <tensor/tensor.h>

TEST(TensorTest, Addition_ForMatchingShapes_Works) {
  // arrange
  auto t1 = Tensor(std::vector<float>({1, 2}), std::vector<int>({1}));
  auto t2 = Tensor(std::vector<float>({2, 4}), std::vector<int>({1}));

  // act
  auto sum = t1 + t2;

  // assert
  EXPECT_EQ(sum.data, std::vector<float>({3, 6}));
}

TEST(TensorTest, Multiplication_ForMatchingShapes_Works) {
  // arrange
  auto t1 = Tensor(std::vector<float>({1, 2}), std::vector<int>({1}));
  auto t2 = Tensor(std::vector<float>({2, 4}), std::vector<int>({1}));

  // act
  auto sum = t1 * t2;

  // assert
  EXPECT_EQ(sum.data, std::vector<float>({2, 8}));
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

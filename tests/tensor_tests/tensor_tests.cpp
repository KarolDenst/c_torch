#include <gtest/gtest.h>
#include <tensor/tensor.h>

TEST(SampleTest, AssertionTrue) { EXPECT_TRUE(true); }

TEST(TensorTest, ConstructorTest) {
  auto t1 = Tensor(std::vector<float>({1}), std::vector<int>({1}));
  EXPECT_TRUE(true);
}

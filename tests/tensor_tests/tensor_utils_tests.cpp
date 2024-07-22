#include "../../src/tensor/tensor.h"
#include "../../src/tensor/tensor_utils.h"
#include "tensor_utils.h"
#include <gtest/gtest.h>

using namespace tensor;

TEST(TensorFunTest, Stack_Works) {
  // arrange
  auto t1 = Tensor({1.0, 2.0, 3.0, 4.0}, {2, 2});
  auto t2 = Tensor({2.0, 4.0, 6.0, 6.0}, {2, 2});

  // act
  auto result = stack({&t1, &t2});

  // assert
  ExpectVectorsNear(result.data, std::vector<float>(
                                     {1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 6.0}));
  EXPECT_EQ(result.shape, std::vector<int>({2, 2, 2}));
}

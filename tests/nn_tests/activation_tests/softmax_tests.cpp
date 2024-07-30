#include "../../../../src/nn/activation/softmax.h"
#include "../../../../src/tensor/tensor.h"
#include "../../tensor_tests/tensor_utils.h"
#include <gtest/gtest.h>

TEST(SoftmaxTest, Softmax_WithoutDim_Works) {
  // arrange
  auto t1 = tensor::Tensor(std::vector<float>({1, 2, 3, 4}),
                           std::vector<int>({2, 2}));
  auto softmax = nn::activation::Softmax();

  // act
  auto result = softmax.forward(t1);
  result.backward();

  // assert
  ExpectVectorsNear(result.data(),
                    std::vector<float>({0.0321, 0.0871, 0.2369, 0.6439}));
}

TEST(SoftmaxTest, Softmax_ForSpecificDim_Works) {
  // arrange
  auto t1 = tensor::Tensor(std::vector<float>({1, 2, 3, 4}),
                           std::vector<int>({2, 2}));
  auto softmax = nn::activation::Softmax(1);

  // act
  auto result = softmax.forward(t1);
  result.backward();

  // assert
  ExpectVectorsNear(result.data(),
                    std::vector<float>({0.2689, 0.7311, 0.2689, 0.7311}));
}

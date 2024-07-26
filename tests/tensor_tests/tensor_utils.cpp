#include "tensor_utils.h"
#include <gtest/gtest.h>

void ExpectVectorsNear(const std::vector<float> &actual,
                       const std::vector<float> &expected, float tolerance) {
  ASSERT_EQ(expected.size(), actual.size()) << "Vectors are of unequal length";

  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], tolerance)
        << "Vectors differ at index " << i;
  }
}

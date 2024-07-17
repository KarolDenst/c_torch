#include "tensor_utils.h"
#include <gtest/gtest.h>


void ExpectVectorsNear(const std::vector<float> &expected,
                       const std::vector<float> &actual) {
  auto tolerance = 1.0e-4f;
  ASSERT_EQ(expected.size(), actual.size()) << "Vectors are of unequal length";

  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(expected[i], actual[i], tolerance)
        << "Vectors differ at index " << i;
  }
}

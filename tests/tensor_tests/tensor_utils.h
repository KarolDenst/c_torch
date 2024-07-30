#include <vector>

void ExpectVectorsNear(const std::vector<float> &actual,
                       const std::vector<float> &expected,
                       float tolerance = 1.0e-4f);

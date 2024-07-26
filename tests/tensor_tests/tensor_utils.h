#include <vector>

void ExpectVectorsNear(const std::vector<float> &expected,
                       const std::vector<float> &actual,
                       float tolerance = 1.0e-4f);

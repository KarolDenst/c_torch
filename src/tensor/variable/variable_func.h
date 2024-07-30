#ifndef VARIABLE_FUNC_H
#define VARIABLE_FUNC_H

#include "variable.h"
#include <optional>

namespace variable {

std::shared_ptr<Variable> tanh(std::shared_ptr<Variable> variable);
std::shared_ptr<Variable> relu(std::shared_ptr<Variable> variable);
std::shared_ptr<Variable> log(std::shared_ptr<Variable> variable);
std::shared_ptr<Variable> exp(std::shared_ptr<Variable> variable);
std::shared_ptr<Variable> sum(std::shared_ptr<Variable> variable,
                              std::optional<int> dim = std::nullopt,
                              bool keepdim = false);
std::shared_ptr<Variable> mean(std::shared_ptr<Variable> variable);

} // namespace variable

#endif // VARIABLE_FUNC_H

#ifndef VARIABLE_H
#define VARIABLE_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

#define ROW_COL_PARALLEL_INNER_TILING_TILE_SIZE 16
#define EPS 0.0000001f

namespace variable {

class Variable {
public:
  std::vector<float> data;
  std::vector<int> shape;
  std::vector<int> strides;
  std::vector<float> grad;
  std::string name = "";
  std::function<void(void)> back;
  std::vector<std::shared_ptr<Variable>> prev;

  Variable(std::vector<float> data, std::vector<int> shape,
           std::string name = "");
  Variable(std::vector<float> data, std::vector<int> shape,
           std::vector<std::shared_ptr<Variable>> prev, std::string name = "");

  float &get(std::initializer_list<int> args);

  void print(bool print_prev = false);
  static std::shared_ptr<Variable> add(std::shared_ptr<Variable> first,
                                       std::shared_ptr<Variable> second);
  static std::shared_ptr<Variable> sub(std::shared_ptr<Variable> first,
                                       std::shared_ptr<Variable> second);
  static std::shared_ptr<Variable> mul(std::shared_ptr<Variable> first,
                                       std::shared_ptr<Variable> second);
  static std::shared_ptr<Variable> div(std::shared_ptr<Variable> first,
                                       std::shared_ptr<Variable> second);
  static std::shared_ptr<Variable> mat_mul(std::shared_ptr<Variable> first,
                                           std::shared_ptr<Variable> second);
  static std::shared_ptr<Variable> greater(std::shared_ptr<Variable> variable,
                                           float val);
  void view(std::vector<int> shape);
  void backward();

private:
  static std::vector<int> compute_strides(std::vector<int> shape);
  static std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
  compute_broadcast_strides(Variable &first, Variable &second);
  static std::shared_ptr<Variable>
  transform(std::shared_ptr<Variable> first, std::shared_ptr<Variable> second,
            void (*front)(Variable *, Variable *, Variable *, int, int, int),
            void (*back)(Variable *, Variable *, Variable *, int, int, int),
            std::string name = "");
  static void transform_rec(int, int, int, int, Variable *, Variable *,
                            Variable *, std::vector<int> &, std::vector<int> &,
                            void (*front)(Variable *, Variable *, Variable *,
                                          int, int, int));
};

// https://siboehm.com/articles/22/Fast-MMM-on-CPU
template <bool transpose1 = false, bool transpose2 = false,
          bool transpose3 = false>
inline void
fast_mat_mul(const float *left, const float *right, float *result, int rows,
             int columns, int inners,
             int tileSize = ROW_COL_PARALLEL_INNER_TILING_TILE_SIZE);

} // namespace variable

#endif // VARIABLE_H

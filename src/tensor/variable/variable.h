#ifndef VARIABLE_H
#define VARIABLE_H

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#define ROW_COL_PARALLEL_INNER_TILING_TILE_SIZE 16
#define EPS 0.0000001f

namespace variable {

template <typename DType>
concept Numeric = std::is_arithmetic_v<DType>;

template <Numeric DType = float> class Variable {
public:
  std::vector<DType> data;
  std::vector<int> shape;
  std::vector<int> strides;
  std::vector<DType> grad;
  std::string name = "";
  std::function<void(void)> back;
  std::vector<std::shared_ptr<Variable<DType>>> prev;

  Variable<DType>(std::vector<DType> data, std::vector<int> shape,
                  std::string name = "");
  Variable<DType>(std::vector<DType> data, std::vector<int> shape,
                  std::vector<std::shared_ptr<Variable<DType>>> prev,
                  std::string name = "");

  DType &get(std::initializer_list<int> args);

  void print(bool print_prev = false);

  static std::shared_ptr<Variable<DType>>
  add(std::shared_ptr<Variable<DType>> first,
      std::shared_ptr<Variable<DType>> second);

  static std::shared_ptr<Variable<DType>>
  sub(std::shared_ptr<Variable<DType>> first,
      std::shared_ptr<Variable<DType>> second);

  static std::shared_ptr<Variable<DType>>
  mul(std::shared_ptr<Variable<DType>> first,
      std::shared_ptr<Variable<DType>> second);

  static std::shared_ptr<Variable<DType>>
  div(std::shared_ptr<Variable<DType>> first,
      std::shared_ptr<Variable<DType>> second);

  static std::shared_ptr<Variable<DType>>
  mat_mul(std::shared_ptr<Variable<DType>> first,
          std::shared_ptr<Variable<DType>> second);

  static std::shared_ptr<Variable<DType>>
  greater(std::shared_ptr<Variable<DType>> variable, float val);

  void view(std::vector<int> shape);

  void backward();

private:
  static std::vector<int> compute_strides(std::vector<int> shape);

  static std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>

  compute_broadcast_strides(Variable<DType> &first, Variable<DType> &second);

  static std::shared_ptr<Variable<DType>>
  transform(std::shared_ptr<Variable<DType>> first,
            std::shared_ptr<Variable<DType>> second,
            void (*front)(Variable<DType> *, Variable<DType> *,
                          Variable<DType> *, int, int, int),
            void (*back)(Variable<DType> *, Variable<DType> *,
                         Variable<DType> *, int, int, int),
            std::string name = "");

  static void transform_rec(int, int, int, int, Variable<DType> *,
                            Variable<DType> *, Variable<DType> *,
                            std::vector<int> &, std::vector<int> &,
                            void (*front)(Variable<DType> *, Variable<DType> *,
                                          Variable<DType> *, int, int, int));

  // https://siboehm.com/articles/22/Fast-MMM-on-CPU
  template <bool transpose1 = false, bool transpose2 = false,
            bool transpose3 = false>
  static inline void
  fast_mat_mul(const DType *left, const DType *right, DType *result, int rows,
               int columns, int inners,
               int tileSize = ROW_COL_PARALLEL_INNER_TILING_TILE_SIZE);
};

template <Numeric DType>
Variable<DType>::Variable(std::vector<DType> data, std::vector<int> shape,
                          std::string name)
    : data(data), shape(shape), strides(compute_strides(shape)), name(name),
      prev(std::vector<std::shared_ptr<Variable<DType>>>()),
      grad(std::vector<DType>(data.size())), back([]() {}) {}

template <Numeric DType>
Variable<DType>::Variable(std::vector<DType> data, std::vector<int> shape,
                          std::vector<std::shared_ptr<Variable<DType>>> prev,
                          std::string name)
    : data(data), shape(shape), strides(compute_strides(shape)), name(name),
      prev(prev), grad(std::vector<DType>(data.size())), back([]() {}) {}

template <Numeric DType>
DType &Variable<DType>::get(std::initializer_list<int> args) {
  assert(args.size() == shape.size());
  int index = 0;
  int i = 0;
  for (auto it = args.begin(); it != args.end(); it++) {
    index += *it * strides[i++];
  }
  return data[index];
}

template <Numeric DType> void Variable<DType>::print(bool print_prev) {
  std::cout << name;
  std::cout << std::endl << "Data: ";
  for (int i = 0; i < std::min(static_cast<int>(this->data.size()), 10); i++) {
    std::cout << this->data[i] << " ";
  }
  if (this->grad.size() > 10) {
    std::cout << "...";
  }
  std::cout << std::endl << "Shape: ";
  for (auto i : this->shape) {
    std::cout << i << " ";
  }
  std::cout << std::endl << "Grad: ";
  for (int i = 0; i < std::min(static_cast<int>(this->grad.size()), 10); i++) {
    std::cout << this->grad[i] << " ";
  }
  if (this->grad.size() > 10) {
    std::cout << "...";
  }
  if (print_prev) {
    std::cout << "\nPrev: \n";
    for (auto &t : this->prev) {
      t->print(print_prev);
    }
  }
  std::cout << std::endl;
  std::cout << std::endl;
}

template <Numeric DType>
std::shared_ptr<Variable<DType>>
Variable<DType>::add(std::shared_ptr<Variable<DType>> first,
                     std::shared_ptr<Variable<DType>> second) {
  auto front = [](Variable<DType> *first, Variable<DType> *second,
                  Variable<DType> *out, int i, int j,
                  int k) { out->data[k] = first->data[i] + second->data[j]; };
  auto back = [](Variable<DType> *first, Variable<DType> *second,
                 Variable<DType> *out, int i, int j, int k) {
    first->grad[i] += out->grad[k];
    second->grad[j] += out->grad[k];
  };
  return transform(first, second, front, back, "+");
}

template <Numeric DType>
std::shared_ptr<Variable<DType>>
Variable<DType>::sub(std::shared_ptr<Variable<DType>> first,
                     std::shared_ptr<Variable<DType>> second) {
  auto front = [](Variable<DType> *first, Variable<DType> *second,
                  Variable<DType> *out, int i, int j,
                  int k) { out->data[k] = first->data[i] - second->data[j]; };
  auto back = [](Variable<DType> *first, Variable<DType> *second,
                 Variable<DType> *out, int i, int j, int k) {
    first->grad[i] += out->grad[k];
    second->grad[j] -= out->grad[k];
  };
  return transform(first, second, front, back, "-");
}

template <Numeric DType>
std::shared_ptr<Variable<DType>>
Variable<DType>::mul(std::shared_ptr<Variable<DType>> first,
                     std::shared_ptr<Variable<DType>> second) {
  auto front = [](Variable<DType> *first, Variable<DType> *second,
                  Variable<DType> *out, int i, int j,
                  int k) { out->data[k] = first->data[i] * second->data[j]; };
  auto back = [](Variable<DType> *first, Variable<DType> *second,
                 Variable<DType> *out, int i, int j, int k) {
    first->grad[i] += second->data[j] * out->grad[k];
    second->grad[j] += first->data[i] * out->grad[k];
  };
  return transform(first, second, front, back, "*");
}

template <Numeric DType>
std::shared_ptr<Variable<DType>>
Variable<DType>::div(std::shared_ptr<Variable<DType>> first,
                     std::shared_ptr<Variable<DType>> second) {
  auto front = [](Variable<DType> *first, Variable<DType> *second,
                  Variable<DType> *out, int i, int j, int k) {
    out->data[k] = first->data[i] / second->data[j] + EPS;
  };
  auto back = [](Variable<DType> *first, Variable<DType> *second,
                 Variable<DType> *out, int i, int j, int k) {
    first->grad[i] += 1.0 / (second->data[j]) * out->grad[k] + EPS;
    second->grad[j] +=
        -first->data[i] / (second->data[j] * second->data[j]) * out->grad[k] +
        EPS;
  };
  return transform(first, second, front, back, "/");
}

template <Numeric DType>
std::shared_ptr<Variable<DType>>
Variable<DType>::mat_mul(std::shared_ptr<Variable<DType>> first,
                         std::shared_ptr<Variable<DType>> second) {
  assert(first->shape.back() == second->shape.front());
  auto shape1 = std::vector<int>{
      static_cast<int>(first->data.size() / first->shape.back()),
      first->shape.back()};
  auto shape2 = std::vector<int>{
      second->shape.front(),
      static_cast<int>(second->data.size() / second->shape.front())};

  auto shape = std::vector<int>(first->shape.size() + second->shape.size() - 2);
  for (int i = 0; i < shape.size(); i++) {
    if (i < first->shape.size() - 1)
      shape[i] = first->shape[i];
    else {
      int j = i - first->shape.size() + 2;
      shape[i] = second->shape[j];
    }
  }
  auto data = std::vector<float>(shape1[0] * shape2[1], 0);
  fast_mat_mul(first->data.data(), second->data.data(), data.data(), shape1[0],
               shape2[1], shape1[1]);
  auto prev = std::vector<std::shared_ptr<Variable<DType>>>{first, second};
  auto out = std::make_shared<Variable<DType>>(
      data, shape, prev, first->name + " & " + second->name);

  auto backward = [out, first, second, shape1, shape2]() {
    fast_mat_mul<false, true, false>(out->grad.data(), second->data.data(),
                                     first->grad.data(), shape1[0], shape1[1],
                                     shape2[1]);

    fast_mat_mul<true, false, false>(first->data.data(), out->grad.data(),
                                     second->grad.data(), shape1[1], shape2[1],
                                     shape1[0]);
  };
  out->back = backward;

  return out;
}

template <Numeric DType>
std::shared_ptr<Variable<DType>>
Variable<DType>::greater(std::shared_ptr<Variable<DType>> variable, float val) {
  auto data = std::vector<float>(variable->data.size());
  for (int i = 0; i < variable->data.size(); i++) {
    data[i] = variable->data[i] > val;
  }

  auto prev = std::vector<std::shared_ptr<Variable<DType>>>{variable};
  auto out = std::make_shared<Variable<DType>>(
      data, variable->shape, prev, std::to_string(val) + "<" + variable->name);

  auto backward = [variable, out, val]() {
    for (int i = 0; i < out->grad.size(); i++) {
      if (out->data[i] > val) {
        variable->grad[i] += out->grad[i];
      }
    }
  };
  out->back = backward;
  return out;
}

template <Numeric DType> void Variable<DType>::view(std::vector<int> shape) {
  auto dim1 =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  auto dim2 = std::accumulate(this->shape.begin(), this->shape.end(), 1,
                              std::multiplies<int>());
  assert(dim1 == dim2);
  this->shape = shape;
}

template <Numeric DType> void Variable<DType>::backward() {
  auto topo = std::vector<Variable<DType> *>();
  auto visited = std::unordered_set<Variable<DType> *>();
  std::function<void(Variable<DType> *)> build_topo = [&](Variable<DType> *t) {
    if (visited.find(t) != visited.end()) {
      return;
    }
    visited.insert(t);
    for (const auto &p : t->prev) {
      build_topo(p.get());
    }
    topo.push_back(t);
  };

  for (int i = 0; i < this->grad.size(); i++) {
    this->grad[i] = 1;
  }
  build_topo(this);
  std::reverse(topo.begin(), topo.end());
  for (Variable<DType> *t : topo) {
    t->back();
  }
}

template <Numeric DType>
std::vector<int> Variable<DType>::compute_strides(std::vector<int> shape) {
  std::vector<int> strides(shape.size());
  strides[shape.size() - 1] = 1;
  for (int i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return strides;
}

template <Numeric DType>
std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
Variable<DType>::compute_broadcast_strides(Variable<DType> &first,
                                           Variable<DType> &second) {
  std::vector<int> out_shape = std::vector<int>();
  std::vector<int> shape1 = first.shape;
  std::vector<int> shape2 = second.shape;
  while (shape1.size() < shape2.size()) {
    shape1.insert(shape1.begin(), 1);
  }
  while (shape2.size() < shape1.size()) {
    shape2.insert(shape2.begin(), 1);
  }
  for (int i = 0; i < shape1.size(); i++) {
    if (shape1[i] != shape2[i] && shape1[i] != 1 && shape2[i] != 1) {
      for (int i = 0; i < first.shape.size(); i++)
        std::cout << first.shape[i] << " ";
      std::cout << std::endl;
      for (int i = 0; i < second.shape.size(); i++)
        std::cout << second.shape[i] << " ";
      std::cout << std::endl;
      throw std::runtime_error("Shape missmatch");
    }
    if (shape1[i] == 1) {
      out_shape.push_back(shape2[i]);
    } else {
      out_shape.push_back(shape1[i]);
    }
  }

  std::vector<int> s1 = compute_strides(shape1);
  std::vector<int> s2 = compute_strides(shape2);
  std::vector<int> stride1 = std::vector<int>(s1.size());
  std::vector<int> stride2 = std::vector<int>(s2.size());
  for (int i = 0; i < s1.size(); i++) {
    if (shape1[i] == shape2[i]) {
      stride1[i] = s1[i];
      stride2[i] = s2[i];
    } else if (shape1[i] == 1) {
      stride1[i] = 0;
      stride2[i] = s2[i];
    } else if (shape2[i] == 1) {
      stride1[i] = s1[i];
      stride2[i] = 0;
    } else {
      throw std::runtime_error("Invalid shape");
    }
  }

  return std::make_tuple(out_shape, stride1, stride2);
}

template <Numeric DType>
std::shared_ptr<Variable<DType>>
Variable<DType>::transform(std::shared_ptr<Variable<DType>> first,
                           std::shared_ptr<Variable<DType>> second,
                           void (*front)(Variable<DType> *, Variable<DType> *,
                                         Variable<DType> *, int, int, int),
                           void (*back)(Variable<DType> *, Variable<DType> *,
                                        Variable<DType> *, int, int, int),
                           std::string name) {
  auto tuple = compute_broadcast_strides(*first, *second);
  auto out_shape = std::get<0>(tuple);
  auto stride1 = std::get<1>(tuple);
  auto stride2 = std::get<2>(tuple);

  auto data_size = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                   std::multiplies<int>());
  auto prev = std::vector<std::shared_ptr<Variable<DType>>>{first, second};
  auto out = std::make_shared<Variable<DType>>(
      std::vector<DType>(data_size), out_shape, prev,
      first->name + name + second->name);

  transform_rec(0, 0, 0, 0, out.get(), first.get(), second.get(), stride1,
                stride2, front);

  auto backward = [first, second, out, back]() {
    auto tuple = compute_broadcast_strides(*first, *second);
    auto stride1 = std::get<1>(tuple);
    auto stride2 = std::get<2>(tuple);
    transform_rec(0, 0, 0, 0, out.get(), first.get(), second.get(), stride1,
                  stride2, back);
  };
  out->back = backward;

  return out;
}

template <Numeric DType>
void Variable<DType>::transform_rec(
    int dim, int offset1, int offset2, int index, Variable<DType> *out,
    Variable<DType> *first, Variable<DType> *second, std::vector<int> &stride1,
    std::vector<int> &stride2,
    void (*func)(Variable<DType> *, Variable<DType> *, Variable<DType> *, int,
                 int, int)) {
  if (dim >= out->shape.size()) {
    func(first, second, out, offset1, offset2, index);
    return;
  }
  int s1 = 0, s2 = 0;
  for (int i = 0; i < out->shape[dim]; i++) {
    transform_rec(dim + 1, offset1 + s1, offset2 + s2, index, out, first,
                  second, stride1, stride2, func);
    index += out->strides[dim];
    s1 += stride1[dim];
    s2 += stride2[dim];
  }
};

template <Numeric DType>
template <bool transpose1, bool transpose2, bool transpose3>
inline void Variable<DType>::fast_mat_mul(const DType *left, const DType *right,
                                          DType *result, int rows, int columns,
                                          int inners, int tileSize) {
#pragma omp parallel for shared(result, left, right) default(none) collapse(2) \
    num_threads(8)
  for (int rowTile = 0; rowTile < rows; rowTile += 256) {
    for (int columnTile = 0; columnTile < columns; columnTile += 256) {
      for (int innerTile = 0; innerTile < inners; innerTile += tileSize) {
        int rowTileEnd = std::min(rows, rowTile + 256);
        int colTileEnd = std::min(columns, columnTile + 256);
        int innerTileEnd = std::min(inners, innerTile + tileSize);
        for (int row = rowTile; row < rowTileEnd; row++) {
          for (int inner = innerTile; inner < innerTileEnd; inner++) {
            for (int col = columnTile; col < colTileEnd; col++) {
              DType l;
              if (transpose1)
                l = left[inner * rows + row];
              else
                l = left[row * inners + inner];

              DType r;
              if (transpose2)
                r = right[col * inners + inner];
              else
                r = right[inner * columns + col];

              if (transpose3)
                result[row * columns + col] += l * r;
              else
                result[row * columns + col] += l * r;
            }
          }
        }
      }
    }
  }
}

} // namespace variable

#endif // VARIABLE_H

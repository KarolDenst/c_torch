#include "csv_reader.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace utils {
namespace data {

CSVReader::CSVReader(std::string filename, bool has_header) {
  std::fstream f;
  f.open(filename, std::ios::in);

  std::vector<std::string> row;
  std::string line, word;

  while (f >> line) {
    row.clear();
    std::stringstream s(line);

    while (std::getline(s, word, ',')) {
      row.push_back(word);
    }

    if (has_header && headers.empty()) {
      headers = row;
      continue;
    }

    this->data.push_back(row);
  }
}

std::vector<std::string> CSVReader::pop(int index) {
  assert(index < headers.size());
  auto result = std::vector<std::string>();

  for (auto row : data) {
    result.push_back(row[index]);
    row.erase(row.begin() + index);
  }

  return result;
}

std::vector<std::string> CSVReader::pop(std::string column_name) {
  int index = get_index(column_name);
  return pop(index);
}

std::vector<std::string> CSVReader::operator[](int index) {
  assert(index < headers.size());
  auto result = std::vector<std::string>();

  for (auto row : data) {
    result.push_back(row[index]);
  }

  return result;
}

std::vector<std::string> CSVReader::operator[](std::string column_name) {
  int index = get_index(column_name);
  return data[index];
}

int CSVReader::get_index(std::string column_name) {
  auto it = std::find(headers.begin(), headers.end(), column_name);
  assert(it != headers.end());
  return it - headers.begin();
}

} // namespace data
} // namespace utils

#ifndef CSV_READER_H
#define CSV_READER_H

#include <string>
#include <vector>

namespace utils {
namespace data {

class CSVReader {
public:
  std::vector<std::string> headers;
  std::vector<std::vector<std::string>> data;

  CSVReader(std::string filename, bool has_header = true);
  std::vector<std::string> pop(int index);
  std::vector<std::string> pop(std::string column_name);
  std::vector<std::string> operator[](int index);
  std::vector<std::string> operator[](std::string column_name);

private:
  int get_index(std::string column_name);
};

} // namespace data
} // namespace utils

#endif // CSV_READER_H

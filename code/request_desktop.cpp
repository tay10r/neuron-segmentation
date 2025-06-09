#include "request.h"

#include <fstream>
#include <iostream>

namespace {

class request_impl final : public request
{
  std::string path_;

  bool failed_{ true };

  std::string body_;

public:
  explicit request_impl(std::string path)
  {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.good()) {
      return;
    }

    file.seekg(0, std::ios::end);

    const auto file_size = file.tellg();
    if (file_size < 0) {
      return;
    }

    body_.resize(static_cast<size_t>(file_size));

    file.seekg(0, std::ios::beg);

    file.read(body_.data(), body_.size());

    failed_ = false;
  }

  auto done() const -> bool override { return true; }

  auto failed() const -> bool override { return failed_; }

  [[nodiscard]] auto response_data() -> void* override { return body_.data(); }

  [[nodiscard]] auto response_size() const -> size_t override { return body_.size(); }
};

} // namespace

auto
request::get(const std::string& path) -> std::unique_ptr<request>
{
  return std::make_unique<request_impl>(path);
}

auto
request::post(const std::string& path, std::string payload) -> std::unique_ptr<request>
{
  std::cout << path << ": " << payload << std::endl;

  return std::make_unique<request_impl>("does_not_exist");
}

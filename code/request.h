#pragma once

#include <memory>

#include <stddef.h>

class request
{
public:
  static auto get(const std::string& url) -> std::unique_ptr<request>;

  static auto post(const std::string& url, std::string body) -> std::unique_ptr<request>;

  virtual ~request() = default;

  [[nodiscard]] virtual auto done() const -> bool = 0;

  [[nodiscard]] virtual auto failed() const -> bool = 0;

  [[nodiscard]] virtual auto response_data() -> void* = 0;

  [[nodiscard]] virtual auto response_size() const -> size_t = 0;
};

#include "request.h"

#include <emscripten/fetch.h>

#include <memory>
#include <vector>

#include <string.h>

namespace {

class request_impl final : public request
{
  std::string url_;
  std::vector<char> body_;
  bool done_{ false };
  bool failed_{ true };

public:
  request_impl(std::string url, std::string payload = {})
    : url_(std::move(url))
  {
    emscripten_fetch_attr_t attr;
    emscripten_fetch_attr_init(&attr);

    // Set method
    if (!payload.empty()) {
      strcpy(attr.requestMethod, "POST");
    } else {
      strcpy(attr.requestMethod, "GET");
    }

    attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
    attr.onsuccess = [](emscripten_fetch_t* fetch) {
      auto* self = static_cast<request_impl*>(fetch->userData);
      self->body_.assign(fetch->data, fetch->data + fetch->numBytes);
      self->failed_ = false;
      self->done_ = true;
      emscripten_fetch_close(fetch);
    };

    attr.onerror = [](emscripten_fetch_t* fetch) {
      auto* self = static_cast<request_impl*>(fetch->userData);
      self->failed_ = true;
      self->done_ = true;
      emscripten_fetch_close(fetch);
    };

    attr.userData = this;

    if (!payload.empty()) {
      attr.requestData = payload.data();
      attr.requestDataSize = payload.size();
    }

    emscripten_fetch(&attr, url_.c_str());
  }

  auto done() const -> bool override { return done_; }

  auto failed() const -> bool override { return failed_; }

  auto response_data() -> void* override { return body_.empty() ? nullptr : body_.data(); }

  auto response_size() const -> size_t override { return body_.size(); }
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
  return std::make_unique<request_impl>(path, std::move(payload));
}

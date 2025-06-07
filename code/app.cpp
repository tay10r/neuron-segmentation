#include <uikit/main.hpp>

namespace {

class app_impl final : public uikit::app {
 public:
  void setup(uikit::platform&) override {}

  void teardown(uikit::platform&) override {}

  void loop(uikit::platform&) override {
    if (ImGui::Begin("Gallery")) {
      ImGui::Text("It works!");
      ImGui::End();
    }
  }
};

}  // namespace

UIKIT_APP(app_impl)

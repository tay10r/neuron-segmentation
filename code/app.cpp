#include <uikit/main.hpp>

#include "request.h"

#include <GLES2/gl2.h>
#include <base64.hpp>
#include <implot.h>
#include <json.hpp>
#include <stb_image.h>
#include <stb_image_write.h>

#include <algorithm>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

class app_impl final : public uikit::app
{
  std::unique_ptr<request> image_list_request_;

  std::unique_ptr<request> image_request_;

  std::unique_ptr<request> inference_request_;

  std::vector<std::string> image_list_;

  size_t selected_image_{ static_cast<size_t>(-1) };

  stbi_uc* current_image_{ nullptr };

  stbi_uc* segmentation_image_{ nullptr };

  stbi_uc* noisy_image_{ nullptr };

  int width_{};

  int height_{};

  GLuint texture_{};

  float noise_{ 0.1F };

  std::string tmp_;

public:
  void setup(uikit::platform&) override
  {
    image_list_request_ = request::get("images/list.txt");

    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  }

  void teardown(uikit::platform&) override
  {
    clear_segmentation();

    if (current_image_) {
      stbi_image_free(current_image_);
    }

    if (noisy_image_) {
      free(noisy_image_);
    }

    glDeleteTextures(1, &texture_);
  }

  void loop(uikit::platform&) override
  {
    poll_requests();

    auto& io = ImGui::GetIO();

    ImGui::SetNextWindowSize(io.DisplaySize, ImGuiCond_Always);

    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);

    if (ImGui::Begin("##mainwindow", nullptr, ImGuiWindowFlags_NoDecoration)) {
      render_main_window();
    }
    ImGui::End();
  }

protected:
  void open_image(const std::string& filename) { image_request_ = request::get(std::string("images/") + filename); }

  void clear_segmentation()
  {
    if (segmentation_image_) {
      stbi_image_free(segmentation_image_);
      segmentation_image_ = nullptr;
    }
  }

  void issue_segmentation_request()
  {
    // Clear the current segmentation image if it exists,
    // so that it does not get inserted into the network input.
    clear_segmentation();

    // Reproduce the image without the segmentation overlay.
    process_image();

    std::vector<unsigned char> gray_buffer(width_ * height_);

    const int num_pixels{ width_ * height_ };

    for (int i = 0; i < num_pixels; i++) {
      int g = 0;
      g += noisy_image_[i * 4 + 0];
      g += noisy_image_[i * 4 + 1];
      g += noisy_image_[i * 4 + 2];
      gray_buffer[i] = g / 3;
    }

    std::string png_buffer;

    auto write_png = [](void* buf_ptr, void* data, const int size) {
      auto* buf = static_cast<std::string*>(buf_ptr);
      const auto offset = buf->size();
      buf->resize(buf->size() + size);
      for (int i = 0; i < size; i++) {
        buf->at(offset + i) = static_cast<unsigned char*>(data)[i];
      }
    };

    stbi_write_png_to_func(write_png, &png_buffer, width_, height_, 1, gray_buffer.data(), width_);

    auto payload = std::string("{ \"inputs\": [ \"" + base64::to_base64(png_buffer) + "\" ], \"params\": {} }");

    inference_request_ = request::post("/invocations", std::move(payload));
  }

  void render_main_window()
  {
    const char* selected{ (selected_image_ < image_list_.size()) ? image_list_[selected_image_].c_str()
                                                                 : "(pick one)" };

    if (ImGui::BeginCombo("Image", selected)) {
      for (size_t i = 0; i < image_list_.size(); i++) {
        if (ImGui::Selectable(image_list_[i].c_str(), i == selected_image_)) {
          selected_image_ = i;
          open_image(image_list_[i]);
        }
      }
      ImGui::EndCombo();
    }

    ImGui::BeginDisabled(!!inference_request_ || !current_image_);

    if (ImGui::Button("Segment")) {
      issue_segmentation_request();
    }

    ImGui::SameLine();

    if (ImGui::SliderFloat("Noise", &noise_, 0, 1)) {
      process_image();
    }

    ImGui::EndDisabled();

    if (!ImPlot::BeginPlot(
          "##Viewport", ImVec2(-1, -1), ImPlotFlags_Equal | ImPlotFlags_Crosshairs | ImPlotFlags_NoFrame)) {
      return;
    }

    ImPlot::SetupAxesLimits(0, 512, 0, 512, ImPlotCond_Once);

    ImPlot::PlotImage(
      "##Image", reinterpret_cast<ImTextureID>(texture_), ImPlotPoint(0, 0), ImPlotPoint(width_, height_));

    ImPlot::EndPlot();
  }

  void render_gallery()
  {
    ImGui::BeginDisabled(!!image_request_);

    for (size_t i = 0; i < image_list_.size(); i++) {
      if (ImGui::Selectable(image_list_[i].c_str(), i == selected_image_)) {
        selected_image_ = i;
        open_image(image_list_[i]);
      }
    }

    ImGui::EndDisabled();
  }

  void load_image_list(const char* data, size_t size)
  {
    std::istringstream stream(std::string(data, size));
    std::string line;
    while (stream) {
      std::getline(stream, line);
      if (!line.empty()) {
        image_list_.emplace_back(line);
      }
      line.clear();
    }
  }

  void load_image(const void* data, const size_t size)
  {
    clear_segmentation();

    if (noisy_image_) {
      free(noisy_image_);
    }

    if (current_image_) {
      stbi_image_free(current_image_);
    }

    width_ = 0;
    height_ = 0;
    current_image_ = stbi_load_from_memory(static_cast<const stbi_uc*>(data), size, &width_, &height_, nullptr, 4);
    if (!current_image_) {
      return;
    }

    noisy_image_ = static_cast<stbi_uc*>(malloc(width_ * height_ * 4));

    process_image();
  }

  void load_inference_result(const void* data, const size_t size)
  {
    clear_segmentation();

    const auto root = nlohmann::json::parse(static_cast<const char*>(data), static_cast<const char*>(data) + size);

    std::string_view prediction;

    for (const auto& pred : root.at("predictions")) {

      const auto png_buffer = base64::from_base64(pred.get<std::string_view>());

      int w{};
      int h{};

      segmentation_image_ = stbi_load_from_memory(
        reinterpret_cast<const stbi_uc*>(png_buffer.data()), png_buffer.size(), &w, &h, nullptr, 3);
      if (!segmentation_image_) {
        continue;
      }

      if ((w != width_) || (h != height_)) {
        stbi_image_free(segmentation_image_);
        segmentation_image_ = nullptr;
        continue;
      }

      break;
    }

    if (segmentation_image_) {
      process_image();
    }
  }

  void process_image()
  {
    std::minstd_rand rng(0);

    std::uniform_real_distribution<float> dist(0, 1);

    for (int i = 0; i < (width_ * height_); i++) {

      uint8_t pixel[3]{ current_image_[i * 4 + 0], current_image_[i * 4 + 1], current_image_[i * 4 + 2] };

      uint8_t out[3]{ 0, 0, 0 };

      for (int j = 0; j < 3; j++) {
        const float a = static_cast<float>(pixel[j]) * (1.0F / 255.0F);
        const float b = dist(rng);
        const float c = a + (b - a) * noise_;
        out[j] = static_cast<uint8_t>(std::clamp(static_cast<int>(c * 255), 0, 255));
      }

      if (segmentation_image_) {
        const auto* mask = &segmentation_image_[i * 3];
        for (int j = 0; j < 3; j++) {
          out[j] = static_cast<uint8_t>((static_cast<int>(mask[j]) + static_cast<int>(out[j])) >> 1);
        }
      }

      noisy_image_[i * 4 + 0] = out[0];
      noisy_image_[i * 4 + 1] = out[1];
      noisy_image_[i * 4 + 2] = out[2];
      noisy_image_[i * 4 + 3] = 255;
    }

    update_texture(noisy_image_);
  }

  void update_texture(const stbi_uc* data)
  {
    glBindTexture(GL_TEXTURE_2D, texture_);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
  }

  void poll_requests()
  {
    if (image_list_request_) {
      if (image_list_request_->done()) {
        if (!image_list_request_->failed()) {
          load_image_list(static_cast<const char*>(image_list_request_->response_data()),
                          image_list_request_->response_size());
        }
        image_list_request_.reset();
      }
    }

    if (image_request_) {
      if (image_request_->done()) {
        if (!image_request_->failed()) {
          load_image(image_request_->response_data(), image_request_->response_size());
        }
        image_request_.reset();
      }
    }

    if (inference_request_) {
      if (inference_request_->done()) {
        if (!inference_request_->failed()) {
          load_inference_result(inference_request_->response_data(), inference_request_->response_size());
        }
        inference_request_.reset();
      }
    }
  }
};

} // namespace

UIKIT_APP(app_impl)

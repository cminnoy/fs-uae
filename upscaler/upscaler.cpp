#include "upscaler.h"

#include <vector>
#include <string>
#include <array>
#include <numeric>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <iostream>
#include <thread>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <omp.h>

//#define WITH_TIMING

// --- Global State ---
namespace {

Ort::Env *g_env = nullptr;
Ort::Session *g_session = nullptr;
Ort::SessionOptions *g_session_options = nullptr;
Ort::IoBinding* g_io_binding = nullptr;

std::vector<const char*> g_input_names_ptr;
std::vector<const char*> g_output_names_ptr;
std::vector<std::string> g_input_names_str;
std::vector<std::string> g_output_names_str;
std::vector<int64_t> g_input_shape;
std::vector<int64_t> g_output_shape;

// Fixed dimensions - Match user's image buffer dimensions (Width=752, Height=576)
constexpr int kInputWidth = 752;
constexpr int kInputHeight = 576;

bool do_ai_upscale = true;

} // namespace


#define ORT_CHECK(expr) \
    try { expr; } catch (const Ort::Exception& e) { \
        fprintf(stderr, "ONNX Runtime Error: %s\n", e.what()); \
        cleanup_model(); \
        return false; \
    }

#define ORT_CHECK_VOID(expr) \
    try { expr; } catch (const Ort::Exception& e) { \
        fprintf(stderr, "ONNX Runtime Error: %s\n", e.what()); \
        /* In a void function, we cannot return false, just log and stop processing */ \
        return; \
    }

extern "C" {

void cleanup_model() {
    delete g_io_binding;
    g_io_binding = nullptr;

    delete g_session;
    g_session = nullptr;

    delete g_session_options;
    g_session_options = nullptr;

    delete g_env;
    g_env = nullptr;

    g_input_names_ptr.clear();
    g_input_names_str.clear();
    g_output_names_ptr.clear();
    g_output_names_str.clear();
    g_input_shape.clear();
    g_output_shape.clear();
}

bool load_model(const char* model_path) {
    if (g_session) {
        fprintf(stderr, "Model already loaded.\n");
        return true;
    }

    ORT_CHECK(g_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "UpscalerEnv"));
    ORT_CHECK(g_session_options = new Ort::SessionOptions());
    g_session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Enable profiling: Specify a prefix for the output file
    //g_session_options->EnableProfiling("onnxruntime_profile");

#if 0
    // OLD ROCm EP usage - kept for reference
    OrtROCMProviderOptions rocm_options{};
    rocm_options.device_id = 0; // Use device 0
    g_session_options->AppendExecutionProvider_ROCM(rocm_options);
    printf("ONNX Runtime session options configured with ROCm EP (device 0).\n");
#else
    // New MiGraphX EP usage
    OrtMIGraphXProviderOptions migraphx_options{};
    migraphx_options.device_id = 0; // Use device 0
    g_session_options->AppendExecutionProvider_MIGraphX(migraphx_options);
    printf("ONNX Runtime session options configured with MIGraphX EP (device 0).\n");
#endif

    ORT_CHECK(g_session = new Ort::Session(*g_env, model_path, *g_session_options)); 

    // Create the IoBinding object
    ORT_CHECK(g_io_binding = new Ort::IoBinding(*g_session));

    size_t num_inputs = g_session->GetInputCount();
    size_t num_outputs = g_session->GetOutputCount();

    if (num_inputs != 1 || num_outputs != 1) {
        fprintf(stderr, "Model must have 1 input and 1 output.\n");
        cleanup_model();
        return false;
    }

    Ort::AllocatorWithDefaultOptions allocator;

    // Input
    Ort::AllocatedStringPtr input_name = g_session->GetInputNameAllocated(0, allocator);
    g_input_names_str.emplace_back(input_name.get());
    g_input_names_ptr.push_back(g_input_names_str[0].c_str());

    Ort::TypeInfo in_type_info = g_session->GetInputTypeInfo(0);
    auto in_tensor_info = in_type_info.GetTensorTypeAndShapeInfo();
    if (in_tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
        fprintf(stderr, "Input tensor must be UINT8.\n");
        cleanup_model();
        return false;
    }

    g_input_shape = in_tensor_info.GetShape();
    // Expect NWHC shape [1, 752, 576, 4] matching image dimensions
    if (g_input_shape.size() != 4 || g_input_shape[0] != 1 || g_input_shape[1] != kInputHeight || g_input_shape[2] != kInputWidth || g_input_shape[3] != 4) {
        fprintf(stderr, "Model input shape {%lld, %lld, %lld, %lld} does not match expected {%d, %d, %d, %d}.\n",
                (long long)g_input_shape[0], (long long)g_input_shape[1], (long long)g_input_shape[2], (long long)g_input_shape[3],
                1, kInputHeight, kInputWidth, 4);
        cleanup_model();
        return false;
    }

    // Output
    Ort::AllocatedStringPtr output_name = g_session->GetOutputNameAllocated(0, allocator);
    g_output_names_str.emplace_back(output_name.get());
    g_output_names_ptr.push_back(g_output_names_str[0].c_str());

    Ort::TypeInfo out_type_info = g_session->GetOutputTypeInfo(0);
    auto out_tensor_info = out_type_info.GetTensorTypeAndShapeInfo();
    // Expect output tensor to be UINT8
    if (out_tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
        fprintf(stderr, "Output tensor must be UINT8.\n");
        cleanup_model();
        return false;
    }

    g_output_shape = out_tensor_info.GetShape();
    // Expect NWHC shape [1, 752, 576, 4] matching image dimensions
    if (g_output_shape.size() != 4 || g_output_shape[0] != 1 || g_output_shape[1] != kInputHeight || g_output_shape[2] != kInputWidth || g_output_shape[3] != 4) {
        fprintf(stderr, "Model output shape {%lld, %lld, %lld, %lld} does not match expected {%d, %d, %d, %d}.\n",
                (long long)g_output_shape[0], (long long)g_output_shape[1], (long long)g_output_shape[2], (long long)g_output_shape[3],
                1, kInputHeight, kInputWidth, 4);
        cleanup_model();
        return false;
    }

    fprintf(stdout, "Model loaded: input shape = {");
    for (size_t i = 0; i < g_input_shape.size(); ++i)
        fprintf(stdout, "%lld%s", (long long)g_input_shape[i], i + 1 < g_input_shape.size() ? ", " : "}\n");

    fprintf(stdout, "Model loaded: output shape = {");
    for (size_t i = 0; i < g_output_shape.size(); ++i)
        fprintf(stdout, "%lld%s", (long long)g_output_shape[i], i + 1 < g_output_shape.size() ? ", " : "}\n");

    return true;
}

void run_inference(const uint8_t* rgba, uint8_t* output_rgba, int limit_x, int limit_y, int limit_w, int limit_h) {
#ifdef WITH_TIMING
    auto start_inference = std::chrono::high_resolution_clock::now();
#endif
    bool do_inference = do_ai_upscale;

    // Add checks for rgba and output_rgba validity if necessary
    if (!g_session || !g_io_binding || !rgba || !output_rgba) {
        fprintf(stderr, "Model not loaded, IoBinding not created, or input/output buffers are null.\n");
        do_inference = false;
    }

    if (!do_inference) {
        // Fallback: copy original RGBA data (Keep this fallback)
        if (rgba && output_rgba) {
            int start_y = std::max(0, limit_y);
            int end_y = std::min((int)kInputHeight, limit_y + limit_h);
            int start_x = std::max(0, limit_x);
            int end_x = std::min((int)kInputWidth, limit_x + limit_w);

            if (start_x < end_x && start_y < end_y) {
                 size_t row_bytes = (size_t)(end_x - start_x) * 4;
                 for (int y = start_y; y < end_y; ++y) {
                     const uint8_t* src_row = rgba + (size_t)(y * kInputWidth + start_x) * 4;
                     uint8_t* dst_row = output_rgba + (size_t)(y * kInputWidth + start_x) * 4;
                     std::memcpy(dst_row, src_row, row_bytes);
                 }
            }
        }
        return;
    }

    // Clear previous bindings (Line 243)
    ORT_CHECK_VOID(g_io_binding->ClearBoundInputs());
    ORT_CHECK_VOID(g_io_binding->ClearBoundOutputs());
    
    // Memory allocator on CPU (assuming rgba and output_rgba are on CPU)
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Shape array for ONNX tensor in WHC-like format: {1, Width, Height, Channels}
    // This reflects the direct RGBA chunky format from the emulator.
    constexpr std::array<int64_t, 4> shape = {1, kInputHeight, kInputWidth, 4};
    size_t data_size_bytes = (size_t)kInputHeight * kInputWidth * 4; // 4 bytes per pixel

    // Create input tensor directly from the rgba buffer using the HWC-like shape
    Ort::Value input_tensor = Ort::Value::CreateTensor<uint8_t>(mem_info, const_cast<uint8_t*>(rgba), data_size_bytes, shape.data(), shape.size());

    // Create output tensor directly using the output_rgba buffer using the HWC-like shape
    Ort::Value output_tensor = Ort::Value::CreateTensor<uint8_t>(mem_info, output_rgba, data_size_bytes, shape.data(), shape.size());

    // Bind the input and output tensors
    ORT_CHECK_VOID(g_io_binding->BindInput(g_input_names_ptr[0], input_tensor));
    ORT_CHECK_VOID(g_io_binding->BindOutput(g_output_names_ptr[0], output_tensor));

    // --- End of Direct Binding Setup ---

    // ONNX Runtime will use the buffers bound in g_io_binding
    try {
        g_session->Run(Ort::RunOptions{nullptr}, *g_io_binding);

    } catch (const Ort::Exception& e) {
#ifdef WITH_TIMING
        auto stop_inference_error = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_inference_error = stop_inference_error - start_inference;
        fprintf(stderr, "ONNX Runtime Inference Error: %s (Inference took %.2f ms)\n", e.what(), duration_inference_error.count());
#endif
        // Fallback: copy original RGBA data using limits (Keep this fallback)
        if (rgba && output_rgba) {
            int start_y = std::max(0, limit_y);
            int end_y = std::min((int)kInputHeight, limit_y + limit_h);
            int start_x = std::max(0, limit_x);
            int end_x = std::min((int)kInputWidth, limit_x + limit_w);
            if (start_x < end_x && start_y < end_y) {
                    size_t row_bytes = (size_t)(end_x - start_x) * 4;
                    for (int y = start_y; y < end_y; ++y) {
                        const uint8_t* src_row = rgba + (size_t)(y * kInputWidth + start_x) * 4;
                        uint8_t* dst_row = output_rgba + (size_t)(y * kInputWidth + start_x) * 4;
                        std::memcpy(dst_row, src_row, row_bytes);
                    }
            }
        }
        return;
    }
#ifdef WITH_TIMING
    auto stop_inference = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_inference = stop_inference - start_inference;
    fprintf(stderr, "Inference time: %.2f ms\n", duration_inference.count());
#endif
}

void allow_ai_upscale(bool flag) {
    do_ai_upscale = flag;
}

bool ai_upscale_on() {
    return do_ai_upscale;
}

} // extern "C"
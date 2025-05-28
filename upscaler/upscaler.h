#include <stdint.h>

#include "fs/ml/opengl.h"

#ifdef __cplusplus
extern "C" {
#endif

void print_thread_id();
bool load_model(const char* model_path);
void run_inference(const uint8_t* rgba_framebuffer, uint8_t* enhanced_rgba_framebuffer, int limit_x, int limit_y, int limit_w, int limit_h);
void cleanup_model();
void allow_ai_upscale(bool flag);
bool ai_upscale_on();

#ifdef __cplusplus
}
#endif


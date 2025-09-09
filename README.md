# INIVerse-MIX_Pony Hyperrealistic Generator

https://api.runpod.io/badge/runpod-workers/worker-a1111

Modified to perfection by [Ruminansi_Art](https://github.com/ruminansiart-arch)

Runs [Automatic1111 Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) with the optimized **INIVerse-MIX_Pony** workflow for generating hyperrealistic portraits and landscapes.

Comes pre-packaged with:
- **INIVerse-MIX_Pony** model
- **Pony VAE**
- **Detail Tweaker XL LoRA**
- **4x-UltraSharp** upscaler
- **ADetailer** for face & hand fixing

---

## 🚀 Available Modes

The `input` object requires a `mode` parameter. Choose one of the following:

### 1. Portrait Mode
**Mode:** `{"mode": "portrait"}`

Generates a **512x768** hyperrealistic portrait using:
- Euler sampler, 30 steps, CFG 7
- Permanent high-quality prompts (invisible)
- Pony VAE & Detail Tweaker XL LoRA
- ADetailer for face and hands

```json
{
   "input": {
     "mode": "portrait",
     "prompt": "a photograph of a woman with freckles and green eyes",
     "seed": -1
   }
}
```

### 2. Landscape Mode
**Mode:** `{"mode": "landscape"}`

Generates a **768x512** hyperrealistic landscape using the same high-fidelity settings as Portrait mode.

```json
{
   "input": {
     "mode": "landscape",
     "prompt": "a serene mountain lake at sunset, hyperrealistic",
     "seed": -1
   }
}
```

### 3. Refiner Mode (Standalone)
**Mode:** `{"mode": "refiner"}`

Takes ANY base64-encoded image and refines it to 2K resolution:
1.  Upscales 4x using **4x-UltraSharp**.
2.  Refines with **Img2Img** (50 steps, denoising strength 0.35, CFG 5).
3.  Applies **ADetailer (face & hands)** for final polish.

```json
{
   "input": {
     "mode": "refiner",
     "image": "base64_encoded_image_string_here",
     "prompt": "enhance details, make hyperrealistic" // Optional
   }
}
```

> 💡 **Important**: Negative prompts and quality-enforcing positive prompts are permanently embedded and cannot be overridden. The user's prompt is appended to the high-quality base prompt.

import time
import runpod
import requests
import base64
from PIL import Image
from io import BytesIO
from requests.adapters import HTTPAdapter, Retry

LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"

automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))

def wait_for_service(url):
    retries = 0
    max_retries = 30  # 1 minute max wait
    while retries < max_retries:
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                print("WebUI service is ready!")
                return
        except requests.exceptions.RequestException:
            pass
        except Exception as err:
            print(f"Error checking service: {err}")
        
        retries += 1
        if retries % 5 == 0:
            print(f"Service not ready yet ({retries}/{max_retries}). Retrying...")
        time.sleep(2)
    
    print("Service failed to start within timeout period")
    # Continue anyway - the handler might still work

def call_api(endpoint, payload):
    try:
        response = automatic_session.post(url=f'{LOCAL_URL}/{endpoint}', json=payload, timeout=600)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"API call failed: {e}")
        return {"error": str(e)}

def get_image_size(base64_str):
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        return image.size
    except Exception as e:
        print(f"Error getting image dimensions: {e}")
        return (512, 512)

# ===== PERMANENT PROMPTS =====
PERMANENT_POSITIVE = """(score_9, score_8_up, score_7_up), subsurface scattering, soft natural lighting, rim light, hyperrealistic skin details, natural anatomy, <lora:add-detail-xl:1>"""

ADETAILER_FACE_PROMPT = "perfect human face, symmetrical features, natural skin texture, skin pores, subtle skin imperfections, defined cheekbones, balanced jawline, realistic eyes, detailed eyes, eye catchlights, fine eyelashes, natural eyebrows, detailed nose structure, soft lips, lip moisture, healthy skin tone, subsurface scattering"
ADETAILER_HAND_PROMPT = "perfect human hands, natural hands, delicate hands, realistic fingers, perfect fingers, anatomical hands, detailed knuckles, subtle skin wrinkles"

PERMANENT_NEGATIVE = """(worst quality, low quality, normal quality:1.4), lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry, artist name, trademark, logo, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, blurry eyes, disfigured, extra limbs, gross proportions, malformed limbs, missing arms, missing legs, fused fingers, too many fingers, long neck, bad feet, poorly drawn feet, bad toes, unnatural pose, asymmetrical eyes, cross-eyed, unnatural body proportions, disconnected limbs, cloned face, doll-like, plastic, mannequin, airbrushed, (3D render, cartoon, anime, sketch, drawing, illustration, painting, digital art:1.3), duplicate, morbid, mutilated"""

def handler(event):
    job_input = event["input"]
    
    # MODE 3: STANDALONE REFINER
    if job_input.get("mode") == "refiner":
        init_image = job_input["image"]
        original_width, original_height = get_image_size(init_image)
        target_width = original_width * 4
        target_height = original_height * 4

        # Stage 1: Upscale using 4x-UltraSharp
        extras_payload = {
            "image": init_image,
            "upscaling_resize": 4,
            "upscaler_1": "4x-UltraSharp"
        }
        extras_result = call_api('extra-single-image', extras_payload)
        if "error" in extras_result:
            return extras_result
        upscaled_image = extras_result['image']

        # Stage 2: Img2Img Refine
        user_prompt = job_input.get("prompt", "")
        full_prompt = f"{PERMANENT_POSITIVE}, {user_prompt}" if user_prompt else PERMANENT_POSITIVE

        i2i_payload = {
            "init_images": [upscaled_image],
            "prompt": full_prompt,
            "negative_prompt": PERMANENT_NEGATIVE,
            "width": target_width,
            "height": target_height,
            "cfg_scale": 5,
            "steps": 50,
            "denoising_strength": 0.35,
            "sampler_name": "Euler",
            "alwayson_scripts": {
                "adetailer": {
                    "args": [
                        {
                            "ad_model": "face_yolov8n.pt",
                            "ad_confidence": 0.3,
                            "ad_prompt": ADETAILER_FACE_PROMPT
                        },
                        {
                            "ad_model": "hand_yolov8n.pt",
                            "ad_confidence": 0.3,
                            "ad_prompt": ADETAILER_HAND_PROMPT
                        }
                    ]
                }
            }
        }
        return call_api('img2img', i2i_payload)

    # MODE 1 & 2: PORTRAIT or LANDSCAPE
    mode = job_input.get("mode")
    if mode not in ["portrait", "landscape"]:
        return {"error": "Invalid mode. Use 'portrait', 'landscape', or 'refiner'."}

    # Set dimensions based on mode
    width, height = (512, 768) if mode == "portrait" else (768, 512)

    # Build Text2Image payload
    user_prompt = job_input.get("prompt", "")
    full_prompt = f"{PERMANENT_POSITIVE}, {user_prompt}"

    t2i_payload = {
        "prompt": full_prompt,
        "negative_prompt": PERMANENT_NEGATIVE,
        "width": width,
        "height": height,
        "cfg_scale": 7,
        "steps": 30,
        "seed": job_input.get("seed", -1),
        "sampler_name": "Euler",
        "alwayson_scripts": {
            "adetailer": {
                "args": [
                    {
                        "ad_model": "face_yolov8n.pt",
                        "ad_confidence": 0.3,
                        "ad_prompt": ADETAILER_FACE_PROMPT
                    },
                    {
                        "ad_model": "hand_yolov8n.pt",
                        "ad_confidence": 0.3,
                        "ad_prompt": ADETAILER_HAND_PROMPT
                    }
                ]
            }
        }
    }

    # Execute Text2Image
    return call_api('txt2img', t2i_payload)

if __name__ == "__main__":
    wait_for_service(url=f'{LOCAL_URL}/sd-models')
    print("WebUI API Service is ready. Starting RunPod Serverless...")
    runpod.serverless.start({"handler": handler})

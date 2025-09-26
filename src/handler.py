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
PERMANENT_POSITIVE = """(score_9, score_8_up, score_7_up), subsurface scattering, soft natural lighting, rim light, hyperrealistic skin details, natural anatomy, subtle skin wrinkles"""

ADETAILER_FACE_PROMPT = "symmetrical features, natural skin pores, detailed eyes, big cute eyes, fine eyelashes, detailed nose structure, soft lips, lip moisture, healthy skin tone"

PERMANENT_NEGATIVE = """(worst quality, low quality, normal quality:1.4), destroyed nails, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry, artist name, trademark, logo, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, blurry eyes, disfigured, extra limbs, gross proportions, malformed limbs, missing arms, missing legs, fused fingers, too many fingers, long neck, bad feet, poorly drawn feet, bad toes, unnatural pose, asymmetrical eyes, cross-eyed, unnatural body proportions, disconnected limbs, cloned face, doll-like, plastic, mannequin, airbrushed, duplicate, morbid, mutilated"""

def handler(event):
    job_input = event["input"]
    mode = job_input.get("mode")
    
    if mode not in ["portrait", "landscape"]:
        return {"error": "Invalid mode. Use 'portrait' or 'landscape' only."}

    # Set base and target dimensions
    if mode == "portrait":
        base_w, base_h = 512, 768
        target_w, target_h = 1024, 1536
    else:  # landscape
        base_w, base_h = 768, 512
        target_w, target_h = 1536, 1024

    user_prompt = job_input.get("prompt", "").strip()
    lora_weight = 3.0
    full_prompt = f"{PERMANENT_POSITIVE}, {user_prompt}, <lora:add-detail-xl:{lora_weight}>" if user_prompt else f"{PERMANENT_POSITIVE}, <lora:add-detail-xl:{lora_weight}>"
    
    adetailer_face_prompt = f"{PERMANENT_POSITIVE}, {user_prompt}, {ADETAILER_FACE_PROMPT}, <lora:add-detail-xl:2.5>" if user_prompt else f"{PERMANENT_POSITIVE}, {ADETAILER_FACE_PROMPT}, <lora:add-detail-xl:2.5>"

    t2i_payload = {
        "prompt": full_prompt,
        "negative_prompt": PERMANENT_NEGATIVE,
        "width": base_w,
        "height": base_h,
        "cfg_scale": 9,
        "steps": 50,
        "seed": job_input.get("seed", -1),
        "sampler_name": "DPM++ 2M Karras",
        "enable_hr": True,
        # ✅ Use hr_resize instead of hr_scale
        "hr_resize_x": target_w,
        "hr_resize_y": target_h,
        "hr_upscaler": "4x-UltraSharp",
        "hr_second_pass_steps": 20,
        "denoising_strength": 0.35,
        "alwayson_scripts": {
            "adetailer": {
                "args": [
                    {
                        "ad_model": "face_yolov8n.pt",
                        "ad_confidence": 0.3,
                        "ad_prompt": adetailer_face_prompt
                    }
                ]
            }
        }
    }

    return call_api('txt2img', t2i_payload)

if __name__ == "__main__":
    wait_for_service(url=f'{LOCAL_URL}/sd-models')
    print("WebUI API Service is ready. Starting RunPod Serverless...")
    runpod.serverless.start({"handler": handler})

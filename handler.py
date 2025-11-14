"""
RunPod Serverless Handler for Qwen-Image
"""
import runpod
from diffusers import (
    DiffusionPipeline,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
)
import torch
from PIL import Image
import base64
import io
import os
from typing import Optional# Global model instance (loaded once on cold start)
pipeline = None

# Available schedulers/samplers
SCHEDULERS = {
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "dpm": DPMSolverMultistepScheduler,
    "ddim": DDIMScheduler,
    "pndm": PNDMScheduler,
    "lms": LMSDiscreteScheduler,
    "kdpm2": KDPM2DiscreteScheduler,
    "kdpm2_a": KDPM2AncestralDiscreteScheduler,
}

def load_model():
    """Load model once during cold start"""
    global pipeline
    if pipeline is not None:
        return pipeline

    print("🚀 Loading Qwen-Image model...")

    model_name = "Qwen/Qwen-Image"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipeline = pipeline.to(device)

    # Load LoRA weights if available
    lora_path = "/workspace/Qwen4Play_v2.safetensors"
    if os.path.exists(lora_path):
        print("🎨 Loading Qwen4Play_v2 LoRA weights...")
        pipeline.load_lora_weights(".", weight_name="Qwen4Play_v2.safetensors")
        print("✅ LoRA weights loaded successfully")
    else:
        print("⚠️ Qwen4Play_v2.safetensors not found, proceeding without LoRA")

    print(f"✅ Model loaded on {device}")
    print(f"📊 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    return pipeline

def generate_image(job):
    """
    RunPod handler function - mirrors generate_image() from runpod_startup.sh lines 88-112
    Input format: {"input": {"prompt": "...", "width": 1024, ...}}
    Output format: {"image": "base64...", "seed": 123}
    """
    job_input = job["input"]

    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "prompt is required"}

    negative_prompt = job_input.get("negative_prompt", " ")
    width = job_input.get("width", 1024)
    height = job_input.get("height", 1024)
    num_inference_steps = job_input.get("num_inference_steps", 50)
    true_cfg_scale = job_input.get("true_cfg_scale", 4.0)
    seed = job_input.get("seed", None)
    scheduler_name = job_input.get("scheduler", None)  # Optional scheduler/sampler selection
    lora_scale = job_input.get("lora_scale", 1.0)  # LoRA weight scale (0.0 to 1.0)

    print(f"🎨 Generating: {prompt[:100]}...")

    # Load model if not already loaded
    pipe = load_model()

    # Set scheduler if specified
    if scheduler_name and scheduler_name.lower() in SCHEDULERS:
        print(f"🔧 Using scheduler: {scheduler_name}")
        scheduler_class = SCHEDULERS[scheduler_name.lower()]
        pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
    elif scheduler_name:
        print(f"⚠️ Unknown scheduler '{scheduler_name}', using default")

    # Setup generator for seed
    generator = None
    if seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)

    # Generate image
    with torch.inference_mode():
        # Prepare cross_attention_kwargs for LoRA scale
        cross_attention_kwargs = {"scale": lora_scale} if lora_scale != 1.0 else None
        
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=generator,
            cross_attention_kwargs=cross_attention_kwargs
        )

    # Convert to base64
    image = result.images[0]
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    used_seed = seed if seed is not None else (generator.initial_seed() if generator else 0)

    print(f"✅ Generated successfully! Seed: {used_seed}")

    return {
        "image": img_b64,
        "seed": used_seed
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": generate_image})

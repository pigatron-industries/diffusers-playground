import os, time, torch

print("HF_HOME =", os.environ.get("HF_HOME"))
t0 = time.time()

from diffusers import Flux2KleinPipeline

print("Loading Flux2KleinPipeline ...")
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9b-kv",
    torch_dtype=torch.bfloat16,
)
pipe.to("mps")
pipe.text_encoder.to("mps")
print(f"Loaded in {time.time()-t0:.1f}s")

prompt = "A photograph of a red cat sitting on a windowsill, soft morning light"
t1 = time.time()
out = pipe(
    prompt=prompt,
    height=512,
    width=512,
    num_inference_steps=4,
    guidance_scale=3.5,
    generator=torch.Generator(device="mps").manual_seed(0),
    return_dict=True,
)
img = out.images[0]
print(f"Generated {img.size} in {time.time()-t1:.1f}s")
outp = "outputs/flux2_klein_test.png"
os.makedirs("outputs", exist_ok=True)
img.save(os.path.join(os.getcwd(), outp))
print("Saved to", outp)

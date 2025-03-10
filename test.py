from transformers import pipeline

ckpt = "google/siglip2-so400m-patch14-384"
pipe = pipeline(model=ckpt, task="zero-shot-image-classification")

inputs = {
    "images": [
        "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg", # bear
        "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000776.jpg", # teddy bear
    ],
    "texts": [
        "bear looking into the camera",
        "bear looking away from the camera",
        "a bunch of teddy bears",
        "two teddy bears",
        "three teddy bears"
    ],
}

outputs = pipe(inputs["images"], candidate_labels=inputs["texts"])
print(outputs)
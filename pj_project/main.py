import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

from transformers import pipeline

model_id = "llava-hf/llava-1.5-7b-hf"

SYS = """
You are a Dark Pattern Website Checker. Given a screenshot of a website, you have to identify whether the website shows any Dark Patterns. You have to summarize briefly for the following points:
    Types of deceptive pattern
Tricks used in websites and apps that make you do things that you didn't mean to. Also known as 'dark patterns'.
Comparison prevention
The user struggles to compare products because features and prices are combined in a complex manner, or because essential information is hard to find.
Confirmshaming
The user is emotionally manipulated into doing something that they would not otherwise have done.
Disguised ads
The user mistakenly believes they are clicking on an interface element or native content, but it's actually a disguised advertisment.
Fake scarcity
The user is pressured into completing an action because they are presented with a fake indication of limited supply or popularity.
Fake social proof
The user is misled into believing a product is more popular or credible than it really is, because they were shown fake reviews, testimonials, or activity messages.
Fake urgency
The user is pressured into completing an action because they are presented with a fake time limitation.
Forced action
The user wants to do something, but they are required to do something else undesirable in return.
Hard to cancel
The user finds it easy to sign up or subscribe, but when they want to cancel they find it very hard.
Hidden Costs
The user is enticed with a low advertised price. After investing time and effort, they discover unexpected fees and charges when they reach the checkout.
Hidden subscription
The user is unknowingly enrolled in a recurring subscription or payment plan without clear disclosure or their explicit consent.
Nagging
The user tries to do something, but they are persistently interrupted by requests to do something else that may not be in their best interests.
Obstruction
The user is faced with barriers or hurdles, making it hard for them to complete their task or access information.
Preselection
The user is presented with a default option that has already been selected for them, in order to influence their decision-making.
Sneaking
The user is drawn into a transaction on false pretences, because pertinent information is hidden or delayed from being presented to them.
Trick wording
The user is misled into taking an action, due to the presentation of confusing or misleading language.
Visual interference
The user expects to see information presented in a clear and predictable way on the page, but it is hidden, obscured or disguised.
If the website looks legit then feel free to write "Safe" for the corresponding points.
"""
prompt = f"USER: <image>\n+{SYS}\nASSISTANT:"

pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

def image_to_text(image, prompt=prompt):
    #image is a PIL image
    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
    return outputs[0]['generated_text']

# fastapi app that uses the image_to_text function by taking an image and a prompt as input and returning the generated text
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/image_to_text")
async def image_to_text_endpoint(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    text = image_to_text(image)
    return JSONResponse(content={"text": text})

# example usage by javascript
# const formData = new FormData();
# formData.append("file", file);
# fetch("http://.../image_to_text", {
#     method: "POST",
#     body: formData
# }).then(response => response.json()).then(data => console.log(data.text));
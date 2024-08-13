from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import cv2
import torch
from torchvision import transforms
from dataset import preprocess_image
import uvicorn
from REC.REC import BeautyREC

app = FastAPI()

params = {
    'dim':48,
    'style_dim':48,
    'activ': 'relu',
    'n_downsample':2,
    'n_res':3,
    'pad_type':'reflect'
}

# Load your pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BeautyREC(params).to(device)
model.load("./checkpoints/BeautyREC.pt", device)
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@app.post("/apply-makeup/")
async def apply_makeup(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    makeup_image = Image.open('/Users/mihir/Desktop/Projects/CosVTO/CosVTO/data/makeup/1.jpg')  # Load your makeup image

    # Apply model
    with torch.no_grad():
        makeup_seg, nonmakeup_seg, makeup_img, nonmakeup_img = preprocess_image(makeup_image, image, device)
        makeup_img = makeup_img.unsqueeze(0).to(device)
        nonmakeup_img = nonmakeup_img.unsqueeze(0).to(device)
        makeup_seg = makeup_seg.unsqueeze(0).to(device)
        nonmakeup_seg = nonmakeup_seg.unsqueeze(0).to(device)
        output_tensor = model(nonmakeup_img, makeup_img, makeup_seg, nonmakeup_seg)
    
    # Convert tensor to image
    output_image = output_tensor[0].detach().cpu().numpy().transpose([1,2,0])/2+0.5
    output_image = (output_image.copy()*255).astype(np.uint8)
    output_image = output_image[:,:,::-1]
    cv2.imwrite('output.jpg', output_image)
    
    # Send the processed image back
    output_pil = Image.fromarray(output_image)
    output_buffer = io.BytesIO()
    output_pil.save(output_buffer, format="PNG")
    return JSONResponse(content={"image": output_buffer.getvalue().decode("latin1")})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

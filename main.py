
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from routers import CropPlanner
import io

app = FastAPI()
#app.mount("/media", StaticFiles(directory="media"), name="media")

app.include_router(CropPlanner.router)

model = load_model('tomato_disease_classifier.h5') 
labelInfo = {0:'Healthy', 1:'Bacterial Spot', 2:'Early Blight', 3:'Late Blight', 4:'Leaf Mold', 5:'Septoria leaf spot', 6:'Spider mites', 7:'Target Spot', 8:'Tomato mosaic virus', 9:'Tomato Yellow Leaf Curl Virus'}

img_height, img_width = 224, 224  

@app.post("/imgPrediction")
async def imgPrediction(file: UploadFile = File(...)):
    fileObj = await file.read()
    testimage = Image.open(io.BytesIO(fileObj)).convert("RGB")

    img = testimage.resize((img_height, img_width))
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)

    predictions = model.predict(x)
    predictedLabel = labelInfo[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
              
    return {
        "predictedLabel": predictedLabel,
        "confidence": confidence
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

import io
from PIL import Image
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

tflite_model_path = "./assets/garbage_model.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class PredictionResult(BaseModel):
    class_name: str
    probability: float

def preprocess_image(image: Image.Image) -> np.ndarray:
    # Preprocess the input image according to your model requirements
    image = image.resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        preprocessed_image = preprocess_image(image)

        interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data)
        print("prediction", prediction)
        class_names = ["cardboard","glass", "metal", "paper", "plastic", "trash"]
        predicted_class = class_names[prediction]
        probability = output_data[0][prediction]

        return {"class_name": predicted_class, "probability": float(probability)}

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


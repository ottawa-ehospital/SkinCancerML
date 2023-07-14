# 1. Library imports
import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
from fastapi import FastAPI, File, UploadFile,Form
from pydantic import BaseModel
from keras.models import load_model
from keras.utils import load_img
import cv2
from fastapi.middleware.cors import CORSMiddleware


# 2. Create the app object
app = FastAPI()
pickle_in = open("classifier.pkl","rb")
#classifier=pickle.load(pickle_in)
classifier=load_model('my_model.hdf5')

origins = ["http://localhost", "http://localhost:8080", "https://myapp.herokuapp.com","http://www.e-hospital.ca/","http://www.e-hospital.ca","http://localhost:5000"]

app.add_middleware(
    CORSMIDDLEware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"]
    allow_headers=["*"],
)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Skin Cancer Detector'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def hello_name(name: str):
    return {'message': f'Our project {name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_skinCancer(file: UploadFile = File(...)):
   
    image = load_img(file.filename)
    image = np.asarray(image)
    img = image
    img_resize = (cv2.resize(img, dsize=(244, 244),    interpolation=cv2.INTER_CUBIC))/255.
        
    img_reshape = img_resize[np.newaxis,...]
 
    
    prediction = classifier.predict(img_reshape)
    if(prediction[0]>0):
        prediction="Cancer"
        print(prediction)
    else:
        prediction="Not Cancer"
        print(prediction)
    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload
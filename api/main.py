from fastapi import FastAPI,UploadFile,HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel,Field
from model.linear_regression import Linear_Regression
import pandas as pd
from io import BytesIO
from sklearn.preprocessing import StandardScaler
import imageio


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "for documentation go to /doc site"}


@app.post("/linear_regression")
async def linear_regression(dataset:UploadFile):
    if dataset.filename.endswith(".csv"):
        data = await dataset.read()
        data = BytesIO(data)
        dataFrame = pd.read_csv(data)
    
    elif dataset.filename.endswith(".xls"):
        data = await dataset.read()
        data = BytesIO(data)
        dataFrame = pd.read_excel(data)
    else:
        raise HTTPException(status_code=400,detail="file format not supported")
    
    columns = list(dataFrame.columns.values)
    X = dataFrame.drop(columns=[columns[-1]])
    Y = dataFrame[columns[-1]]

    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X) 

    model = Linear_Regression()
    model.fit(X,Y.values)

    image_files = model.get_image_list()


    durations = [1 - 0.004*i for i in range(len(image_files))]
    durations = [max(d, 0.05) for d in durations]

    visualization = BytesIO()


    imageio.mimsave(visualization, image_files, format='GIF',duration=durations)
    visualization.seek(0)
    
    return StreamingResponse(visualization, media_type="image/gif")
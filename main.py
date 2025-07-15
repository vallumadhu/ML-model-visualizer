from fastapi import FastAPI,UploadFile,HTTPException,Query
from fastapi.responses import StreamingResponse,RedirectResponse
from pydantic import BaseModel,Field
import pandas as pd
from io import BytesIO
from sklearn.preprocessing import StandardScaler
import imageio

from model.linear_regression import Linear_Regression
from model.polynomial_regression import Polynomial_Regression

app = FastAPI()

async def convert_to_df(dataset):
    dataFrame = None
    if dataset.filename.endswith(".csv"):
        data = await dataset.read()
        data = BytesIO(data)
        dataFrame = pd.read_csv(data)
    
    elif dataset.filename.endswith(".xls"):
        data = await dataset.read()
        data = BytesIO(data)
        dataFrame = pd.read_excel(data)
    return dataFrame

def df_to_X_Y(dataFrame):
    columns = list(dataFrame.columns.values)

    X = dataFrame.drop(columns=[columns[-1]])
    Y = dataFrame[columns[-1]]

    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X) 

    return (X,Y)

def image_list_to_gif(image_files):

    durations = [1 - 0.004*i for i in range(len(image_files))]
    durations = [max(d, 0.05) for d in durations]

    visualization = BytesIO()


    imageio.mimsave(visualization, image_files, format='GIF',duration=durations)
    visualization.seek(0)

    return visualization


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.post("/linear_regression")
async def linear_regression(dataset:UploadFile):

    dataFrame = await convert_to_df(dataset)
    if dataFrame is None or dataFrame.empty:
        raise HTTPException(status_code=400,detail="file format not supported") 

    X,Y = df_to_X_Y(dataFrame)      

    model = Linear_Regression()
    model.fit(X,Y.values)
    
    image_files = model.get_image_list()
    visualization = image_list_to_gif(image_files)
    
    return StreamingResponse(visualization, media_type="image/gif")




@app.post("/polynomial_regression")
async def polynomial_regression(dataset:UploadFile,degree:int=Query(default=2,gt=1)):
    dataFrame = await convert_to_df(dataset)
    if dataFrame is None or dataFrame.empty:
        raise HTTPException(status_code=400,detail="file format not supported") 

    X,Y = df_to_X_Y(dataFrame)    

    model = Polynomial_Regression(degree)
    model.fit(X,Y.values)

    image_files = model.get_image_list()
    visualization = image_list_to_gif(image_files)
    
    return StreamingResponse(visualization, media_type="image/gif")
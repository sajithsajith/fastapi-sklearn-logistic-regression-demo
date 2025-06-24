from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ML.preprocessing import load_and_clean_data, split_data
from ML.train import train_model, predict_accuracy
from ML.visualization import visualize_data, display_decision_boundary

app = FastAPI(title="ML Model Runner")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def create_upload_file(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a CSV file."
        )

    try:
        df = load_and_clean_data(file.file)

        viz_vars = [
            "CPT Code",
            "Insurance Company",
            "Physician Name",
            "Payment Amount",
            "Balance",
        ]
        viz_hue = "Denial Reason"
        viz_uri = visualize_data(df, viz_vars, viz_hue)

        feature_columns = ["CPT Code", "Balance"]
        target_column = "Denial Reason"

        X_train, X_test, y_train, y_test = split_data(
            df, feature_columns, target_column, 0.3, 123
        )

        model = train_model(X_train, y_train)
        accuracy = predict_accuracy(model, X_test, y_test)

        boundary_uri = display_decision_boundary(model, X_test, y_test, feature_columns)

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "accuracy": accuracy,
                "viz_uri": viz_uri,
                "boundary_uri": boundary_uri,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred during processing: {e}"
        )

import io
import base64
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
import seaborn as sns


def visualize_data(data_frame, vars, hue):
    g = sns.PairGrid(
        data=data_frame,
        vars=vars,
        hue=hue,
    )
    g.map(sns.scatterplot)
    g.add_legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    image_uri = f"data:image/png;base64,{image_base64}"
    return image_uri


def display_decision_boundary(model, X_test, y_test, features):
    disp = DecisionBoundaryDisplay.from_estimator(
        model,
        X_test,
        response_method="predict",
        alpha=0.5,
    )
    disp.ax_.scatter(
        X_test[features[0]],
        X_test[features[1]],
        c=y_test,
        edgecolor="k",
    )
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    image_uri = f"data:image/png;base64,{image_base64}"
    return image_uri

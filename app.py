import gradio as gr
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()

model = LogisticRegression(max_iter=200)
model.fit(iris.data, iris.target)


def predict(sepal_length, sepal_width, petal_length, petal_width):

    data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(data)[0]

    return iris.target_names[prediction]


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Sepal Length"),
        gr.Number(label="Sepal Width"),
        gr.Number(label="Petal Length"),
        gr.Number(label="Petal Width")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Iris Flower Machine Learning Model"
)

demo.launch()
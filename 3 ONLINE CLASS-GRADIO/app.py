import gradio as gr
import joblib
 
 
# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------
 
model = joblib.load("student_model.pkl")
 
 
# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------
 
def predict_marks(
    study_hours,
    attendance,
    previous_score,
    assignments
):
 
    prediction = model.predict([
        [
            study_hours,
            attendance,
            previous_score,
            assignments
        ]
    ])
 
    predicted_marks = prediction[0]
 
    return f"Predicted Marks: {predicted_marks:.2f}"
 
 
# --------------------------------------------------
# GRADIO INTERFACE
# --------------------------------------------------
 
with gr.Blocks(
    title="Student Performance Predictor"
) as app:
 
    gr.Markdown(
        """
        # Student Performance Prediction
 
        Enter the student's information below.
        The trained machine learning model will
        predict the expected marks.
        """
    )
 
    with gr.Row():
 
        study_hours = gr.Number(
            label="Study Hours",
            value=5
        )
 
        attendance = gr.Number(
            label="Attendance (%)",
            value=75
        )
 
    with gr.Row():
 
        previous_score = gr.Number(
            label="Previous Score",
            value=65
        )
 
        assignments = gr.Number(
            label="Assignments Completed",
            value=7
        )
 
    predict_button = gr.Button(
        "Predict Marks",
        variant="primary"
    )
 
    result = gr.Textbox(
        label="Prediction"
    )
 
    predict_button.click(
        fn=predict_marks,
        inputs=[
            study_hours,
            attendance,
            previous_score,
            assignments
        ],
        outputs=result
    )
 
 
# --------------------------------------------------
# START APPLICATION
# --------------------------------------------------
 
if __name__ == "__main__":
    app.launch()

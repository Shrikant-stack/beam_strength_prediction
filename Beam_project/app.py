from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import joblib
import pandas as pd

app = Flask(__name__)


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
db = SQLAlchemy(app)

model = joblib.load("model.pkl")

class BeamData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    Beam_Length_m = db.Column(db.Float, nullable=False)
    Thickness_m = db.Column(db.Float, nullable=False)
    Cross_Section_Type = db.Column(db.String(50), nullable=False)
    Factor_of_Safety = db.Column(db.Float, nullable=False)
    Material = db.Column(db.String(50), nullable=False)
    Predicted_Load = db.Column(db.Float, nullable=False)

with app.app_context():
    db.create_all()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Collect data from form
        beam_length = float(request.form["Beam_Length_m"])
        thickness = float(request.form["Thickness_m"])
        cross_section = request.form["Cross_Section_Type"]
        fos = float(request.form["Factor_of_Safety"])
        material = request.form["Material"]

        # Convert input into DataFrame for prediction
        input_data = pd.DataFrame([{
            "Beam_Length_m": beam_length,
            "Thickness_m": thickness,
            "Cross_Section_Type": cross_section,
            "Factor_of_Safety": fos,
            "Material": material
        }])

        # One-hot encode like training
        input_data = pd.get_dummies(input_data)
        # Align with model's training columns
        # (Fill missing with 0)
        trained_columns = model.feature_names_in_
        input_data = input_data.reindex(columns=trained_columns, fill_value=0)

        # Prediction
        prediction = model.predict(input_data)[0]

        # Save to DB
        new_entry = BeamData(
            Beam_Length_m=beam_length,
            Thickness_m=thickness,
            Cross_Section_Type=cross_section,
            Factor_of_Safety=fos,
            Material=material,
            Predicted_Load=prediction
        )
        db.session.add(new_entry)
        db.session.commit()

    return render_template("form.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

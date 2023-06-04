from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)


app=application



@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            ApplicantIncome=float(request.form.get('ApplicantIncome')),
            CoapplicantIncome = float(request.form.get('CoapplicantIncome')),
            LoanAmount = float(request.form.get('LoanAmount')),
            Loan_Amount_Term = float(request.form.get('Loan_Amount_Term')),
            Credit_History = float(request.form.get('Credit_History')),
            Gender = request.form.get('Gender'),
            Married= request.form.get('Married'),
            Dependents = request.form.get('Dependents'),
            Education= request.form.get('Education'),
            Self_Employed = request.form.get('Self_Employed'),
            Property_Area = request.form.get('Property_Area'),
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('result.html',final_result=results)






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
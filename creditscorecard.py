import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
from datetime import datetime, timedelta
# from dateutil.relativedelta import relativedelta

app = Flask(__name__)

etc_model = joblib.load(r'C:\Users\nisrin.dhoondia\PycharmProjects\DefaultPredictionML\ExtraTreesClassifier_model')
minmax_scaler = joblib.load(r'C:\Users\nisrin.dhoondia\PycharmProjects\DefaultPredictionML\minmax_scaler')

@app.route('/CreditScoreCard')
def creditscorecard():
    return render_template('index.html')

@app.route('/CreditScoreCard', methods=['POST'])
def predict():
    tdic = {}
    format_str = '%Y-%m-%d'
    todays_date = datetime.today().date()
    # # test_data = json.loads(request.data)
    # # testdata = np.array(list(map(np.float, test_data['values']))).reshape(1,-1)
    tdic['DAYS_BIRTH'] = [float((datetime.strptime(datetime.strptime(request.form['days_birth'].strip(), "%d/%m/%Y").strftime("%Y-%m-%d"), "%Y-%m-%d").date() - todays_date).days)]
    tdic['DAYS_EMPLOYED'] = [float((datetime.strptime(datetime.strptime(request.form['days_employed'].strip(), "%d/%m/%Y").strftime("%Y-%m-%d"), "%Y-%m-%d").date() - todays_date).days)]
    # tdic['DAYS_EMPLOYED'] = [float(request.form['days_employed'].strip())]
    tdic['DAYS_CREDIT_ENDDATE'] = [float((datetime.strptime(datetime.strptime(request.form['days_credit_enddate'].strip(), "%d/%m/%Y").strftime("%Y-%m-%d"), "%Y-%m-%d").date() - todays_date).days)]
    # tdic['DAYS_CREDIT_ENDDATE'] = [float(request.form['days_credit_enddate'].strip())]
    tdic['AMT_CREDIT_SUM'] = [float(request.form['amt_credit_sum'].strip())]
    tdic['Active'] = [float(request.form['active_contract_status'].strip())]
    tdic['Completed']= [float(request.form['completed_contract_status'].strip())]
    tdic['REGION_POPULATION_RELATIVE'] = [float(request.form['region_population_relative'].strip())]
    tdic['AMT_GOODS_PRICE'] = [float(request.form['amt_goods_price'].strip())]
    tdic['AMT_INCOME_TOTAL'] = [float(request.form['amt_income_total'].strip())]
    tdic['CNT_CHILDREN'] = [float(request.form['cnt_children'].strip())]
    if request.form['flag_own_realty'].strip().upper() == 'Y':
        tdic['FLAG_OWN_REALTY'] = [float(1)]
    else:
        tdic['FLAG_OWN_REALTY'] = [float(0)]
    testdata = pd.DataFrame.from_dict(tdic)
    testdata = testdata[['DAYS_BIRTH',
                         'DAYS_EMPLOYED',
                         'DAYS_CREDIT_ENDDATE',
                         'AMT_CREDIT_SUM',
                         'Active',
                         'Completed',
                         'REGION_POPULATION_RELATIVE',
                         'AMT_GOODS_PRICE',
                         'AMT_INCOME_TOTAL',
                         'CNT_CHILDREN',
                         'FLAG_OWN_REALTY']]
    col_name = testdata.columns.to_list()
    testdata =  minmax_scaler.transform(testdata)
    testdata = pd.DataFrame(testdata, columns = col_name)
    pred_actuals = etc_model.predict(testdata)
    pred_prob = etc_model.predict_proba(testdata)
    output_str = 'Your Credit Score is: ' + str(pred_actuals[0]) + '<br/>' + 'And the probability of your Credit Score is: ' + str(pred_prob[0][0])
    # return str(tdic['DAYS_BIRTH'])
    # return str(todays_date)
    return output_str
    # return render_template('simple.html',  tables=[testdata.to_html(classes='data')], titles=testdata.columns.values)


if __name__ == '__main__':
    app.run(debug = True)
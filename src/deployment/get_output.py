def get_output():   
    from preprocessor import preprocessor
    from datetime import datetime
    date = datetime.today().strftime("%B %d, %Y")
    try:
        data = getdata()
        data = preprocessor(data)
        y = get_prediction(data)
    except:
        y = 1

    # convert the result from 0, 1, 2 to an output
    result = 'a rise (bull 🦏)' if y ==2 else \
            'a fall (bear 🐻)' if y == 0 else \
            'steady' # always defaults to 1
    
    output = f"Today is {date}. The predicted 24hr price movement is {result}."
    
    return output

def getdata():
    """
    when called this retrieves information from the website
    """
    import pandas as pd
    ## api call
    ## demo csv
    ## or 
    # read the demo data
    data = pd.read_csv('demo.csv', parse_dates=['Date']).set_index('Date')
    return data


def get_prediction(data):
    """receives preprocessed data
    returns y, predicted label
    """
    import pandas as pd
    import numpy as np
    from tensorflow import keras
    model = keras.models.load_model('../output/predictor/model')
    scaler = pickle.load (open('../output/predictor/scaler', 'rb')) # standardscaler
    X = scaler.transform(data)
    try:
        y = model.predict(X)
        y = np.squeeze(y).argmax(axis=1)
    except:
        return -1
    return y   
    


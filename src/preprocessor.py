# Code to preprocess the input data to the model

def preprocessor (tmp):
    import pandas as pd
    import numpy as np

    """
    Receives a frame of at least 5 records and preprocesses it
    the records must be in order from the oldest record at index 0
    to the most recent record at index -1
    """
    def make_windows(tmp, window_size=3, horizon=1):
        """
        Input: 1-D time series Dataframe
        output: X 2D time series of window_size
        y: 1D time series lagged by horizon
        """
        data = tmp.copy()

        windows = list(data.rolling(window=window_size))[window_size:]
        window_data = pd.concat([pd.DataFrame(window.T.values) for window in windows])
        window_data.index = data.index[window_size:]
        return window_data

    df = tmp.copy()
    # checking compatibility
    required_columns = {'Close', 'SPX', 'Volume'}
    if len(df) >= 5 and required_columns.issubset(set(df.columns)):
        pass
    else:
        return ("Invalid input")

    # transformations
    df['Close'] = df['Close'].diff()/df['Close']
    spx = df['SPX'].diff()/df['SPX']
    volume = np.sqrt(np.sqrt(np.log(np.log(np.log(df['Volume'])))))
    spx = spx.diff()
    close = df['Close'].diff()
    def finstat_cut(value, bins = [-0.029,  0.024]):
            if value < bins[0]:
                bin = 0
            elif value > bins[1]:
                bin = 2
            else:
                bin = 1
            return bin
    bins = [-0.029,  0.024]
    move = df['Close'].apply(lambda x: finstat_cut(x, bins))

    # windowing
    
    nflx_windows = make_windows(close.to_frame())
    spx_windows = make_windows(spx.to_frame())
    

    # concatenating the Series
    data = pd.DataFrame([volume,spx,close]).T # merge the normalized features
    data['Move'] = move

    data = data.merge(nflx_windows, right_index=True, left_index=True) # merge the windows
    data = data.merge(spx_windows, right_index=True, left_index=True, suffixes=('', '_SPX')) # spx window
    data.drop(columns=['SPX', 'Close'], inplace=True) # they have been made redundant by the windowed parameters

    return data.iloc[1]

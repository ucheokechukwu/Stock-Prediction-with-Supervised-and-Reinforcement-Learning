from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np

def ADF_Stationarity_Test (timeseries, significance_level=0.05):
    """
    function to analyze and report the stationarity of a time series
    """
    result = adfuller(timeseries)
    print('ADF Statistic: %f' % result[0])
    print('p value: %f' % result[1])
    print('Critical value: ')
    for key, value in result[4].items():
        print('t%s: %.3f' % (key, value))

    if result[1]<significance_level:
        print ('Null hypothesis that the series is not stationary can be REJECTED.')
    else:
        print('The null hypothesis that the series is NOT stationary cannot be rejected.\n')
    return result[1]
        
        

def stationary_test(df, col):
    """
    function to plot visualization and call ADFuller
    to check stationarity of timeseries
    """
    import matplotlib.pyplot as plt
    df= df[col].copy().dropna()
    plt.plot(df)
    plt.title(f'No transformation of {col}')
    plt.show()
    
    
    p = ADF_Stationarity_Test(df.values)
    
    if p < 0.05:
        return

    diff = df.diff().dropna().values

    plt.plot(diff)
    plt.title(f'Diff transformation of {col}')
    plt.show()
    
    p = ADF_Stationarity_Test(diff)
    
    if p < 0.05:
        return
        
    logdf = np.log(df.values)
    plt.plot(logdf)
    plt.title(f'Log transformation of {col}')
    plt.show()
    
    p = ADF_Stationarity_Test(logdf)
    
    if p < 0.05:
        return
        
    sqdf = np.sqrt(df.values)
    plt.plot(sqdf)
    plt.title(f'Square root transformation of {col}')
    plt.show()
    
    p = ADF_Stationarity_Test(sqdf)
    
    
    
def plot_dual_line_graph(col1, col2):
    """
    plots 2 time series into one line graph, does auto-scaling
    """
    import numpy as np
    import matplotlib.pyplot as plt
    # col1 = col1.copy().to_frame()
    # col2 = col2.copy().to_frame()

    fig, ax1 = plt.subplots(figsize=(10,10))

    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel(col1.name, color=color)
    ax1.plot(col1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(col2.name, color=color)  # we already handled the x-label with ax1
    ax2.plot(col2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    
    
    
def OLS_results():
    """
    runs the statsmodel OLS tests and displays results
    """
    import statsmodels.api as sm
    import matplotlib.pyplot as plt

    from sklearn.preprocessing import MinMaxScaler
    scaler_x = MinMaxScaler()
    X = df.drop(columns='Close')
    y = df['Close']
    X_scaled = scaler_x.fit_transform(X)
    X_scaled = pd.DataFrame(X, columns=df.drop(columns='Close').columns)

    # scaler_y = MinMaxScaler()
    # y_scaled = scaler_y.fit_transform(df['Close'].values.reshape(-1,1))


    model = sm.OLS(y, X_scaled)
    results = model.fit()

    fig, ax = plt.subplots()
    fig = sm.graphics.plot_fit(results, 0, ax=ax)
    ax.set_ylabel("Close")
    ax.set_title("Linear Regression")

    print(results.summary())
    
    
def hypertune_bins(col, bins=[-0.020,  0.017]):
    """
    function to hypertune the discretization bins.
    receives bins, a list of 2 numbers that will act as the "cut-offs"
    and return the value counts and the one-Way F test results
    """
    def finstat_cut(value, bins=[-0.020,  0.017]):

        if value < bins[0]:
            bin = 0
        elif value > bins[1]:
            bin = 2
        else:
            bin = 1
        return bin

    move = col.apply(lambda x: finstat_cut(x, bins))
    plt.figure(figsize=(2,2))
    plt.scatter(col, move)
    display(move.value_counts())

    return move


# bins = [-0.020,  0.017]
# df['Move'] = hypertune_bins(bins)
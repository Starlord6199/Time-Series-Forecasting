from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
def parser(x):
	return datetime.strptime(x, '%H:%M')

dataset = read_csv('hourly_water_consumption.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
X = dataset.values
size = int(len(X) * 0.65)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,1))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
pyplot.hist(X)
pyplot.show()
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
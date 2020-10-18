#https://matplotlib.org/index.html
#http://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.RegressionResults.html

#Current Working Directory; http://www.pythonforbeginners.com/os/pythons-os-module
#import os.path
#os.getcwd()
#Change the current working directory to path.
#os.chdir("C:/path/to/location")
#os.chdir("C:/Users/Stein_Wang/Documents/Local_Personal/Cornell/NBA6450 Advanced Investment Strategies/Python/python_scripts")



#==============================================
#Sandbox test export - still needs loop - see 

#https://stackoverflow.com/questions/39068124/how-to-append-a-dataframe-to-existing-excel-sheet
#http://xlsxwriter.readthedocs.io/working_with_pandas.html
#http://xlsxwriter.readthedocs.io/working_with_memory.html



import os.path
os.chdir("C:/Users/Stein_Wang/Documents/Local_Personal/Cornell/NBA6450 Advanced Investment Strategies/Python/python_scripts")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy import stats
def cls():print('\n'*5)

#import data
data_file = pd.read_excel('test1.xlsx')
print(data_file)
x = data_file['NI']
y = data_file['PS']

#plotting parameters
#slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
#plt.plot(x, y, 'o', label='original data')
#plt.plot(x, intercept + slope*x, 'r', label='fitted line')
#plt.legend()
#plt.show()

#OLS regression
x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
t_stat = results.params/results.bse

print(results.summary())
print(results.conf_int(alpha=0.05, cols=None))
#print('R2: ', results.rsquared, 'Coeff/Betas: ', results.params, 'Standard Errors:', results.bse, 't-stat:', t_stat, 'p-value:', results.pvalues)


#look into numpy array


#output data
#http://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.RegressionResults.html
output = {"R2": results.rsquared, "Betas": results.params, "Std_Errors": results.bse, "t_stats": t_stat, "p_values": results.pvalues, "observations": results.nobs, "f_stat": results.fvalue, "significance_f": results.f_pvalue} 
df = pd.DataFrame(output)
writer = pd.ExcelWriter('z_ouput1.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='py_export1')
writer.save()






#==============================================
#donor_code #https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.stats.linregress.html#scipy.stats.linregress
>>> import matplotlib.pyplot as plt
>>> from scipy import stats
>>> np.random.seed(12345678)
>>> x = np.random.random(10)
>>> y = np.random.random(10)
>>> slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
#To get coefficient of determination (r_squared)

>>> print("r-squared:", r_value**2)
('r-squared:', 0.080402268539028335)
#Plot the data along with the fitted line

>>> plt.plot(x, y, 'o', label='original data')
>>> plt.plot(x, intercept + slope*x, 'r', label='fitted line')
>>> plt.legend()
>>> plt.show()




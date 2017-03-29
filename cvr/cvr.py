import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


##read data

cvr = pd.read_csv('conversion_data.csv')

##descriptive analysis

cvr = cvr[cvr.age < 111]

cvr_by_country = cvr.groupby(['country'])['converted'].mean()
cvr_by_country = cvr_by_country.reset_index()
##print(cvr_by_country)


cvr_by_page = cvr.groupby(['total_pages_visited'])['converted'].mean()
cvr_by_page = cvr_by_page.reset_index()
##print(cvr_by_page)
sns.set_style('darkgrid')
sns.regplot(x = 'total_pages_visited', y = 'converted', data = cvr_by_page, ci = None, logistic = True)
##sns.barplot(x = 'country', y = 'converted', data = cvr_by_country)
##plt.show()



cvr_country_dummy = pd.get_dummies(cvr['country'])
cvr_source_dummy = pd.get_dummies(cvr['source'])
cvr = cvr.join(cvr_country_dummy)
cvr = cvr.join(cvr_source_dummy)


print(cvr.head())

train, test = train_test_split(cvr, test_size = 0.2)


cols = ['age', 'new_user','total_pages_visited'] + list(cvr_country_dummy.columns.values) \
       + list(cvr_source_dummy.columns.values)

x_train = train[list(cols)].values
y_train = train['converted'].values

x_test = test[list(cols)].values
y_test = test['converted'].values

##training the model

clf = RandomForestClassifier(n_estimators = 10, max_depth=None,min_samples_split=2, random_state=0)
et_scores = cross_val_score(clf, x_train, y_train).mean()
print(et_scores)

clf.fit(x_train, y_train)
clf.predict(x_test)

##clf.fit(x_train, y_train)















# coding: utf-8

# In[38]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import linear_model
from wordcloud import WordCloud
get_ipython().magic(u'matplotlib inline')

import glob


# # Read Files

# In[39]:

# list all files in the directory
files = glob.glob('nyt-comments/*')
files


# ### Read comment files

# In[40]:

# list all comment files
files_comment = glob.glob('nyt-comments/NYTComments*')
files_comment


# In[41]:

# import all comment files as a data frame
dat_comment = pd.concat([pd.read_csv(f, low_memory=False) for f in files_comment], ignore_index=True)


# In[42]:

# drop duplicates
dat_comment = dat_comment.drop_duplicates(subset = 'commentID')


# In[43]:

dat_comment.shape


# In[44]:

dat_comment.head()


# In[45]:

dat_comment.dtypes


# ### Read article files

# In[46]:

# list all article files
files_article = glob.glob('nyt-comments/NYTArticles*')
files_article


# In[47]:

# import all article files as a data frame
dat_article = pd.concat([pd.read_csv(f, low_memory=False) for f in files_article], ignore_index=True)


# In[48]:

# drop duplicates
dat_article = dat_article.drop_duplicates(subset = 'articleID')


# In[49]:

dat_article.shape


# In[50]:

dat_article.head()


# In[51]:

dat_article.dtypes


# # Objectives
# 1. What kinds of articles are likely to have more comments?
# 2. What kinds of comments are likely to have more recommendations?

# # Exploratory Data Analysis and Data Manipulation

# ### Outcome and predictors of objective 1

# In[52]:

# compute number of comments for each article
comments = dat_comment.groupby('articleID').size()
comments = pd.DataFrame({'articleID': comments.index, 'comments': comments.values})


# In[53]:

dat_article2 = pd.merge(left=dat_article, right=comments, on='articleID', validate='1:1')


# In[54]:

dat_article2.drop(['abstract', 'articleID', 'byline', 'headline', 'keywords', 'snippet', 'webURL'],
                  axis = 1, inplace = True) # we'll deal with keywords and/or headline later
dat_article2 = dat_article2.astype(dtype = {'documentType': 'category',
                                            'newDesk': 'category',
                                            'sectionName': 'category',
                                            'source': 'category',
                                            'typeOfMaterial': 'category'
                                         })
dat_article2['pubDate'] = pd.to_datetime(dat_article2['pubDate'],format='%Y-%m-%d %H:%M:%S')
dat_article2.dtypes


# In[55]:

dat_article2.describe(include='category')


# In[56]:

dat_article2.describe(include=None)


# In[57]:

pd.plotting.scatter_matrix(dat_article2[['articleWordCount', 'multimedia', 'printPage', 'comments']], figsize = (16, 16),
                           hist_kwds={'bins': 100}, grid = True)


# In[58]:

dat_article2.plot(x='pubDate', y='comments', kind='line', figsize = (16, 8),                   title='Comments vs pubDate (log scale, partial data)', logy = True, style = '.',                   xlim=(pd.to_datetime('2017-01-01'), pd.to_datetime('2017-03-1')))


# In[59]:

# log transform on comments
dat_article2['log_comments'] = np.log(dat_article2['comments'])
dat_article2.drop('comments', axis = 1, inplace = True)


# In[60]:

dat_article2['pubDate_weekday'] = dat_article2['pubDate'].dt.weekday.astype(dtype = 'category')
dat_article2['pubDate_hour'] = dat_article2['pubDate'].dt.hour.astype(dtype = 'category')
dat_article2.drop('pubDate', axis = 1, inplace = True)


# In[61]:

dat_article2.boxplot(column='log_comments', by='pubDate_weekday', figsize = (16, 8))


# In[62]:

dat_article2.boxplot(column='log_comments', by='pubDate_hour', figsize = (16, 8))


# In[63]:

dat_article2.dtypes


# In[64]:

dat_article2_dummies = pd.get_dummies(data=dat_article2,
                                      prefix = {'documentType':'documentType',
                                                'newDesk':'newDesk',
                                                'sectionName':'sectionName',
                                                'source':'source',
                                                'typeOfMaterial':'typeOfMaterial',
                                                'pubDate_weekday':'pubDate_weekday',
                                                'pubDate_hour':'pubDate_hour'
                                               })
dat_article2_dummies.head()


# In[65]:

X_article2 = dat_article2_dummies.drop('log_comments', axis = 1)
Y_article2 = dat_article2_dummies['log_comments']


# ### Outcome and predictors of objective 2

# # Modeling

# ### Objective 1:  LASSO (with CV) on log_comments

# In[66]:

mod1 = linear_model.LassoCV(alphas = np.logspace(-5, -2, num=100), cv = 10, n_jobs = -1, selection = 'random', max_iter=1e5)
mod1.fit(X_article2, Y_article2)


# In[67]:

alpha = mod1.alphas_
mse_mean = np.mean(mod1.mse_path_, axis = 1)
plt.figure(figsize=(16, 8))
plt.semilogx(alpha, mse_mean)
plt.ylim(1.6, 1.8)
plt.vlines(x=mod1.alpha_, ymin = 0, ymax = 5)


# In[68]:

alpha


# In[69]:

mse_mean


# In[70]:

mod1.alpha_


# In[71]:

plt.figure(figsize=(8, 24))
plt.plot(mod1.coef_, X_article2.columns.values, '.')


# In[81]:

result1 = dict(zip(X_article2.columns.values, mod1.coef_))
result1neg = dict(zip(X_article2.columns.values, - mod1.coef_))


# In[87]:

wordcloud1 = WordCloud(colormap='YlOrBr', background_color = 'black')
wordcloud1.generate_from_frequencies(frequencies=result1)

wordcloud2 = WordCloud(colormap='GnBu', background_color = 'black')
wordcloud2.generate_from_frequencies(frequencies=result1neg)


# In[88]:

plt.figure(figsize=(16, 10))
plt.imshow(wordcloud1, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Effects")


# In[89]:

plt.figure(figsize=(16, 10))
plt.imshow(wordcloud2, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Effects")


# In[ ]:




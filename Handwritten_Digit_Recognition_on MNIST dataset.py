#!/usr/bin/env python
# coding: utf-8

# ## Fetching datasets

# In[ ]:


from sklearn.datasets import fetch_openml


# In[20]:


mnist=fetch_openml('mnist_784')


#  ## x e mnist data thakbe ar y e target

# In[73]:


x,y=mnist['data'],mnist['target']


# In[74]:


x


# ## etake 28X28 aray te nite hobe eta 1Darray 2D te nite hobe

# In[75]:


x.shape 


# ## 7000 labels ache

# In[76]:


y.shape 


# ## plot korbo

# In[77]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[78]:


import matplotlib
import matplotlib.pyplot as plt


# ## take anydigit and reshape in nextline

# In[79]:


some_digit=x[3601]
some_digit_image=some_digit.reshape(28,28) # lets reshape to plot it


# In[80]:


plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")


# In[81]:


y[3601]


# ## we have put some training data aside but in mnist dataset it's already splitted

# In[82]:


x_train=x[:6000]


# In[83]:


x_test=x[6000:]


# In[84]:


y_train, y_test=y[:6000], y[6000:]


# In[85]:


import numpy as np # shuffling
shuffle_index=np.random.permutation(6000)
x_train,y_train=x_train[shuffle_index],y_train[shuffle_index]


# ## Creating 2 detector

# In[86]:


y_train=y_train.astype(np.int8)
y_test=y_test.astype(np.int8)
y_train_2= (y_train==2) # train a binary classifier
y_test_2=(y_test==2)


# In[87]:


y_train_2


# In[88]:


from sklearn.linear_model import LogisticRegression
clf= LogisticRegression(tol=0.1)


# ## 2detector banbo tai train_2

# In[89]:


clf.fit(x_train,y_train_2) 


# In[90]:


clf.predict([some_digit])


# ## Cross validation

# In[92]:


from sklearn.model_selection import cross_val_score
a=cross_val_score(clf,x_train,y_train_2, cv=3, scoring="accuracy")


# In[93]:


a.mean()


# In[ ]:





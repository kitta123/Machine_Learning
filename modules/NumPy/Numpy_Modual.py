#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


a1=np.array([1,2,3,4,5])
a2=np.array([6,7,8,9,10])
c=a1+a2
print(c)


# In[5]:


l1=[1,2,3,4,5]
l2=[6,7,8,9,10]
c=l1+l2
print(c)


# In[9]:


arr2d=np.array([[1,2,3],[4,5,6]])
print(arr2d)


# In[10]:


help(np.array)


# In[13]:


type(l1)


# In[15]:


al1=np.array(l1)
print(al1)


# In[16]:


help(np.ones)


# In[17]:


f1=np.ones(4)
print(f1)


# In[18]:


f1=np.ones(4,dtype=int)
print(f1)


# In[21]:


f2=np.ones((5,3),dtype=int)
print(f2)


# In[23]:


z1=np.zeros(4)
print(z1)


# In[25]:


z1=np.zeros(3,dtype=int)
print(z1)


# In[26]:


r1=np.random.random(4)
print(r1)


# In[27]:


r2=np.linspace(3,10,2)
print(r2)


# In[29]:


r2=np.linspace(5,10,4)
print(r2)


# In[31]:


z2=np.arange(5,100,5)
print(z2)


# In[32]:


z2.shape


# In[33]:


f2.shape


# In[34]:


f2.ndim


# In[35]:


f2.itemsize


# In[36]:


f2.dtype  


# In[37]:


e1=np.arange(2,32,2)
print(e1)


# In[43]:


e1[[0,1,2]]


# In[46]:


e1[4]


# In[47]:


e1[5:10]


# In[52]:


rsq=e1**2
for sq in rsq:
    print(sq)


# In[60]:


for i in e1:
    if(i%4==0)
    print(i)


# In[75]:


r=f2.reshape(3,5)
print(r)


# In[82]:


a1=np.array([[1:6],[3:8]])
print(a1)


# In[100]:


a1=np.arange(1,9)
a2=np.arange(9,17)
print(a1)
print(a2)


# In[101]:


h1=np.hstack((a1,a2))
v1=np.vstack((a1,a2))
print(h1,v1)


# In[ ]:





# In[87]:


help(np.hstack)


# In[ ]:





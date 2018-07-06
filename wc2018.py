
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


fi="tabula-qcuxk3y7c1ezwo5yylnn.csv"
df = pd.read_csv(fi)


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df[20:30]


# In[6]:


df.dtypes


# In[7]:


# remove header row (is there because frame is conctenation of each team's table)
df = df[df.Team != 'Team']
df[10:30]


# In[8]:


df.describe()


# In[9]:


# the tallest player :
df['Height'].max()


# In[10]:


# make object-> numeric so can run aggregations on top of it
df.apply(pd.to_numeric, errors='ignore')


# In[11]:


df.dtypes


# In[12]:


df['Height'] = df['Height'].astype('int64')


# In[13]:



df = df.reset_index(drop=True)


# In[14]:


df.head()


# In[15]:


# find index of the tallest player
ind = df['Height'].idxmax()
ind


# In[16]:


pd.__version__


# In[17]:


# and the tallest player is:
df.iloc[149]


# In[18]:


df[149:150]


# In[19]:


df.dtypes


# In[20]:


# and the shortest player is ...
df.iloc[df['Height'].idxmin()]


# In[21]:


# average hight by team
df.groupby('Team', as_index=False)['Height'].mean().sort_values('Height')


# In[22]:


# shortest teams are Peru, Saudi Arabia, Argentina
# tallest Serbia, Denmark, Germany


# In[23]:


# same but add a column for median and the variance (standard deviation)
df.groupby('Team', as_index=False).agg({'Height':['mean','median','std']}).sort_values([("Height","mean")])


# In[24]:


import datetime


# In[25]:


def func(x):
    return datetime.datetime.strptime(x, '%d.%m.%Y').date()

df['birth date corrected']=df['Birth Date'].apply(lambda x: func(x))


# In[26]:


df.head()


# In[27]:


from datetime import date
def calculate_age(born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

df['age']=df['birth date corrected'].apply(lambda x: calculate_age(x))


# In[28]:


df.head()


# In[29]:


def calc_age_as_fraction(birth_date):
    days_in_year = 365.2425  
    return (date.today() - birth_date).days / days_in_year
df['age_fraction']=df['birth date corrected'].apply(lambda x: calc_age_as_fraction(x))


# In[30]:


df.head()


# In[31]:


# average age by team with median and variance (standard deviation)
df_age =df.groupby('Team', as_index=False).agg({'age_fraction':['mean','median','std']}).sort_values([("age_fraction","mean")])


# In[32]:


df_age


# In[33]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[34]:


df_age.unstack(['age_fraction']).head()


# In[35]:


df['Team'] = df['Team'].astype('|S')


# In[36]:


ax = df_age['age_fraction']['mean'].plot(x=df_age['Team'],kind="bar",color=['blue'])
# ax.set_xticklabels(df_age['Team'], rotation=90)


# In[37]:


# ax = df_age['age_fraction']['mean'].plot(x=df_age['Team'],kind="bar",color=['blue'])
# ax.set_xticklabels(df_age['Team'], rotation=90)
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)
width=0.75
# ax.bar(list(df_age['Team']),sorted(list(df_age['age_fraction']['mean'].sort_values())), width , color='b', yerr=list(df_age['age_fraction']['std']))
ax = df_age['age_fraction']['mean'].plot(x=df_age['Team'],kind="bar",color=['blue'],yerr=df_age['age_fraction']['std'])

ax.set_xticklabels(list(df_age['Team']), rotation=90)
ax.set_title("National team age")
plt.show()


# In[38]:


df_height =df.groupby('Team', as_index=False).agg({'Height':['mean','median','std']}).sort_values([("Height","mean")])
df_height


# In[39]:


fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)
width=0.75
ax = df_height['Height']['mean'].plot(x=df_height['Team'],kind="bar",color=['blue'],yerr=df_height['Height']['std'])
ax.set_xticklabels(list(df_height['Team']), rotation=90)
ax.set_title("National team height")
plt.show()


# In[40]:


#calc bmi - body mass index
# bmi = mass/height^2
ctn=0
def calc_bmi(x):
#     global ctn
    
#     ctn+=1
#     bmi=int(x['Weight'])/int(x['Height'])**2
#     print(ctn,bmi)
    mass = x['Weight']
    height = x['Height']
    bmi = int(mass) / int((height/100)**2)
#     print(ctn,mass,height,bmi)

    return bmi

df['BMI']=df.apply(lambda x: calc_bmi(x),axis=1)
# d.apply(lambda row: min([row['A'], row['B']])-row['C'], axis=1)


# In[41]:


df


# In[42]:


df.describe()


# In[43]:


df[df['age']==45.0]


# In[44]:


df[df['age']==19.0]


# In[45]:


df[df['BMI']==20.0]


# In[46]:


# club origin
def get_club_origin(x):
    return str(x['Club']).split("(")[1].split(")")[0]
 
#test
x0=dict()
x0["Club"] = "Manchester United FC (ENG)"
print(get_club_origin(x0))


# In[47]:


df['Club origin']=df.apply(lambda x:get_club_origin(x),axis=1)


# In[48]:


df.head()


# In[49]:


# get the 3letter code of team
import pycountry
print(pycountry.countries.get(name="England"))


# In[50]:


# remove b from team
df['Team'] = df['Team'].str.decode('utf-8') 


# In[51]:


def get_3_letter_code(x):
    # special cases not handled by pycountry (why not??)
    if (str(x['Team'])=='England'):
        return "ENG"
    if (str(x['Team'])=="IR Iran"):
        return "IRN"
    if (str(x['Team'])=="Korea Republic"):
        return "KOR"
    if (str(x['Team'])=="Russia"):
        return "RUS"
    return pycountry.countries.get(name=str(x["Team"])).alpha_3

df['Country_3']=df.apply(lambda x:get_3_letter_code(x),axis=1)


# In[52]:


for p in pycountry.countries:
    print(p)


# In[53]:


df[df['Team']=="Russia"]


# In[54]:


def plays_in_home_country(x):
    return (x['Club origin']==x['Country_3'])
df['Plays_in_home_country']=df.apply(lambda x:plays_in_home_country(x),axis=1)


# In[55]:


df_same_country =df[df['Plays_in_home_country']==True].groupby('Team',as_index=False).agg({'Plays_in_home_country':'count'})
# df.groupby('Team', as_index=False).agg({'Height':['mean','median','std']}).sort_values([("Height","mean")])


# In[56]:


df_same_country


# In[57]:


df[df['Plays_in_home_country']==True]


# In[122]:


df_clubs_origin = df.groupby("Club origin",as_index=False).agg({"Plays_in_home_country":"count"})
df_clubs_origin


# In[127]:


df_clubs_origin_sorted =df_clubs_origin.sort_values(['Plays_in_home_country'],ascending=False)
df_clubs_origin_sorted.head(10)


# In[136]:


df_clubs_origin_sorted['Club origin'][0:10]


# In[133]:


ax = df_clubs_origin_sorted['Plays_in_home_country'][0:10].plot(x=df_clubs_origin_sorted['Club origin'][0:10],kind="bar",color=['blue'])


# In[74]:


df_clubs = df.groupby("Club",as_index=True).agg({'Club':['count']}).sort_values([("Club","count")],ascending=False)
# df.groupby('Team', as_index=False).agg({'Height':['mean','median','std']}).sort_values([("Height","mean")])
df_clubs


# In[75]:


df['Team'].unique()


# In[76]:


round_16_teams=['Argentina', 'Belgium', 'Brazil', 'Colombia',
       'Croatia', 'Denmark', 'England', 'France',
       'Japan', 
       'Mexico', 
       'Portugal', 'Russia', 'Spain',
       'Sweden', 'Switzerland', 'Uruguay']
len(round_16_teams)


# In[79]:


quarter_finals_temas = ['Belgium', 'Brazil',
       'Croatia',  'England', 'France',
        'Russia', 
       'Sweden','Uruguay']
len(quarter_finals_temas)


# In[84]:


df_only_R16 = df[df['Team'].isin( round_16_teams)]
df_only_R16['Team'].unique()


# In[85]:


df_only_QF = df[df['Team'].isin(quarter_finals_temas)]
df_only_QF['Team'].unique()


# In[116]:


def visualise_top_k_represented_clubs(df, k):
    df_grouped = df.groupby("Club",as_index=True).agg({'Club':['count']}).sort_values([("Club","count")],ascending=False)
    df_grouped_top_k=df_grouped[0:k]
    ax = df_grouped_top_k['Club']['count'].plot(x=df_grouped_top_k.index,kind="bar",color=['blue'])
    
    # ax.set_xticklabels(df_age['Team'], rotation=90)
    return ax


# In[104]:


visualise_top_k_represented_clubs(df,10)


# In[96]:


visualise_top_k_represented_clubs(df_only_R16,10)


# In[97]:


visualise_top_k_represented_clubs(df_only_QF,10)


# In[120]:


fig, ax = plt.subplots()
fig.set_figheight(5)
fig.set_figwidth(15)

plt.subplot(1,3,1)
plt.ylim((0,17)) 
ax = visualise_top_k_represented_clubs(df,15)
ax = plt.gca()

plt.subplot(1,3,2)
plt.ylim((0,17))
ax2 = visualise_top_k_represented_clubs(df_only_R16,15)
ax2 = plt.gca()


plt.subplot(1,3,3)
plt.ylim((0,17))
ax3 = visualise_top_k_represented_clubs(df_only_QF,15)
ax3 = plt.gca()


# In[143]:


"""
==================
Animated histogram
==================

This example shows how to use a path patch to draw a bunch of
rectangles for an animated histogram.

"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation
from IPython.display import HTML
fig, ax = plt.subplots()

# histogram our data with numpy
data = np.random.randn(1000)
n, bins = np.histogram(data, 100)

# get the corners of the rectangles for the histogram
left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + n
nrects = len(left)

# here comes the tricky part -- we have to set up the vertex and path
# codes arrays using moveto, lineto and closepoly

# for each rect: 1 for the MOVETO, 3 for the LINETO, 1 for the
# CLOSEPOLY; the vert for the closepoly is ignored but we still need
# it to keep the codes aligned with the vertices
nverts = nrects*(1 + 3 + 1)
verts = np.zeros((nverts, 2))
codes = np.ones(nverts, int) * path.Path.LINETO
codes[0::5] = path.Path.MOVETO
codes[4::5] = path.Path.CLOSEPOLY
verts[0::5, 0] = left
verts[0::5, 1] = bottom
verts[1::5, 0] = left
verts[1::5, 1] = top
verts[2::5, 0] = right
verts[2::5, 1] = top
verts[3::5, 0] = right
verts[3::5, 1] = bottom

barpath = path.Path(verts, codes)
patch = patches.PathPatch(
    barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
ax.add_patch(patch)

ax.set_xlim(left[0], right[-1])
ax.set_ylim(bottom.min(), top.max())


def animate(i):
    # simulate new data coming in
    data = np.random.randn(1000)
    n, bins = np.histogram(data, 100)
    top = bottom + n
    verts[1::5, 1] = top
    verts[2::5, 1] = top
    return [patch, ]

ani = animation.FuncAnimation(fig, animate, 1000, repeat=False, blit=True)
plt.show()
# HTML(ani.to_html5_video())
ani.save('myvideo.mp4', codec='h264')


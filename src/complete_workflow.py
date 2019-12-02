
# -*- coding: utf-8 -*-
'''
# Created on Nov-21-19 11:11
# compelte_workflow.py
'''

import numpy as np 
import pandas as pd 


# import user profile information
user = pd.read_excel('RnAPortal.xlsx', 'PORTAL_USERS')
user_ids = user.USER_PID.unique().tolist()

# import apps metadata information
app = pd.read_excel('RnAPortal.xlsx', 'PORTAL_APPS')

# fillna of average rating
app['AVG_RATING'] = app['AVG_RATING'].fillna(0)
app_ids = app.APP_ID.unique().tolist()

# import entitlement information
ent = pd.read_excel('RnAPortal.xlsx', 'APP_ENTITLEMENTS')

# import log information
log = pd.read_excel('RnAPortal.xlsx', 'APP_ACCESS_LOG')

# delete rows with missing APP_ID
log = log.dropna()
log['APP_ID'] = log['APP_ID'].astype(int)

# filter the app that apears in app_ids
log = log[log['APP_ID'].isin(app_ids)]
log_user = log.USER_PID.unique().tolist()
log_app = log.APP_ID.unique().tolist()

'''
    
Description on raw data:

1. Total number of users: 3408, total number of apps: 262.<br>
2. Number of users who interacted with apps is: 1541. Number of apps being used is: 211. (based on log data) <br>
3. Number of users of any kind of record filtering is 1634, number of apps of any kind of record is 224.

Temporary Conclusion: 

1. It is noticeable that some users rate or create workspace without actually using apps. Can this data be counted as reliable? <br>
2. We can use workspace, log, rating, app category and user attributes for collaborative filtering recommendation. Rating and log can be used to rank recommendations.<br>
3. For new users, use user-based collaborative filtering recommendation.<br>
4. For existing users, use hybrid collaborative filtering and content-based recommendation.<br>
5. For new apps, use content-based recommendation.</font>

'''


# Combine Dataframes

## Add total log time into app_df
log_sum = pd.DataFrame()
log_sum['log_sum'] = log.groupby('APP_ID')['USAGE_TIME_MINS'].sum()
app_new = pd.merge(app, log_sum, on = 'APP_ID', how = 'left')
app_new['log_sum'] = app_new['log_sum'].fillna(0)


## Combine user and log

# this shows how user interact with apps
user_log = pd.merge(user, log, on='USER_PID', how = "left")
user_log['USAGE_TIME_MINS'] = user_log['USAGE_TIME_MINS'].fillna(0)

log_time = pd.DataFrame(user_log.groupby(['USER_PID','APP_ID']).sum())
log_time.columns = ['total_time'] 
log_time = log_time.reset_index()


# user & mean_usage_time matrix
L_df = log_time.pivot_table(index='USER_PID',columns='APP_ID',values='total_time').fillna(0)

# Normalization of Conversion to numpy array
# De-mean the data (normalize by each users mean) and convert it from a dataframe to a numpy array.

L = L_df.values
user_log_mean = np.mean(L, axis = 1)
L_demeaned = L - user_log_mean.reshape(-1, 1)

user_log_mean.reshape(-1, 1)


# Singular Value Decomposition

from scipy.sparse.linalg import svds
U, sigma, Vt = svds(L_demeaned, k = 60)
sigma = np.diag(sigma)


# Making Predictions from the Decomposed Matrices


all_user_predicted_logs = np.dot(np.dot(U, sigma), Vt) + user_log_mean.reshape(-1, 1)


# Making App Recommendations

preds_df = pd.DataFrame(all_user_predicted_logs, columns = L_df.columns)


exi_users = log_user
new_users = [x for x in user_ids if x not in exi_users]

user_df = pd.read_excel('RnAPortal.xlsx', 'PORTAL_USERS')
# log_df = pd.read_excel('RnAPortal.xlsx', 'APP_ACCESS_LOG')


# Content-based Filtering for new users
from collections import defaultdict
mapped_users = defaultdict(list)
MAPPING_LIMIT = 5

# mapping process I: map new users to existing users who are in the same department
def mapping_user_dep(new_users, exi_users, mapped_users, limit):
    for new_u in new_users:
        department = user_df.loc[user_df['USER_PID'] == new_u, 'DEPARTMENT_NAME']
        users_in_department = user_df.loc[user_df['DEPARTMENT_NAME'] == department.iloc[0], 'USER_PID']

        for user in users_in_department:
            if user in exi_users and user not in mapped_users[new_u] and len(mapped_users[new_u]) < limit:
                mapped_users[new_u].append(user)
                
mapping_user_dep(new_users, exi_users, mapped_users, MAPPING_LIMIT)

# check mapping results
# def check_mpp(mapped_users, newuser_cnt):
#     mpp_cnt = len(mapped_users)
#     print(mpp_cnt, newuser_cnt, mpp_cnt / newuser_cnt)
    
# check_mpp(mapped_users, len(new_users))

# mapping process II: map new users to existing users who are in the same division

def mapping_user_div(new_users, exi_users, mapped_users, limit):
    for new_u in new_users:
        division = user_df.loc[user_df['USER_PID'] == new_u, 'DIVISION']
        users_in_division = user_df.loc[user_df['DIVISION'] == division.iloc[0], 'USER_PID']

        for user in users_in_division:
            if user in exi_users and user not in mapped_users[new_u] and len(mapped_users[new_u]) < limit:
                mapped_users[new_u].append(user)
                
mapping_user_div(new_users, exi_users, mapped_users, MAPPING_LIMIT)

# check mapping results
# check_mpp(mapped_users, len(new_users))

# Users in descending order of number of log records
user_sort = log_time.groupby('USER_PID').size().sort_values(ascending=False)
user_list = user_sort.index.tolist()

from operator import itemgetter
for k,v in mapped_users.items():
    sorted(v, key=lambda x: user_list.index(x), reverse=True)

def recommend_apps(predictions_df, user_pid, apps_df, user_log_df, num_recommendations=5):
    
    flag = 0
    original_pid = user_pid
    if user_pid in mapped_users:
        user_mapped = mapped_users[user_pid][0]
        user_pid = user_mapped
        flag = 1 # this is a new user
        
    # Get and sort the user's predictions
    user_row_number = log_user.index(user_pid) # find index of UserID 
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False) 

    # Get the user's data and merge in the app information.
    user_log_df = user_log_df[['USER_PID','APP_ID']]
    apps_df = apps_df[['APP_ID','APP_NAME','BUSINESS_DOMAIN','CATEGORY','APP_TYPE','log_sum']]
    user_data = user_log_df[user_log_df.USER_PID == (user_pid)]
    user_full = (user_data.merge(apps_df, how = 'left', left_on = 'APP_ID', right_on = 'APP_ID').
                     sort_values(['log_sum'], ascending=False))
    new_user = user[['USER_PID','COUNTRY','DEPARTMENT_NAME','DIVISION']]
    user_full['USER_PID'] = user_pid
    user_full = pd.merge(new_user,user_full,on="USER_PID")

    if flag == 0:
        # Recommend the highest predicted app that the user hasn't used yet.
        recommendations = (apps_df[~apps_df['APP_ID'].isin(user_full['APP_ID'])].
             merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
                   left_on = 'APP_ID',
                   right_on = 'APP_ID').
             rename(columns = {user_row_number: 'Predictions'}).
             sort_values('Predictions', ascending = False).
                           iloc[:num_recommendations, :-1])
        # print ('User {0} has already used {1} apps.'.format(user_pid, user_full.shape[0]))
        # print ('Recommending apps with highest {0} predicted value not already used.'.format(num_recommendations))
    else:
        # All apps included
        recommendations = (apps_df.
             merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
                   left_on = 'APP_ID',
                   right_on = 'APP_ID').
             rename(columns = {user_row_number: 'Predictions'}).
             sort_values('Predictions', ascending = False).
                           iloc[:num_recommendations, :-1])
        # print ('User {0} has already used 0 apps.'.format(original_pid))
        # print ('Mapped user {0} has already used {1} apps.'.format(user_pid, user_full.shape[0]))
        # print ('Recommending apps with highest {0} predicted value.'.format(num_recommendations))
    recommendations['USER_PID'] = original_pid
    new_user = user[['USER_PID','COUNTRY','DEPARTMENT_NAME','DIVISION']]
    recommendations = pd.merge(new_user,recommendations,on="USER_PID")



    # ent_apps = ent[ent['USER_PID'] == user_pid]['APP_ID']
    # recommendations = recommendations[recommendations['APP_ID'].isin(ent_apps)]

    return user_full, recommendations

already_used, predictions = recommend_apps(preds_df, 236193, app_new, log_time)

print(already_used)

print(predictions)
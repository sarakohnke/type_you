#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:07:39 2020

@author: sarakohnke
"""
import pickle
#to open normal pickle
with open('model_pkl.pickle','rb') as input_file:
    model=pickle.load(input_file)


#Make patient lists from test data for app
patient1_list=X_test_rf2.iloc[115,:].values.tolist()
patient2_list=X_test_rf2.iloc[253,:].values.tolist()
patient3_list=X_test_rf2.iloc[603,:].values.tolist()

#Find current predicted A1C score
A1C_prediction_drug1_2 = clf_rf2.predict([patient1_list])
print('your current score: '+str(A1C_prediction_drug1_2[0]))

#insulin - 1-3?->just1
patient1_list[6]=1
A1C_prediction_drug1_2 = clf_rf2.predict([patient1_list])
print('your predicted score with insulin: '+str(A1C_prediction_drug1_2[0]))

#bmi    
patient1_list[18]=24
A1C_prediction_drug1_2 = clf_rf2.predict([patient1_list])
print('your predicted score with ideal bmi: '+str(A1C_prediction_drug1_2[0]))

#bp     
patient1_list[3]=120
patient1_list[4]=80
A1C_prediction_drug1_2 = clf_rf2.predict([patient1_list])
print('your predicted score with ideal blood pressure: '+str(A1C_prediction_drug1_2[0]))

#triglycerides
patient1_list[32]=150
A1C_prediction_drug1_2 = clf_rf2.predict([patient1_list])
print('your predicted score with ideal triglycerides: '+str(A1C_prediction_drug1_2[0]))

#healthy diet
patient1_list[43]=1
A1C_prediction_drug1_2 = clf_rf2.predict([patient1_list])
print('your predicted score with healthy diet: '+str(A1C_prediction_drug1_2[0]))

#cholesterol 
patient1_list[19]=170
A1C_prediction_drug1_2 = clf_rf2.predict([patient1_list])
print('your predicted score with cholesterol: '+str(A1C_prediction_drug1_2[0]))
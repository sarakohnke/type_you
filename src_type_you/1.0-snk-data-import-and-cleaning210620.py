#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 21:13:58 2020

@author: sarakohnke
"""

#Set working directory
import os
path="/Users/sarakohnke/Desktop/data_type_you/interim-tocsv"
os.chdir(path)
os.getcwd()

#Import required packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#Import data for files with demographic information
#(This is after converted .xpt to .csv)
#import 7 demo files
demo05=pd.read_csv('DEMO_D_NHANES_Demographics_2005.csv')
demo07=pd.read_csv('DEMO_E_NHANES_Demographics_2007.csv')
demo09=pd.read_csv('DEMO_F_NHANES_Demographics_2009.csv')
demo11=pd.read_csv('DEMO_G_NHANES_Demographics_2011.csv')
demo13=pd.read_csv('DEMO_H_NHANES_Demographics_2013.csv')
demo15=pd.read_csv('DEMO_I_NHANES_Demographics_2015.csv')
demo17=pd.read_csv('DEMO_J_NHANES_Demographics_2017.csv')

#add year as a column
demo05['Year']=2005
demo07['Year']=2007
demo09['Year']=2009
demo11['Year']=2011
demo13['Year']=2013
demo15['Year']=2015
demo17['Year']=2017

#append all dfs together
demographics_allyears=demo05.append(demo07, ignore_index = True) 
demographics_allyears=demographics_allyears.append(demo09, ignore_index = True)
demographics_allyears=demographics_allyears.append(demo11, ignore_index = True)
demographics_allyears=demographics_allyears.append(demo13, ignore_index = True)
demographics_allyears=demographics_allyears.append(demo15, ignore_index = True)
demographics_allyears=demographics_allyears.append(demo17, ignore_index = True)

#select only desired cols
demographics_allyears2=demographics_allyears[['SEQN','RIAGENDR','RIDAGEYR']].copy()

#rename cols
demographics_allyears2.rename(columns={'SEQN':'Patient ID',
                                       'RIAGENDR':'Male',
                                       'RIDAGEYR':'Age (years)'}, inplace=True)

#see if there are unknowns (eg 777)
demographics_allyears2['Age (years)'].value_counts().sort_index()

#replace 2 with 0 for female
demographics_allyears2['Male'].replace(2,0,inplace=True)

#drop rows with nas
demographics_allyears3=demographics_allyears2.dropna(axis=0)

#filter for adults
demographics_allyears4=demographics_allyears3[demographics_allyears3['Age (years)']>=18]

#Import data for files with blood pressure information
#import 7 bp files
bp05=pd.read_csv('BPX_D_NHANES_Blood_Pressure_2005.csv')
bp07=pd.read_csv('BPX_E_NHANES_Blood_Pressure_2007.csv')
bp09=pd.read_csv('BPX_F_NHANES_Blood_Pressure_2009.csv')
bp11=pd.read_csv('BPX_G_NHANES_Blood_Pressure_2011.csv')
bp13=pd.read_csv('BPX_H_NHANES_Blood_Pressure_2013.csv')
bp15=pd.read_csv('BPX_I_NHANES_Blood_Pressure_2015.csv')
bp17=pd.read_csv('BPX_J_NHANES_Blood_Pressure_2017.csv')

#add year as a column
bp05['Year']=2005
bp07['Year']=2007
bp09['Year']=2009
bp11['Year']=2011
bp13['Year']=2013
bp15['Year']=2015
bp17['Year']=2017

#append all dfs together
bp_allyears=bp05.append(bp07, ignore_index = True) 
bp_allyears=bp_allyears.append(bp09, ignore_index = True)
bp_allyears=bp_allyears.append(bp11, ignore_index = True)
bp_allyears=bp_allyears.append(bp13, ignore_index = True)
bp_allyears=bp_allyears.append(bp15, ignore_index = True)
bp_allyears=bp_allyears.append(bp17, ignore_index = True)

#select only desired cols
bp_allyears2=bp_allyears[['SEQN','BPXPLS','BPXSY1','BPXDI1']].copy()

#rename cols
bp_allyears2.rename(columns={'SEQN':'Patient ID',
                             'BPXPLS':'Pulse (60sec)',
                             'BPXSY1':'Systolic pressure (mmHg)',
                             'BPXDI1':'Diastolic pressure (mmHg)'}, inplace=True)

#see if there are unknowns (eg 777)
bp_allyears2['Systolic pressure (mmHg)'].value_counts().sort_index()

#replace values that don't make sense with NaNs
bp_allyears2['Pulse (60sec)'].replace(0,np.nan,inplace=True)
bp_allyears2['Pulse (60sec)'].value_counts().sort_index()

bp_allyears2['Diastolic pressure (mmHg)'].replace([0,2,4,6,8,10,12,14,16,18],np.nan,inplace=True)
bp_allyears2['Diastolic pressure (mmHg)'].value_counts().sort_index()

#drop rows with nas
bp_allyears3=bp_allyears2.dropna(axis=0)

#Import data for files with body measure information
#import 7 body measure files
bm05=pd.read_csv('BMX_D_NHANES_Body_Measures_2005.csv')
bm07=pd.read_csv('BMX_E_NHANES_Body_Measures_2007.csv')
bm09=pd.read_csv('BMX_F_NHANES_Body_Measures_2009.csv')
bm11=pd.read_csv('BMX_G_NHANES_Body_Measures_2011.csv')
bm13=pd.read_csv('BMX_H_NHANES_Body_Measures_2013.csv')
bm15=pd.read_csv('BMX_I_NHANES_Body_Measures_2015.csv')
bm17=pd.read_csv('BMX_J_NHANES_Body_Measures_2017.csv')

#add year as a column
bm05['Year']=2005
bm07['Year']=2007
bm09['Year']=2009
bm11['Year']=2011
bm13['Year']=2013
bm15['Year']=2015
bm17['Year']=2017

#append all dfs together
bm_allyears=bm05.append(bm07, ignore_index = True) 
bm_allyears=bm_allyears.append(bm09, ignore_index = True)
bm_allyears=bm_allyears.append(bm11, ignore_index = True)
bm_allyears=bm_allyears.append(bm13, ignore_index = True)
bm_allyears=bm_allyears.append(bm15, ignore_index = True)
bm_allyears=bm_allyears.append(bm17, ignore_index = True)

#select only desired cols
bm_allyears2=bm_allyears[['SEQN','BMXBMI']].copy()

#rename cols
bm_allyears2.rename(columns={'SEQN':'Patient ID',
                             'BMXBMI':'BMI (kg/m2)'}, inplace=True)

#see if there are unknowns (eg 777)
bm_allyears2['BMI (kg/m2)'].value_counts().sort_index()

#drop rows with nas
bm_allyears3=bm_allyears2.dropna(axis=0)

#Import data for files with total cholesterol information
#import 7 chol files
chol05=pd.read_csv('TCHOL_D_NHANES_Total_Cholesterol_2005.csv')
chol07=pd.read_csv('TCHOL_E_NHANES_Total_Cholesterol_2007.csv')
chol09=pd.read_csv('TCHOL_F_NHANES_Total_Cholesterol_2009.csv')
chol11=pd.read_csv('TCHOL_G_NHANES_Total_Cholesterol_2011.csv')
chol13=pd.read_csv('TCHOL_H_NHANES_Total_Cholesterol_2013.csv')
chol15=pd.read_csv('TCHOL_I_NHANES_Total_Cholesterol_2015.csv')
chol17=pd.read_csv('TCHOL_J_NHANES_Total_Cholesterol_2017.csv')

#add year as a column
chol05['Year']=2005
chol07['Year']=2007
chol09['Year']=2009
chol11['Year']=2011
chol13['Year']=2013
chol15['Year']=2015
chol17['Year']=2017

#append all dfs together
chol_allyears=chol05.append(chol07, ignore_index = True) 
chol_allyears=chol_allyears.append(chol09, ignore_index = True)
chol_allyears=chol_allyears.append(chol11, ignore_index = True)
chol_allyears=chol_allyears.append(chol13, ignore_index = True)
chol_allyears=chol_allyears.append(chol15, ignore_index = True)
chol_allyears=chol_allyears.append(chol17, ignore_index = True)

#select only desired cols
chol_allyears2=chol_allyears[['SEQN','LBXTC']].copy()

#rename cols
chol_allyears2.rename(columns={'SEQN':'Patient ID',
                             'LBXTC':'Total cholesterol (mg/dl)'}, inplace=True)


#see if there are unknowns (eg 777)
chol_allyears2['Total cholesterol (mg/dl)'].value_counts().sort_index()

#drop rows with nas
chol_allyears3=chol_allyears2.dropna(axis=0)

#Import data for files with blood count information
#import 7 blood count files
cbc05=pd.read_csv('CBC_D_NHANES_Complete_Blood_Count_2005.csv')
cbc07=pd.read_csv('CBC_E_NHANES_Complete_Blood_Count_2007.csv')
cbc09=pd.read_csv('CBC_F_NHANES_Complete_Blood_Count_2009.csv')
cbc11=pd.read_csv('CBC_G_NHANES_Complete_Blood_Count_2011.csv')
cbc13=pd.read_csv('CBC_H_NHANES_Complete_Blood_Count_2013.csv')
cbc15=pd.read_csv('CBC_I_NHANES_Complete_Blood_Count_2015.csv')
cbc17=pd.read_csv('CBC_J_NHANES_Complete_Blood_Count_2017.csv')

#add year as a column
cbc05['Year']=2005
cbc07['Year']=2007
cbc09['Year']=2009
cbc11['Year']=2011
cbc13['Year']=2013
cbc15['Year']=2015
cbc17['Year']=2017

#append all dfs together
cbc_allyears=cbc05.append(cbc07, ignore_index = True) 
cbc_allyears=cbc_allyears.append(cbc09, ignore_index = True)
cbc_allyears=cbc_allyears.append(cbc11, ignore_index = True)
cbc_allyears=cbc_allyears.append(cbc13, ignore_index = True)
cbc_allyears=cbc_allyears.append(cbc15, ignore_index = True)
cbc_allyears=cbc_allyears.append(cbc17, ignore_index = True)

#select only desired cols
cbc_allyears2=cbc_allyears[['SEQN','LBXMPSI','LBXPLTSI','LBXRBCSI','LBDEONO',
                            'LBDLYMNO','LBDBANO','LBDMONO']].copy()

#rename cols
cbc_allyears2.rename(columns={'SEQN':'Patient ID',
                             'LBXMPSI':'Mean platelet volume (fL)',
                             'LBXPLTSI':'Platelet count (1000 cells/uL)',
                             'LBXRBCSI':'Red blood cell count (million cells/uL)',
                             'LBDEONO':'Eosinophils number (1000 cells/uL)',
                             'LBDLYMNO':'Lymphocyte number (1000 cells/uL)',
                             'LBDBANO':'Basophils number (1000 cells/uL)',
                             'LBDMONO':'Monocyte number (1000 cells/uL)'}, 
                 inplace=True)

#see if there are unknowns (eg 777)
cbc_allyears2['Monocyte number (1000 cells/uL)'].value_counts().sort_index()

#drop rows with nas
cbc_allyears3=cbc_allyears2.dropna(axis=0)

#Import data for files with A1c/glycohemoglobin information
#import 7 a1c files
a1c05=pd.read_csv('GHB_D_NHANES_A1C_2005.csv')
a1c07=pd.read_csv('GHB_E_NHANES_A1C_2007.csv')
a1c09=pd.read_csv('GHB_F_NHANES_A1C_2009.csv')
a1c11=pd.read_csv('GHB_G_NHANES_A1C_2011.csv')
a1c13=pd.read_csv('GHB_H_NHANES_A1C_2013.csv')
a1c15=pd.read_csv('GHB_I_NHANES_A1C_2015.csv')
a1c17=pd.read_csv('GHB_J_NHANES_A1C_2017.csv')

#add year as a column
a1c05['Year']=2005
a1c07['Year']=2007
a1c09['Year']=2009
a1c11['Year']=2011
a1c13['Year']=2013
a1c15['Year']=2015
a1c17['Year']=2017

#append all dfs together
a1c_allyears=a1c05.append(a1c07, ignore_index = True) 
a1c_allyears=a1c_allyears.append(a1c09, ignore_index = True)
a1c_allyears=a1c_allyears.append(a1c11, ignore_index = True)
a1c_allyears=a1c_allyears.append(a1c13, ignore_index = True)
a1c_allyears=a1c_allyears.append(a1c15, ignore_index = True)
a1c_allyears=a1c_allyears.append(a1c17, ignore_index = True)

#rename cols
a1c_allyears.rename(columns={'SEQN':'Patient ID',
                             'LBXGH':'A1C (%)'},
                              inplace=True)

#see if there are unknowns (eg 777)
a1c_allyears['A1C (%)'].value_counts().sort_index()

#drop rows with nas
a1c_allyears2=a1c_allyears.dropna(axis=0)

#Import data for files with standard bio information
#import 7 standard bio files
sb05=pd.read_csv('BIOPRO_D_NHANES_Standard_Biochemistry_Profile_2005.csv')
sb07=pd.read_csv('BIOPRO_E_NHANES_Standard_Biochemistry_Profile_2007.csv')
sb09=pd.read_csv('BIOPRO_F_NHANES_Standard_Biochemistry_Profile_2009.csv')
sb11=pd.read_csv('BIOPRO_G_NHANES_Standard_Biochemistry_Profile_2011.csv')
sb13=pd.read_csv('BIOPRO_H_NHANES_Standard_Biochemistry_Profile_2013.csv')
sb15=pd.read_csv('BIOPRO_I_NHANES_Standard_Biochemistry_Profile_2015.csv')
sb17=pd.read_csv('BIOPRO_J_NHANES_Standard_Biochemistry_Profile_2017.csv')

#add year as a column
sb05['Year']=2005
sb07['Year']=2007
sb09['Year']=2009
sb11['Year']=2011
sb13['Year']=2013
sb15['Year']=2015
sb17['Year']=2017

#append all dfs together
sb_allyears=sb05.append(sb07, ignore_index = True) 
sb_allyears=sb_allyears.append(sb09, ignore_index = True)
sb_allyears=sb_allyears.append(sb11, ignore_index = True)
sb_allyears=sb_allyears.append(sb13, ignore_index = True)
sb_allyears=sb_allyears.append(sb15, ignore_index = True)
sb_allyears=sb_allyears.append(sb17, ignore_index = True)

#select only desired cols
sb_allyears2=sb_allyears[['SEQN','LBXSKSI','LBXSNASI','LBXSGB','LBXSPH',
                            'LBXSUA','LBXSTR','LBXSTB','LBXSLDSI','LBXSIR','LBXSGTSI',
                            'LBXSC3SI','LBXSCA','LBXSBU','LBXSAPSI','LBXSATSI','LBXSAL']].copy()

#rename cols
sb_allyears2.rename(columns={'SEQN':'Patient ID',
                             'LBXSKSI':'Potassium (mmol/L)',
                             'LBXSNASI':'Sodium (mmol/L)',
                             'LBXSGB':'Globulin (g/dL)',
                             'LBXSPH':'Phosphorus (mg/dL)',
                             'LBXSUA':'Uric acid (mg/dL)',
                             'LBXSTR':'Triglycerides (mg/dL)',
                             'LBXSTB':'Bilirubin (mg/dL)',
                             'LBXSLDSI':'Lactate dehydrogenase (IU/L)',
                             'LBXSIR':'Iron (ug/dL)',
                             'LBXSGTSI':'Gamma glutamyl transferase (IU/L)',
                             'LBXSC3SI':'Bicarbonate (mmol/L)',
                             'LBXSCA':'Calcium (mg/dL)',
                             'LBXSBU':'Blood Urea Nitrogen (mg/dL)',
                             'LBXSAPSI':'Alkaline phosphatase (IU/L)',
                             'LBXSATSI':'Alanine aminotransferase (IU/L)',
                             'LBXSAL':'Albumin (g/dL)'}, 
                 inplace=True)

#see if there are unknowns (eg 777)
sb_allyears2['Albumin (g/dL)'].value_counts().sort_index()

#replace values that don't make sense with NaNs
sb_allyears2['Triglycerides (mg/dL)'].replace(6057,np.nan,inplace=True)
sb_allyears2['Triglycerides (mg/dL)'].value_counts().sort_index()

sb_allyears2['Alanine aminotransferase (IU/L)'].replace(1363,np.nan,inplace=True)
sb_allyears2['Alanine aminotransferase (IU/L)'].value_counts().sort_index()

#drop rows with nas
sb_allyears3=sb_allyears2.dropna(axis=0)

#Import data for files with diet information
#import 7 diet files
diet05=pd.read_csv('DBQ_D_NHANES_Diet_Behavior_and_Nutrition_2005.csv')
diet07=pd.read_csv('DBQ_E_NHANES_Diet_Behavior_and_Nutrition_2007.csv')
diet09=pd.read_csv('DBQ_F_NHANES_Diet_Behavior_and_Nutrition_2009.csv')
diet11=pd.read_csv('DBQ_G_NHANES_Diet_Behavior_and_Nutrition_2011.csv')
diet13=pd.read_csv('DBQ_H_NHANES_Diet_Behavior_and_Nutrition_2013.csv')
diet15=pd.read_csv('DBQ_I_NHANES_Diet_Behavior_and_Nutrition_2015.csv')
diet17=pd.read_csv('DBQ_J_NHANES_Diet_Behavior_and_Nutrition_2017.csv')

#add year as a column
diet05['Year']=2005
diet07['Year']=2007
diet09['Year']=2009
diet11['Year']=2011
diet13['Year']=2013
diet15['Year']=2015
diet17['Year']=2017

#append all dfs together
diet_allyears=diet05.append(diet07, ignore_index = True) 
diet_allyears=diet_allyears.append(diet09, ignore_index = True)
diet_allyears=diet_allyears.append(diet11, ignore_index = True)
diet_allyears=diet_allyears.append(diet13, ignore_index = True)
diet_allyears=diet_allyears.append(diet15, ignore_index = True)
diet_allyears=diet_allyears.append(diet17, ignore_index = True)

#select only desired COLUMNS
diet_allyears2=diet_allyears[['SEQN','DBQ700']].copy()

#rename COLS
diet_allyears2.rename(columns={'SEQN':'Patient ID',
                             'DBQ700':'How healthy diet (1 is best)'
                              }, 
                 inplace=True)

diet_allyears2['How healthy diet (1 is best)'].replace([7,9],np.nan,inplace=True)
diet_allyears2['How healthy diet (1 is best)'].value_counts().sort_index()

#drop rows with nas
diet_allyears3=diet_allyears2.dropna(axis=0)

#Import data for files with medication information
#import 7 medication files
drug05=pd.read_csv('RXQ_RX_D_NHANES_Prescription_Medications_2005.csv')
drug07=pd.read_csv('RXQ_RX_E_NHANES_Prescription_Medications_2007.csv')
drug09=pd.read_csv('RXQ_RX_F_NHANES_Prescription_Medications_2009.csv')
drug11=pd.read_csv('RXQ_RX_G_NHANES_Prescription_Medications_2011.csv')
drug13=pd.read_csv('RXQ_RX_H_NHANES_Prescription_Medications_2013.csv',encoding='windows-1252')
drug15=pd.read_csv('RXQ_RX_I_NHANES_Prescription_Medications_2015.csv')
drug17=pd.read_csv('RXQ_RX_J_NHANES_Prescription_Medications_2017.csv')

#add year as a column
drug05['Year']=2005
drug07['Year']=2007
drug09['Year']=2009
drug11['Year']=2011
drug13['Year']=2013
drug15['Year']=2015
drug17['Year']=2017

#append all dfs together
drug_allyears=drug05.append(drug07, ignore_index = True) 
drug_allyears=drug_allyears.append(drug09, ignore_index = True)
drug_allyears=drug_allyears.append(drug11, ignore_index = True)
drug_allyears=drug_allyears.append(drug13, ignore_index = True)
drug_allyears=drug_allyears.append(drug15, ignore_index = True)
drug_allyears=drug_allyears.append(drug17, ignore_index = True)

#select only desired COLUMNS
drug_allyears2=drug_allyears[['SEQN','RXDDRUG','RXDCOUNT']].copy()

#rename COLS
drug_allyears2.rename(columns={'SEQN':'Patient ID',
                             'RXDDRUG':'Generic drug name',
                               'RXDCOUNT':'Number of Rx drugs taking'
                              }, 
                 inplace=True)

#replace unknowns
drug_allyears2['Generic drug name'].value_counts().sort_index()
drug_allyears2['Generic drug name'].replace(['55555','77777','99999'],np.nan,inplace=True)
drug_allyears2['Generic drug name'].value_counts().sort_index()

#drop rows with nas
drug_allyears3=drug_allyears2.dropna(axis=0)

#Import data for files with sleep information
#import 7 sleep files
sleep05=pd.read_csv('SLQ_D_NHANES_Sleep_2005.csv')
sleep07=pd.read_csv('SLQ_E_NHANES_Sleep_2007.csv')
sleep09=pd.read_csv('SLQ_F_NHANES_Sleep_2009.csv')
sleep11=pd.read_csv('SLQ_G_NHANES_Sleep_2011.csv')
sleep13=pd.read_csv('SLQ_H_NHANES_Sleep_2013.csv')
sleep15=pd.read_csv('SLQ_I_NHANES_Sleep_2015.csv')
sleep17=pd.read_csv('SLQ_J_NHANES_Sleep_2017.csv')

#add year as a column
sleep05['Year']=2005
sleep07['Year']=2007
sleep09['Year']=2009
sleep11['Year']=2011
sleep13['Year']=2013
sleep15['Year']=2015
sleep15.rename(columns={'SLD012':'SLD010H'}, inplace=True)
sleep17['Year']=2017
sleep17.rename(columns={'SLD012':'SLD010H'}, inplace=True)

#append all dfs together
sleep_allyears=sleep05.append(sleep07, ignore_index = True) 
sleep_allyears=sleep_allyears.append(sleep09, ignore_index = True)
sleep_allyears=sleep_allyears.append(sleep11, ignore_index = True)
sleep_allyears=sleep_allyears.append(sleep13, ignore_index = True)
sleep_allyears=sleep_allyears.append(sleep15, ignore_index = True)
sleep_allyears=sleep_allyears.append(sleep17, ignore_index = True)

#select only desired COLUMNS
sleep_allyears2=sleep_allyears[['SEQN','SLD010H']].copy()

#rename COLS
sleep_allyears2.rename(columns={'SEQN':'Patient ID',
                             'SLD010H':'Average sleep (hours)'                             
                              }, 
                 inplace=True)

sleep_allyears2['Average sleep (hours)'].replace([77,99],np.nan,inplace=True)

#drop rows with nas
sleep_allyears3=sleep_allyears2.dropna(axis=0)

#Drugs file has multiple entries for individuals - aggregate
drug_allyears3.to_csv('drugallyears3.csv')

drug_allyears3_2=pd.read_csv('drugallyears3_2.csv',index_col=[0])

drug_allyears4=pd.get_dummies(drug_allyears3_2,prefix=['Generic drug name'],columns=['Generic drug name'])

drug_allyears5=drug_allyears4.groupby('Patient ID').agg({
 'Number of Rx drugs taking':'first',
 'Generic drug name_Insulin':'sum',
 'Generic drug name_Alpha-glucosidase inhibitor':'sum',
 'Generic drug name_Biguanide':'sum',                                            
 'Generic drug name_DPP-4 inhibitor':'sum',
 'Generic drug name_DPP-4 inhibitor; Biguanide':'sum',
 'Generic drug name_GLP-1R agonist':'sum',
 'Generic drug name_Meglitinide':'sum',                                                
 'Generic drug name_SGLT2 inhibitor':'sum',
 'Generic drug name_SGLT2 inhibitor; Biguanide':'sum',
 'Generic drug name_Sulfonylurea':'sum',
'Generic drug name_Sulfonylurea; Biguanide':'sum',
    'Generic drug name_Thiazolidinedione':'sum'}).reset_index()

#merge all dataframes together
dataframe=demographics_allyears4.merge(bp_allyears3,how='inner',on='Patient ID')
dataframe=dataframe.merge(drug_allyears5,how='outer',on='Patient ID')

dataframe2=pd.read_csv('drugdataframe2.csv',index_col=[0])

dataframe=dataframe2.merge(bm_allyears3,how='inner',on='Patient ID')
dataframe=dataframe.merge(chol_allyears3,how='inner',on='Patient ID')
dataframe=dataframe.merge(cbc_allyears3,how='inner',on='Patient ID')
dataframe=dataframe.merge(a1c_allyears2,how='inner',on='Patient ID')
dataframe=dataframe.merge(sb_allyears3,how='inner',on='Patient ID')
dataframe=dataframe.merge(diet_allyears3,how='inner',on='Patient ID')
dataframe=dataframe.merge(sleep_allyears3,how='inner',on='Patient ID')

#check for nans
dataframe.isnull().sum()

#final product!
dataframe.to_csv('dataframe240620.csv')




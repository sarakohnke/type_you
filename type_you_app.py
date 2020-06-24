import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
import bz2
import _pickle as cPickle




titleimage=Image.open('typeyoulogo.png')
a1c5=Image.open('a1c5.png')
a1c6=Image.open('a1c6.png')
a1c7=Image.open('a1c7.png')
a1c8=Image.open('a1c8.png')
a1c9=Image.open('a1c9.png')
a1c10=Image.open('a1c10.png')
a1c11=Image.open('a1c11.png')

patient_info=pd.read_csv('patient_info220620.csv',index_col=0)

patient1_list=patient_info.iloc[0,:].values.tolist()
patient2_list=patient_info.iloc[1,:].values.tolist()
patient3_list=patient_info.iloc[2,:].values.tolist()

def decompress_pickle(file):
	model=bz2.BZ2File(file,'rb')
	model=cPickle.load(model)
	return model
#model=decompress_pickle('model.bz2')
with open('model_download.pkl','rb') as input_file:
    model=pickle.load(input_file)

st.image(image=titleimage)
patient=st.selectbox('Select patient EMR data',('Patient 1','Patient 2','Patient 3'))

current1=model.predict([patient1_list])
current1_2=str(current1[0])
current2=model.predict([patient2_list])
current2_2=str(current2[0])
current3=model.predict([patient3_list])
current3_2=str(current3[0])

if patient=='Patient 1':
	st.write(patient_info.iloc[0,:])
	st.write("This patient's A1C is currently: "+current1_2[0:5])
if patient=='Patient 1':
	if current1>=11.0:
		st.image(image=a1c11)
	elif current1>=10.0:
		st.image(image=a1c10)
	elif current1>=9.0:
		st.image(image=a1c9)
	elif current1>=8.0:
		st.image(image=a1c8)
	elif current1>=7.0:
		st.image(image=a1c7)
	elif current1>=6.0:
		st.image(image=a1c6)
	else:
		st.image(image=a1c5)

if patient=='Patient 2':
	st.write(patient_info.iloc[1,:])
	st.write("This patient's A1C is currently: "+current2_2[0:5])
if patient=='Patient 2':
	if current2>=11.0:
		st.image(image=a1c11)
	elif current2>=10.0:
		st.image(image=a1c10)
	elif current2>=9.0:
		st.image(image=a1c9)
	elif current2>=8.0:
		st.image(image=a1c8)
	elif current2>=7.0:
		st.image(image=a1c7)
	elif current2>=6.0:
		st.image(image=a1c6)
	else:
		st.image(image=a1c5)

if patient=='Patient 3':
	st.write(patient_info.iloc[2,:])
	st.write("This patient's A1C is currently: "+current3_2[0:5])
if patient=='Patient 3':
	if current3>=11.0:
		st.image(image=a1c11)
	elif current3>=10.0:
		st.image(image=a1c10)
	elif current3>=9.0:
		st.image(image=a1c9)
	elif current3>=8.0:
		st.image(image=a1c8)
	elif current3>=7.0:
		st.image(image=a1c7)
	elif current3>=6.0:
		st.image(image=a1c6)
	else:
		st.image(image=a1c5)


st.sidebar.markdown('Choose possible interventions')
insulin=st.sidebar.selectbox("Add insulin",("No","Yes"))
aginhibitor=st.sidebar.selectbox("Add alpha-glucosidase inhibitor",("No","Yes"))
biguanide=st.sidebar.selectbox("Add biguanide",("No","Yes"))
dpp4inhibitor=st.sidebar.selectbox("Add DPP-4 inhibitor",("No","Yes"))
dpp4biguanide=st.sidebar.selectbox("Add DPP-4 inhibitor;biguanide combo",("No","Yes"))
glp1ragonist=st.sidebar.selectbox("Add GLP-1R agonist",("No","Yes"))
sglt2inhibitor=st.sidebar.selectbox("Add SGLT2 inhibitor",("No","Yes"))
sglt2biguanide=st.sidebar.selectbox("Add SGLT2 inhibitor;biguanide combo",("No","Yes"))
sulfonylurea=st.sidebar.selectbox("Add sufonylurea",("No","Yes"))
thiazolidinedione=st.sidebar.selectbox("Add thiazolidinedione",("No","Yes"))
bmi=st.sidebar.selectbox("Bring BMI to healthy range (24 kg/m2)",("No","Yes"))
diet=st.sidebar.selectbox("Change diet to 'excellent'",("No","Yes"))
total=st.sidebar.selectbox("Bring total cholesterol to desirable range (60 mg/dl)",("No","Yes"))
bp=st.sidebar.selectbox("Bring blood pressure to desirable range (120/80 mmHg)",("No","Yes"))
sleep=st.sidebar.selectbox("Sleep 8hr per night",("No","Yes"))

if st.button("See how your chosen interventions are predicted to affect this patient's A1C"):
	if patient=='Patient 1' and  insulin=='Yes':
		patient1_list[6]=1
	else:
		patient1_list[6]=patient1_list[6]
	if patient=='Patient 1' and  aginhibitor=='Yes':
		patient1_list[7]=1
	else:
		patient1_list[7]=patient1_list[7]
	if patient=='Patient 1' and  biguanide=='Yes':
		patient1_list[8]=1
	else:
		patient1_list[8]=patient1_list[8]
	if patient=='Patient 1' and  dpp4inhibitor=='Yes':
		patient1_list[9]=1
	else:
		patient1_list[9]=patient1_list[9]
	if patient=='Patient 1' and  dpp4biguanide=='Yes':
		patient1_list[10]=1
	else:
		patient1_list[10]=patient1_list[10]
	if patient=='Patient 1' and  glp1ragonist=='Yes':
		patient1_list[11]=1
	else:
		patient1_list[11]=patient1_list[11]
	if patient=='Patient 1' and  sglt2inhibitor=='Yes':
		patient1_list[13]=1
	else:
		patient1_list[13]=patient1_list[13]
	if patient=='Patient 1' and  sglt2biguanide=='Yes':
		patient1_list[14]=1
	else:
		patient1_list[14]=patient1_list[14]
	if patient=='Patient 1' and  sulfonylurea=='Yes':
		patient1_list[15]=1
	else:
		patient1_list[15]=patient1_list[15]
	if patient=='Patient 1' and  thiazolidinedione=='Yes':
		patient1_list[17]=1
	else:
		patient1_list[17]=patient1_list[17]
	if patient=='Patient 1' and  bmi=='Yes':
		patient1_list[18]=24
	else:
		patient1_list[18]=patient1_list[18]
	if patient=='Patient 1' and  total=='Yes':
		patient1_list[19]=150
	else:
		patient1_list[19]=patient1_list[19]
	if patient=='Patient 1' and  bp=='Yes':
		patient1_list[3]=120
		patient1_list[4]=80
	else:
		patient1_list[3]=patient1_list[3]
		patient1_list[4]=patient1_list[4]
	if patient=='Patient 1' and diet=='Yes':
		patient1_list[43]=1
	else:
		patient1_list[43]=patient1_list[43]
	if patient=='Patient 1' and sleep=='Yes':
		patient1_list[44]=8
	else:
		patient1_list[44]=patient1_list[44]
	if patient=='Patient 1':
		a1cpredict1=model.predict([patient1_list])
		a1cpredict1_1=str(a1cpredict1[0])
		st.write("This patient's predicted A1C with interventions is: "+a1cpredict1_1[0:5])
	if patient=='Patient 1':
		if a1cpredict1>=11.0:
			st.image(image=a1c11)
		elif a1cpredict1>=10.0:
			st.image(image=a1c10)
		elif a1cpredict1>=9.0:
			st.image(image=a1c9)
		elif a1cpredict1>=8.0:
			st.image(image=a1c8)
		elif a1cpredict1>=7.0:
			st.image(image=a1c7)
		elif a1cpredict1>=6.0:
			st.image(image=a1c6)
		else:
			st.image(image=a1c5)


	if patient=='Patient 2' and  insulin=='Yes':
		patient2_list[6]=1
	else:
		patient2_list[6]=patient2_list[6]
	if patient=='Patient 2' and  aginhibitor=='Yes':
		patient2_list[7]=1
	else:
		patient2_list[7]=patient2_list[7]
	if patient=='Patient 2' and  biguanide=='Yes':
		patient2_list[8]=1
	else:
		patient2_list[8]=patient2_list[8]
	if patient=='Patient 2' and  dpp4inhibitor=='Yes':
		patient2_list[9]=1
	else:
		patient2_list[9]=patient2_list[9]
	if patient=='Patient 2' and  dpp4biguanide=='Yes':
		patient2_list[10]=1
	else:
		patient2_list[10]=patient2_list[10]
	if patient=='Patient 2' and  glp1ragonist=='Yes':
		patient2_list[11]=1
	else:
		patient2_list[11]=patient2_list[11]
	if patient=='Patient 2' and  sglt2inhibitor=='Yes':
		patient2_list[13]=1
	else:
		patient2_list[13]=patient2_list[13]
	if patient=='Patient 2' and  sglt2biguanide=='Yes':
		patient2_list[14]=1
	else:
		patient2_list[14]=patient2_list[14]
	if patient=='Patient 2' and  sulfonylurea=='Yes':
		patient2_list[15]=1
	else:
		patient2_list[15]=patient2_list[15]
	if patient=='Patient 2' and  thiazolidinedione=='Yes':
		patient2_list[17]=1
	else:
		patient2_list[17]=patient2_list[17]
	if patient=='Patient 2' and  bmi=='Yes':
		patient2_list[18]=24
	else:
		patient2_list[18]=patient2_list[18]
	if patient=='Patient 2' and  total=='Yes':
		patient2_list[19]=150
	else:
		patient2_list[19]=patient2_list[19]
	if patient=='Patient 2' and  bp=='Yes':
		patient2_list[3]=120
		patient2_list[4]=80
	else:
		patient2_list[3]=patient2_list[3]
		patient2_list[4]=patient2_list[4]
	if patient=='Patient 2' and diet=='Yes':
		patient2_list[43]=1
	else:
		patient2_list[43]=patient2_list[43]
	if patient=='Patient 2' and sleep=='Yes':
		patient2_list[44]=8
	else:
		patient2_list[44]=patient2_list[44]


	if patient=='Patient 2':
		a1cpredict2=model.predict([patient2_list])
		a1cpredict2_1=str(a1cpredict2[0])
		st.write("This patient's predicted A1C with interventions is: "+a1cpredict2_1[0:5])
	if patient=='Patient 2':
		if a1cpredict2>=11.0:
			st.image(image=a1c11)
		elif a1cpredict2>=10.0:
			st.image(image=a1c10)
		elif a1cpredict2>=9.0:
			st.image(image=a1c9)
		elif a1cpredict2>=8.0:
			st.image(image=a1c8)
		elif a1cpredict2>=7.0:
			st.image(image=a1c7)
		elif a1cpredict2>=6.0:
			st.image(image=a1c6)
		else:
			st.image(image=a1c5)

	if patient=='Patient 3' and  insulin=='Yes':
		patient3_list[6]=1
	else:
		patient3_list[6]=patient3_list[6]
	if patient=='Patient 3' and  aginhibitor=='Yes':
		patient3_list[7]=1
	else:
		patient3_list[7]=patient3_list[7]
	if patient=='Patient 3' and  biguanide=='Yes':
		patient3_list[8]=1
	else:
		patient3_list[8]=patient3_list[8]
	if patient=='Patient 3' and  dpp4inhibitor=='Yes':
		patient3_list[9]=1
	else:
		patient3_list[9]=patient3_list[9]
	if patient=='Patient 3' and  dpp4biguanide=='Yes':
		patient3_list[10]=1
	else:
		patient3_list[10]=patient3_list[10]
	if patient=='Patient 3' and  glp1ragonist=='Yes':
		patient3_list[11]=1
	else:
		patient3_list[11]=patient3_list[11]
	if patient=='Patient 3' and  sglt2inhibitor=='Yes':
		patient3_list[13]=1
	else:
		patient3_list[13]=patient3_list[13]
	if patient=='Patient 3' and  sglt2biguanide=='Yes':
		patient3_list[14]=1
	else:
		patient3_list[14]=patient3_list[14]
	if patient=='Patient 3' and  sulfonylurea=='Yes':
		patient3_list[15]=1
	else:
		patient3_list[15]=patient3_list[15]
	if patient=='Patient 3' and  thiazolidinedione=='Yes':
		patient3_list[17]=1
	else:
		patient3_list[17]=patient3_list[17]
	if patient=='Patient 3' and  bmi=='Yes':
		patient3_list[18]=24
	else:
		patient3_list[18]=patient3_list[18]
	if patient=='Patient 3' and  total=='Yes':
		patient3_list[19]=150
	else:
		patient3_list[19]=patient3_list[19]
	if patient=='Patient 3' and  bp=='Yes':
		patient3_list[3]=120
		patient3_list[4]=80
	else:
		patient3_list[3]=patient3_list[3]
		patient3_list[4]=patient3_list[4]
	if patient=='Patient 3' and diet=='Yes':
		patient3_list[43]=1
	else:
		patient3_list[43]=patient3_list[43]
	if patient=='Patient 3' and sleep=='Yes':
		patient3_list[44]=8
	else:
		patient3_list[44]=patient3_list[44]

	if patient=='Patient 3':
		a1cpredict3=model.predict([patient3_list])
		a1cpredict3_1=str(a1cpredict3[0])
		st.write("This patient's predicted A1C with interventions is: "+a1cpredict3_1[0:5])
	if patient=='Patient 3':
		if a1cpredict3>=11.0:
			st.image(image=a1c11)
		elif a1cpredict3>=10.0:
			st.image(image=a1c10)
		elif a1cpredict3>=9.0:
			st.image(image=a1c9)
		elif a1cpredict3>=8.0:
			st.image(image=a1c8)
		elif a1cpredict3>=7.0:
			st.image(image=a1c7)
		elif a1cpredict3>=6.0:
			st.image(image=a1c6)
		else:
			st.image(image=a1c5)







import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

titleimage=Image.open('type.png')
redzone=Image.open('redzone.png')
amberzone=Image.open('amberzone.png')
greenzone=Image.open('greenzone.png')

patient_info=pd.read_csv('patient_info.csv',index_col=0)
dataframe=pd.read_csv('dataframe_jun17.csv',index_col=0)

bins=[0,7,10,np.inf]
names=['green','amber','red']
dataframe['last_A1C_stoplight']=pd.cut(dataframe['last_A1C'],bins,labels=names)

X_rf = dataframe.drop(['SEQN','last_A1C','year','last_A1C_stoplight'],1)
y_rf = dataframe['last_A1C_stoplight']
from sklearn.model_selection import train_test_split
X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y_rf, test_size=.2,random_state=0)

from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(class_weight='balanced',bootstrap=True,max_depth=20,max_features='auto',min_samples_leaf=2,min_samples_split=10,n_estimators=100,random_state=0)
clf_rf.fit(X_rf_train, y_rf_train)

patient1_list=X_rf_test.iloc[0,:].values.tolist()
patient2_list=X_rf_test.iloc[1,:].values.tolist()
patient3_list=X_rf_test.iloc[2,:].values.tolist()
patient4_list=X_rf_test.iloc[6,:].values.tolist()
patient5_list=X_rf_test.iloc[7,:].values.tolist()

patient1_a1c=patient_info.loc[patient_info['Patient ID']==81681,'Last A1C (%)'].item()
patient2_a1c=patient_info.loc[patient_info['Patient ID']==77001,'Last A1C (%)'].item()
patient3_a1c=patient_info.loc[patient_info['Patient ID']==96561,'Last A1C (%)'].item()
patient4_a1c=patient_info.loc[patient_info['Patient ID']==96287,'Last A1C (%)'].item()
patient5_a1c=patient_info.loc[patient_info['Patient ID']==78665,'Last A1C (%)'].item()

st.image(image=titleimage)
st.title("Helping physicians provide personalized treatment for patients with type 2 diabetes")
patient=st.selectbox('Select patient EMR data',('Patient 1','Patient 2','Patient 3','Patient 4','Patient 5'))

if patient=='Patient 1':
	st.write(patient_info[patient_info['Patient ID']==81681])
	st.write("This patient's A1C is currently: "+str(patient1_a1c))
if patient=='Patient 1':
	if patient1_a1c>=10.0:
		st.image(image=redzone)
	elif patient1_a1c>=7.0:
		st.image(image=amberzone)
	else:
		st.image(image=greenzone)

if patient=='Patient 2':
	st.write(patient_info[patient_info['Patient ID']==77001])
	st.write("This patient's A1C is currently: "+str(patient2_a1c))
if patient=='Patient 2':
	if patient2_a1c>=10.0:
		st.image(image=redzone)
	elif patient2_a1c>=7.0:
		st.image(image=amberzone)
	else:
		st.image(image=greenzone)

if patient=='Patient 3':
	st.write(patient_info[patient_info['Patient ID']==96561])
	st.write("This patient's A1C is currently: "+str(patient3_a1c))
if patient=='Patient 3':
	if patient3_a1c>=10.0:
		st.image(image=redzone)
	elif patient3_a1c>=7.0:
		st.image(image=amberzone)
	else:
                st.image(image=greenzone)

if patient=='Patient 4':
	st.write(patient_info[patient_info['Patient ID']==96287])
	st.write("This patient's A1C is currently: "+str(patient4_a1c))
if patient=='Patient 4':
	if patient4_a1c>=10.0:
		st.image(image=redzone)
	elif patient4_a1c>=7.0:
		st.image(image=amberzone)
	else:
		st.image(image=greenzone)
if patient=='Patient 5':
	st.write(patient_info[patient_info['Patient ID']==78665])
	st.write("This patient's A1C is currently: "+str(patient5_a1c))
if patient=='Patient 5':
	if patient5_a1c>=10.0:
		st.image(image=redzone)
	elif patient5_a1c>=7.0:
		st.image(image=amberzone)
	else:
		st.image(image=greenzone)


st.sidebar.markdown('Choose possible interventions')
insulin=st.sidebar.selectbox("Add insulin",("No","Yes"))
aginhibitor=st.sidebar.selectbox("Add alpha-glucosidase inhibitor",("No","Yes"))
biguanide=st.sidebar.selectbox("Add biguanide",("No","Yes"))
daagonist=st.sidebar.selectbox("Add dopamine receptor agonist",("No","Yes"))
dpp4inhibitor=st.sidebar.selectbox("Add DPP-4 inhibitor",("No","Yes"))
dpp4biguanide=st.sidebar.selectbox("Add DPP-4 inhibitor;biguanide combo",("No","Yes"))
glp1ragonist=st.sidebar.selectbox("Add GLP-1R agonist",("No","Yes"))
sglt2inhibitor=st.sidebar.selectbox("Add SGLT2 inhibitor",("No","Yes"))
sglt2biguanide=st.sidebar.selectbox("Add SGLT2 inhibitor;biguanide combo",("No","Yes"))
sulfonylurea=st.sidebar.selectbox("Add sufonylurea",("No","Yes"))
thiazolidinedione=st.sidebar.selectbox("Add thiazolidinedione",("No","Yes"))
bmi=st.sidebar.selectbox("Bring BMI to healthy range (24 kg/m2)",("No","Yes"))
diet=st.sidebar.selectbox("Change diet to 'excellent' (5/5 poor to excellent)",("No","Yes"))
hdl=st.sidebar.selectbox("Bring HDL cholesterol to desirable range (60 mg/dl)",("No","Yes"))
bp=st.sidebar.selectbox("Bring blood pressure to desirable range (120/80 mmHg)",("No","Yes"))

if st.button("See how your chosen interventions are predicted to affect this patient's A1C"):
	if patient=='Patient 1' and  insulin=='Yes':
		patient1_list[32]=1
	else:
		patient1_list[32]=patient1_list[32]
	if patient=='Patient 1' and  aginhibitor=='Yes':
		patient1_list[33]=1
	else:
		patient1_list[33]=patient1_list[33]
	if patient=='Patient 1' and  biguanide=='Yes':
		patient1_list[34]=1
	else:
		patient1_list[34]=patient1_list[34]
	if patient=='Patient 1' and  daagonist=='Yes':
		patient1_list[35]=1
	else:
		patient1_list[35]=patient1_list[35]
	if patient=='Patient 1' and  dpp4inhibitor=='Yes':
		patient1_list[36]=1
	else:
		patient1_list[36]=patient1_list[36]
	if patient=='Patient 1' and  dpp4biguanide=='Yes':
		patient1_list[37]=1
	else:
		patient1_list[37]=patient1_list[37]
	if patient=='Patient 1' and  glp1ragonist=='Yes':
		patient1_list[38]=1
	else:
		patient1_list[38]=patient1_list[38]
	if patient=='Patient 1' and  sglt2inhibitor=='Yes':
		patient1_list[39]=1
	else:
		patient1_list[39]=patient1_list[39]
	if patient=='Patient 1' and  sglt2biguanide=='Yes':
		patient1_list[40]=1
	else:
		patient1_list[40]=patient1_list[40]
	if patient=='Patient 1' and  sulfonylurea=='Yes':
		patient1_list[41]=1
	else:
		patient1_list[41]=patient1_list[41]
	if patient=='Patient 1' and  thiazolidinedione=='Yes':
		patient1_list[42]=1
	else:
		patient1_list[42]=patient1_list[42]
	if patient=='Patient 1' and  bmi=='Yes':
		patient1_list[4]=24
	else:
		patient1_list[4]=patient1_list[4]
	if patient=='Patient 1' and  hdl=='Yes':
		patient1_list[5]=60
	else:
		patient1_list[5]=patient1_list[5]
	if patient=='Patient 1' and  bp=='Yes':
		patient1_list[1]=120
		patient1_list[2]=80
	else:
		patient1_list[1]=patient1_list[1]
		patient1_list[2]=patient1_list[2]
	if patient=='Patient 1':
		a1cpredict1=clf_rf.predict([patient1_list])
		a1cpredict1_1=a1cpredict1[0]
		st.write("This patient's predicted A1C with interventions is: "+str(a1cpredict1_1))
	if patient=='Patient 1':
		if a1cpredict1_1=='red':
			st.image(image=redzone)
		elif a1cpredict1_1=='amber':
			st.image(image=amberzone)
		else:
			st.image(image=greenzone)

	if patient=='Patient 2' and  insulin=='Yes':
		patient2_list[32]=1
	else:
		patient2_list[32]=patient2_list[32]
	if patient=='Patient 2' and  aginhibitor=='Yes':
		patient2_list[33]=1
	else:
		patient2_list[33]=patient2_list[33]
	if patient=='Patient 2' and  biguanide=='Yes':
		patient2_list[34]=1
	else:
		patient2_list[34]=patient2_list[34]
	if patient=='Patient 2' and  daagonist=='Yes':
		patient2_list[35]=1
	else:
		patient2_list[35]=patient2_list[35]
	if patient=='Patient 2' and  dpp4inhibitor=='Yes':
		patient2_list[36]=1
	else:
		patient2_list[36]=patient2_list[36]
	if patient=='Patient 2' and  dpp4biguanide=='Yes':
		patient2_list[37]=1
	else:
		patient2_list[37]=patient2_list[37]
	if patient=='Patient 2' and  glp1ragonist=='Yes':
		patient2_list[38]=1
	else:
		patient2_list[38]=patient2_list[38]
	if patient=='Patient 2' and  sglt2inhibitor=='Yes':
		patient2_list[39]=1
	else:
		patient2_list[39]=patient2_list[39]
	if patient=='Patient 2' and  sglt2biguanide=='Yes':
		patient2_list[40]=1
	else:
		patient2_list[40]=patient2_list[40]
	if patient=='Patient 2' and  sulfonylurea=='Yes':
		patient2_list[41]=1
	else:
		patient2_list[41]=patient2_list[41]
	if patient=='Patient 2' and  thiazolidinedione=='Yes':
		patient2_list[42]=1
	else:
		patient2_list[42]=patient2_list[42]
	if patient=='Patient 2' and  bmi=='Yes':
		patient2_list[4]=24
	else:
		patient2_list[4]=patient2_list[4]
	if patient=='Patient 2' and  hdl=='Yes':
		patient2_list[5]=60
	else:
		patient2_list[5]=patient2_list[5]
	if patient=='Patient 2' and  bp=='Yes':
		patient2_list[1]=120
		patient2_list[2]=80
	else:
		patient2_list[1]=patient2_list[1]
		patient2_list[2]=patient2_list[2]
	if patient=='Patient 2':
		a1cpredict2=clf_rf.predict([patient2_list])
		a1cpredict2_1=a1cpredict2[0]
		st.write("This patient's predicted A1C with interventions is: "+str(a1cpredict2_1))
	if patient=='Patient 2':
		if a1cpredict2_1=='red':
			st.image(image=redzone)
		elif a1cpredict2_1=='amber':
			st.image(image=amberzone)
		else:
			st.image(image=greenzone)


	if patient=='Patient 3' and  insulin=='Yes':
		patient3_list[32]=1
	else:
		patient3_list[32]=patient3_list[32]
	if patient=='Patient 3' and  aginhibitor=='Yes':
		patient3_list[33]=1
	else:
		patient3_list[33]=patient3_list[33]
	if patient=='Patient 3' and  biguanide=='Yes':
		patient3_list[34]=1
	else:
		patient3_list[34]=patient3_list[34]
	if patient=='Patient 3' and  daagonist=='Yes':
		patient3_list[35]=1
	else:
		patient3_list[35]=patient3_list[35]
	if patient=='Patient 3' and  dpp4inhibitor=='Yes':
		patient3_list[36]=1
	else:
		patient3_list[36]=patient3_list[36]
	if patient=='Patient 3' and  dpp4biguanide=='Yes':
		patient3_list[37]=1
	else:
		patient3_list[37]=patient3_list[37]
	if patient=='Patient 3' and  glp1ragonist=='Yes':
		patient3_list[38]=1
	else:
		patient3_list[38]=patient3_list[38]
	if patient=='Patient 3' and  sglt2inhibitor=='Yes':
		patient3_list[39]=1
	else:
		patient3_list[39]=patient3_list[39]
	if patient=='Patient 3' and  sglt2biguanide=='Yes':
		patient3_list[40]=1
	else:
		patient3_list[40]=patient3_list[40]
	if patient=='Patient 3' and  sulfonylurea=='Yes':
		patient3_list[41]=1
	else:
		patient3_list[41]=patient3_list[41]
	if patient=='Patient 3' and  thiazolidinedione=='Yes':
		patient3_list[42]=1
	else:
		patient3_list[42]=patient3_list[42]
	if patient=='Patient 3' and  bmi=='Yes':
		patient3_list[4]=24
	else:
		patient3_list[4]=patient3_list[4]
	if patient=='Patient 3' and  hdl=='Yes':
		patient3_list[5]=60
	else:
		patient3_list[5]=patient3_list[5]
	if patient=='Patient 3' and  bp=='Yes':
		patient3_list[1]=120
		patient3_list[2]=80
	else:
		patient3_list[1]=patient3_list[1]
		patient3_list[2]=patient3_list[2]
	if patient=='Patient 3':
		a1cpredict3=clf_rf.predict([patient3_list])
		a1cpredict3_1=a1cpredict3[0]
		st.write("This patient's predicted A1C with interventions is: "+str(a1cpredict3_1))
	if patient=='Patient 3':
		if a1cpredict3_1=='red':
			st.image(image=redzone)
		elif a1cpredict3_1=='amber':
			st.image(image=amberzone)
		else:
			st.image(image=greenzone)


	if patient=='Patient 4' and  insulin=='Yes':
		patient4_list[32]=1
	else:
		patient4_list[32]=patient4_list[32]
	if patient=='Patient 4' and  aginhibitor=='Yes':
		patient4_list[33]=1
	else:
		patient4_list[33]=patient4_list[33]
	if patient=='Patient 4' and  biguanide=='Yes':
		patient4_list[34]=1
	else:
		patient4_list[34]=patient4_list[34]
	if patient=='Patient 4' and  daagonist=='Yes':
		patient4_list[35]=1
	else:
		patient4_list[35]=patient4_list[35]
	if patient=='Patient 4' and  dpp4inhibitor=='Yes':
		patient4_list[36]=1
	else:
		patient4_list[36]=patient4_list[36]
	if patient=='Patient 4' and  dpp4biguanide=='Yes':
		patient4_list[37]=1
	else:
		patient4_list[37]=patient4_list[37]
	if patient=='Patient 4' and  glp1ragonist=='Yes':
		patient4_list[38]=1
	else:
		patient4_list[38]=patient4_list[38]
	if patient=='Patient 4' and  sglt2inhibitor=='Yes':
		patient4_list[39]=1
	else:
		patient4_list[39]=patient4_list[39]
	if patient=='Patient 4' and  sglt2biguanide=='Yes':
		patient4_list[40]=1
	else:
		patient4_list[40]=patient4_list[40]
	if patient=='Patient 4' and  sulfonylurea=='Yes':
		patient4_list[41]=1
	else:
		patient4_list[41]=patient4_list[41]
	if patient=='Patient 4' and  thiazolidinedione=='Yes':
		patient4_list[42]=1
	else:
		patient4_list[42]=patient4_list[42]
	if patient=='Patient 4' and  bmi=='Yes':
		patient4_list[4]=24
	else:
		patient4_list[4]=patient4_list[4]
	if patient=='Patient 4' and  hdl=='Yes':
		patient4_list[5]=60
	else:
		patient4_list[5]=patient4_list[5]
	if patient=='Patient 4' and  bp=='Yes':
		patient4_list[1]=120
		patient4_list[2]=80
	else:
		patient4_list[1]=patient4_list[1]
		patient4_list[2]=patient4_list[2]
	if patient=='Patient 4':
		a1cpredict4=clf_rf.predict([patient4_list])
		a1cpredict4_1=a1cpredict4[0]
		st.write("This patient's predicted A1C with interventions is: "+str(a1cpredict4_1))
	if patient=='Patient 4':
		if a1cpredict4_1=='red':
			st.image(image=redzone)
		elif a1cpredict4_1=='amber':
			st.image(image=amberzone)
		else:
			st.image(image=greenzone)


	if patient=='Patient 5' and  insulin=='Yes':
		patient5_list[32]=1
	else:
		patient5_list[32]=patient5_list[32]
	if patient=='Patient 5' and  aginhibitor=='Yes':
		patient5_list[33]=1
	else:
		patient5_list[33]=patient5_list[33]
	if patient=='Patient 5' and  biguanide=='Yes':
		patient5_list[34]=1
	else:
		patient5_list[34]=patient5_list[34]
	if patient=='Patient 5' and  daagonist=='Yes':
		patient5_list[35]=1
	else:
		patient5_list[35]=patient5_list[35]
	if patient=='Patient 5' and  dpp4inhibitor=='Yes':
		patient5_list[36]=1
	if patient=='Patient 5' and  dpp4biguanide=='Yes':
		patient5_list[37]=1
	else:
		patient5_list[37]=patient5_list[37]
	if patient=='Patient 5' and  glp1ragonist=='Yes':
		patient5_list[38]=1
	else:
		patient5_list[38]=patient5_list[38]
	if patient=='Patient 5' and  sglt2inhibitor=='Yes':
		patient5_list[39]=1
	else:
		patient5_list[39]=patient5_list[39]
	if patient=='Patient 5' and  sglt2biguanide=='Yes':
		patient5_list[40]=1
	else:
		patient5_list[40]=patient5_list[40]
	if patient=='Patient 5' and  sulfonylurea=='Yes':
		patient5_list[41]=1
	else:
		patient5_list[41]=patient5_list[41]
	if patient=='Patient 5' and  thiazolidinedione=='Yes':
		patient5_list[42]=1
	else:
		patient5_list[42]=patient5_list[42]
	if patient=='Patient 5' and  bmi=='Yes':
		patient5_list[4]=24
	else:
		patient5_list[4]=patient5_list[4]
	if patient=='Patient 5' and  hdl=='Yes':
		patient5_list[5]=60
	else:
		patient5_list[5]=patient5_list[5]
	if patient=='Patient 5' and  bp=='Yes':
		patient5_list[1]=120
		patient5_list[2]=80
	else:
		patient5_list[1]=patient5_list[1]
		patient5_list[2]=patient5_list[2]
	if patient=='Patient 5':
		a1cpredict5=clf_rf.predict([patient5_list])
		a1cpredict5_1=a1cpredict5[0]
		st.write("This patient's predicted A1C with interventions is: "+str(a1cpredict5_1))
	if patient=='Patient 5':
		if a1cpredict5_1=='red':
			st.image(image=redzone)
		elif a1cpredict5_1=='amber':
			st.image(image=amberzone)
		else:
			st.image(image=greenzone)







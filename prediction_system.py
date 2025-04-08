# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 20:30:26 2025

@author: ghosh
"""


from PIL import Image
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np



#landing page 

if 'landing_page' not in st.session_state:
    st.session_state.landing_page = True

if st.session_state.landing_page:
    with st.container():
        
        # Add a logo above the heading
        image = Image.open("logo.png")  # Open image manually
        st.image(image, width=250)
        
        st.markdown('<div class="landing-container">', unsafe_allow_html=True)
        st.markdown("<h2>University Of Engineering and Management Hospital, Jaipur</h2>", unsafe_allow_html=True)
        st.markdown("<p>__________________________________________________________________________</p>", unsafe_allow_html=True)
        st.markdown("<h1>H E A L I T I C S</h1>", unsafe_allow_html=True)
        st.markdown("<h4>-  Sensing illness before it strikes</h4>", unsafe_allow_html=True)
        st.markdown("<p><br><br></p>", unsafe_allow_html=True)
        st.markdown(
            "<p>Our advanced AI platform assists in Primary diagnosing only following conditions -"
            "Diabetes, Heart Disease, Parkinson's Disease, Chronic Kidney Stone, Liver Serosis, Darmatology etc.<br><br><br>"
            "<strong>Precautions:</strong> This tool is for informational purposes only and is not a substitute for professional medical advice.<br><br></p>"
            "<strong>Disclaimer:</strong> Results can be Wrong so, Consult with a specialized Doctor for better treatment.</p>",
            unsafe_allow_html=True
        )
        

        if st.button("Enter The Clinic"):
            st.session_state.landing_page = False
            st.rerun()  # Use st.rerun() if you're on Streamlit >=1.18
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()
    
        


# loading the saved models

diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

kidney_model = pickle.load(open('kidney_model.sav', 'rb'))

darmatology_model = pickle.load(open('Darmatology_model.sav', 'rb'))

asthma_model = pickle.load(open('Asthma_model.sav', 'rb'))

anemia_model = pickle.load(open('Anemia_model.sav', 'rb'))

alzheimer_model = pickle.load(open('alzheimer_model.sav', 'rb'))


# sidebar for navigation
with st.sidebar:
    st.image('logo1.png', width=250)
    selected = option_menu('Welcome to Virtual Doctor''s'' Clinic',
                           ['Diabetes Analysis',
                            'Heart Disease Analysis',
                            'Parkinsons Disease Analysis',
                            'ECG Analysis',
                            'Chronic Kidney Disease Analysis',
                            'Thyroid Analysis',
                            'Darmatology Analysis',
                            'Asthma Analysis',
                            'Anemia Analysis',
                            'Alzheimer Analysis',
                            '📝 User Guidance 📝',
                            'ℹ️ About Us ℹ️'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person', 'activity', 'lungs', 'prescription2', 'radioactive',
                                  'lungs-fill', 'file-medical-fill', 'person-arms-up', 'person-gear', 'file-person-fill'],
                           default_index=0)
    
    st.markdown("---")  # Horizontal line
    st.subheader("🧑‍⚕️ Contact Info : ")
    st.write("Email 📧 : sanchayan.ghosh.2022@uem.edu.in")
    st.write("Phone 📞 : +91-8101111950")
    st.write("Linked in 🌐 : www.linkedin.com/in/sanchayan-ghosh-0b735024a")
    st.write("Youtube ▶️ : www.youtube.com/channel/UCi86CDhGg3qDi2M6wy5OUpg")
    

if selected == 'Diabetes Analysis':

    # page title
    st.image("Diabetes.jpg", width=400)
    # page title
    st.title('Virtual Diabetes Analyzer')
    
    st.text("Precaution: Priliminary test ues only, please consult with Specialist")

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies (0 - 17 )')

    with col2:
        Glucose = st.text_input('Glucose Level (0 - 199 )')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value ( 0 - 122 )')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value ( 0 - 99 )')

    with col2:
        Insulin = st.text_input('Insulin Level ( 0 - 848 )')

    with col3:
        BMI = st.text_input('BMI value ( 0 - 67 )')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value ( 0.078 - 2.42 )')

    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'Sorry! Seems like you are diabetic, consult with a Specialist soon.'
        else:
            diab_diagnosis = 'Good News! You are not Diabetic.'

    st.success(diab_diagnosis)
    


# Heart Disease Prediction Page
if selected == 'Heart Disease Analysis':
    st.image("Heart.jpg", width=400)
    # page title
    st.title('Virtual Heart Disease Analyzer')
    
    st.text("Precaution: Priliminary test ues, please consult with Specialist")
    
    col1, col2, col3 = st.columns(3)

    with col1:
       age = st.text_input('Age')

    with col2:
       sex = st.text_input('Gender (0-Female, 1-Male)')

    with col3:
       cp = st.text_input('Chest Pain types ( 0 - 3 )')

    with col1:
       trestbps = st.text_input('Resting Blood Pressure ( 64 - 150 )')

    with col2:
       chol = st.text_input('Cholestoral [ mg/dl ] ( 10 - 200 )')

    with col3:
       fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl for abnormality')

    with col1:
       restecg = st.text_input('Resting Electrocardiographic results ( 0 - 2 )')

    with col2:
       thalach = st.text_input('Maximum Heart Rate achieved ( 70 - 202 )')

    with col3:
       exang = st.text_input('Exercise Induced Angina [Pain] (0 for no, 1 for yes)')

    with col1:
       oldpeak = st.text_input('ST depression induced by exercise (0.0 - 6.2)')

    with col2:
       slope = st.text_input('Slope of the peak exercise ST segment (0-2)')

    with col3:
       ca = st.text_input('Major vessels colored by flourosopy (0-4)')

    with col1:
       thal = st.text_input('Thalassemia: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

       user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

       user_input = [float(x) for x in user_input]

       heart_prediction = heart_disease_model.predict([user_input])

       if heart_prediction[0] == 1:
           heart_diagnosis = 'Sorry! I think you have heart disease'
       else:
           heart_diagnosis = 'Good News! You do not have any Cardiac disease'

    st.success(heart_diagnosis)
    
# Parkinsons Disease Prediction Page
if selected == 'Parkinsons Disease Analysis':
    st.image("Parkinsons.jpg", width=300)
    # page title
    st.title('Virtual Parkinsons Disease Analyzer')
    
    st.text("Precaution: Priliminary test ues, please consult with Specialist")
    
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz) [80.0-253]')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz) [70.0-589.0]')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz) [65.0-143.0]')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%) [0.001-0.035]')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs) [0.00001-0.00026]')

    with col1:
        RAP = st.text_input('MDVP:RAP [0.00060-0.015]')

    with col2:
        PPQ = st.text_input('MDVP:PPQ [0.0009-0.022]')

    with col3:
        DDP = st.text_input('Jitter:DDP [0.001-0.085]')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer [0.005-1.35]')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB) [0.005-0.093]')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3 [0.005-1.5]')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5 [[0.005-0.95]')

    with col3:
        APQ = st.text_input('MDVP:APQ [0.00092-.03]')

    with col4:
        DDA = st.text_input('Shimmer:DDA [0.002-0.085]')

    with col5:
        NHR = st.text_input('NHR [0.001-0.085]')

    with col1:
        HNR = st.text_input('HNR [10.0-35.0]')

    with col2:
        RPDE = st.text_input('RPDE [0.026-0.667]')

    with col3:
        DFA = st.text_input('DFA [1.75-3.7]')

    with col4:
        spread1 = st.text_input('spread1 [0.2-0.7]')

    with col5:
        spread2 = st.text_input('spread2 [0.08-3.7]')

    with col1:
        D2 = st.text_input('D2 [1.7-3.7]')

    with col2:
        PPE = st.text_input('PPE [0.08-3.5]')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "Sorry! I think You have Parkinson's disease"
        else:
            parkinsons_diagnosis = "Good News! You do not have Parkinson's disease"

    st.success(parkinsons_diagnosis)
    
# Kidney Prediction Page
if selected == 'Chronic Kidney Disease Analysis':
    st.image("Kidney.jpg", width=400)
    # page title
    st.title('Virtual Kidney Problem Analyzer')
    
    st.text("Precaution: Priliminary test ues, please consult with Specialist")
    st.markdown("⚠️ Sorry! This service is under maintenance and temporarily out of service.⚠️")

    



# thyroid    
thyroid_model = pickle.load(open('thyroid.sav', 'rb'))

if selected == 'Thyroid Analysis':

    st.image("Thyroid.jpg", width=400)

    st.title('Virtual Thyroid Analyzer')
    
    st.text("Precaution: Preliminary test use only, please consult with a Specialist")

    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.text_input('Age')

    with col2:
        TSH = st.text_input('TSH Level [0.005-530]')

    with col3:
        T3 = st.text_input('T3 Level [0.05-18]')

    with col1:
        TT4 = st.text_input('TT4 Level[2-600]')

    with col2:
        T4U = st.text_input('T4U Level [0.16-2.33]')

    with col3:
        FTI = st.text_input('FTI Value [1.4-800]')

    with col1:
        Sex = st.selectbox('Gender', ('Female', 'Male'))  # Female=0, Male=1


    thyroid_diagnosis = ''

    if st.button('Thyroid Test Result'):
        
        sex_value = 0 if Sex == 'F' else 1  # Encoding Sex Column
        
        user_input = [Age, TSH, T3, TT4, T4U, FTI, sex_value]

        user_input = [float(x) for x in user_input]

        # Check number of features
        if len(user_input) == 7:
            # Append dummy value if model trained on 8 features (like bias or other feature)
            user_input.append(0)   # If required (confirm from your dataset)
        
        thyroid_prediction = thyroid_model.predict([user_input])

        if thyroid_prediction[0] == 1:
            thyroid_diagnosis = 'Warning! Possibility of Thyroid detected, consult with a Specialist.'
        else:
            thyroid_diagnosis = 'Good News! No Thyroid detected.'

    st.success(thyroid_diagnosis)


    
# Darmatology Prediction Page
if selected == 'Darmatology Analysis':
    st.image("Darmatology.jpg", width=400)
    # page title
    st.title('Virtual Dermatology Analyzer')
    
    # Streamlit UI
    st.text("Precaution: Priliminary test ues, please contact Specialist")

    # Input fields
    features = {}
    columns = [
        "erythema", "scaling", "definite_borders", "itching", "koebner_phenomenon", 
        "polygonal_papules", "follicular_papules", "oral_mucosal_involvement", "knee_and_elbow_involvement", "scalp_involvement",
        "family_history", "melanin_incontinence", "eosinophils_in_the_infiltrate", "pnl_infiltrate", "fibrosis_of_the_papillary_dermis", 
        "exocytosis", "acanthosis", "hyperkeratosis", "parakeratosis", "clubbing_of_the_rete_ridges",
        "elongation_of_the_rete_ridges", "thinning_of_the_suprapapillary_epidermis", "spongiform_pustule", "munro_microabcess", "focal_hypergranulosis",
        "disappearance_of_the_granular_layer", "vacuolisation_and_damage_of_basal_layer", "spongiosis", "saw-tooth_appearance_of_retes", "follicular_horn_plug",
        "perifollicular_parakeratosis", "inflammatory_monoluclear_inflitrate", "band-like_infiltrate", "age"
        ]

    cols = st.columns(5)
    for i, col in enumerate(columns):
        with cols[i % 5]:  # Distribute inputs across columns
            features[col] = st.number_input(col, min_value=0.0, max_value=100.0, value=0.0, step=1.0)

    # Predict button
    if st.button("Predict Dermatology Condition"):
        user_input = np.array([list(features.values())]).astype(float)
        prediction = darmatology_model.predict(user_input)
        st.success(f"Predicted Condition Class: {int(prediction[0])}, 0 for negative and 1 for positive.")
    
    

# Asthma Prediction Page
if selected == 'Asthma Analysis':
    st.image("asthma.jpg", width=400)
    
    # Page title
    st.title('Virtual Asthma Analyzer')
    st.text("Precaution: Priliminary test ues, please consult with a Specialist")
    
    # Getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Age = st.text_input('Age')
        BMI = st.text_input('BMI Value [15.0-40.0]')
        DietQuality = st.text_input('Diet Quality [0.0-10.0]')
        PollenExposure = st.text_input('Pollen Exposure (No:0 Yes:1)')
        PetAllergy = st.text_input('Pet Allergy (No:0 Yes:1)')
        LungFunctionFEV1 = st.text_input('FEV1 (Liters) [1.0-4.0]')
        HayFever = st.text_input('Fever (No:0 Yes:1)')
        ChestTightness = st.text_input('Chest Tightness (No:0 Yes:1)')
    
    with col2:
        Gender = st.text_input('Gender (Male:1 Female:0)')
        Smoking = st.text_input('Smoking Status (No:0 Yes:1)')
        SleepQuality = st.text_input('Sleep Quality [4.0-10.0]')
        DustExposure = st.text_input('Dust Exposure (No:0 Yes:1)')
        FamilyHistoryAsthma = st.text_input('Family History of Asthma (No:0 Yes:1)')
        LungFunctionFVC = st.text_input('FVC (Liters) [1.0-6.0]')
        GastroesophagealReflux = st.text_input('Gastroesophageal Reflux (No:0 Yes:1)')
        Coughing = st.text_input('Coughing (No:0 Yes:1)')
    
    with col3:
        PhysicalActivity = st.text_input('Weekly Physical Activity (Hours) [0.001-10.0]')
        PollutionExposure = st.text_input('Pollution Exposure [0.0-10.0]')
        HistoryOfAllergies = st.text_input('History of Allergies (No:0 Yes:1)')
        Eczema = st.text_input('Eczema (No:0 Yes:1)')
        Wheezing = st.text_input('Wheezing (No:0 Yes:1)')
        ShortnessOfBreath = st.text_input('Shortness of Breath (No:0 Yes:1)')
        NighttimeSymptoms = st.text_input('Nighttime Symptoms (No:0 Yes:1)')
        ExerciseInduced = st.text_input('Exercise-Induced Symptoms (No:0 Yes:1)')
    
    # Code for Prediction
    ast_diag = ''
    
    if st.button('Asthma Test'):
        user_input = [Age, Gender, BMI, Smoking, PhysicalActivity, DietQuality, SleepQuality,
                      PollutionExposure, PollenExposure, DustExposure, PetAllergy, FamilyHistoryAsthma,
                      HistoryOfAllergies, Eczema, HayFever, GastroesophagealReflux, LungFunctionFEV1,
                      LungFunctionFVC, Wheezing, ShortnessOfBreath, ChestTightness, Coughing,
                      NighttimeSymptoms, ExerciseInduced]
        
        #user_input = [float(x) for x in user_input]
        user_input = [float(x) if x.strip() != '' else 0.0 for x in user_input]

        
        ast_pred = asthma_model.predict([user_input])
        
        if ast_pred[0] == 1:
            ast_diag = 'Sorry! You show signs of Asthma. We recommend consulting a Doctor soon.'
        else:
            ast_diag = 'Good News! You do not have Asthma.'
            
        ast_pred = asthma_model.predict([user_input])
        st.write("Raw Prediction Output:", ast_pred)
        ast_prob = asthma_model.predict_proba([user_input])
        st.write("Prediction Probabilities:", ast_prob)
    
    st.success(ast_diag)
    

# Anemia analysis    
# Load the saved KNN model and scaler
with open("Anemia_knn_model.sav", "rb") as model_file:
    anemia_model = pickle.load(model_file)

with open("anemia_scaler.sav", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Anemia Analysis UI
if selected == 'Anemia Analysis':
    st.image("anemia.jpg", width=400)
    st.title('Virtual Anemia Analyzer')
    
    st.text("Precaution: Priliminary test ues, please consult a Specialist.")

    # User inputs
    Gender = st.number_input('Gender (Male:1 Female:0)')
    Hemoglobin = st.number_input('Hemoglobin Level (g/dL) [5.0-17.0]')
    MCH = st.number_input('Mean Corpuscular Hemoglobin (pg) [15.0-30.0]')
    MCHC = st.number_input('Mean Corpuscular Hemoglobin Concentration (g/dL) [20.0-33.0]')
    MCV = st.number_input('Mean Corpuscular Volume (fL) [65.0-102.0]')

    # Prediction
    anemia_diag = ''
    if st.button('Anemia Test Result'):
        input_data = np.array([[Gender, Hemoglobin, MCH, MCHC, MCV]])
        input_scaled = scaler.transform(input_data)
        anemia_pred = anemia_model.predict(input_scaled)

        if anemia_pred[0] == 1:
            anemia_diag = 'Sorry! You show signs of Anemia. We recommend consulting a Doctor soon.'
        else:
            anemia_diag = 'Good News! You do not have Anemia!'
    
    st.success(anemia_diag)

# alzhaimer
# Load saved model and scaler
with open("naive_bayes_alzheimers_model.sav", "rb") as model_file:
    alzheimer_model = pickle.load(model_file)

with open("scaler.sav", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Alzheimer Analysis Interface
if selected == 'Alzheimer Analysis':
    st.image("alzheimers.jpg", width=300)
    st.title('Virtual Alzheimers Analyzer')
    st.text("Precaution: Priliminary test ues, please consult a Specialist.")

    # User Inputs
    Age = st.number_input('Age', min_value=0, value=0)
    Gender = st.number_input('Gender (Male:1 Female:0)')
    BMI = st.number_input('BMI Value (Range:15 to 40)')
    Smoking = st.number_input('Smoking Status (No:0 Yes:1)')
    AlcoholConsumption = st.number_input('Weekly Alcohol Consumption (Range:0 to 20 hours)')
    PhysicalActivity = st.number_input('Weekly Physical Activity (Range:0 to 10 hours)')
    DietQuality = st.number_input('Diet Quality (Range:0 to 10)')
    SleepQuality = st.number_input('Sleep Quality (Range:4 to 10)')
    FamilyHistoryAlzheimers = st.number_input('Family history of Alzheimers Disease (No:0 Yes:1)')
    CardiovascularDisease = st.number_input('Cardiovascular Disease (No:0 Yes:1)')
    Diabetes = st.number_input('Diabetes (No:0 Yes:1)')
    Depression = st.number_input('Depression (No:0 Yes:1)')
    HeadInjury = st.number_input('Head Injury (No:0 Yes:1)')
    Hypertension = st.number_input('Hypertension (No:0 Yes:1)')
    SystolicBP = st.number_input('Systolic Blood Pressure (Range:90 to 180)')
    DiastolicBP = st.number_input('Diastolic Blood Pressure (Range:60 to 120)')
    CholesterolTotal = st.number_input('Total Cholesterol Levels (Range:150 to 300)')
    MMSE = st.number_input('Mini-Mental State Examination score (Range:0 to 30)')
    FunctionalAssessment = st.number_input('Functional assessment score (Range:0 to 10)')
    MemoryComplaints = st.number_input('Memory Complaints (No:0 Yes:1)')
    BehavioralProblems = st.number_input('Behavioral Problems (No:0 Yes:1)')
    ADL = st.number_input('Activities of Daily Living score (Range:0 to 10)')
    Confusion = st.number_input('Confusion (No:0 Yes:1)')
    Disorientation = st.number_input('Disorientation (No:0 Yes:1)')
    PersonalityChanges = st.number_input('Personality Changes (No:0 Yes:1)')
    DifficultyCompletingTasks = st.number_input('Difficulty Completing Tasks (No:0 Yes:1)')
    Forgetfulness = st.number_input('Forgetfulness (No:0 Yes:1)')

    # Prediction logic
    alz_diag = ''
    if st.button('Alzheimers Disease Test Result'):
        input_data = np.array([[Age, Gender, BMI, Smoking, AlcoholConsumption, PhysicalActivity, DietQuality,
                                SleepQuality, FamilyHistoryAlzheimers, CardiovascularDisease, Diabetes,
                                Depression, HeadInjury, Hypertension, SystolicBP, DiastolicBP,
                                CholesterolTotal, MMSE, FunctionalAssessment, MemoryComplaints,
                                BehavioralProblems, ADL, Confusion, Disorientation, PersonalityChanges,
                                DifficultyCompletingTasks, Forgetfulness]])
        
        input_scaled = scaler.transform(input_data)
        alz_pred = alzheimer_model.predict(input_scaled)

        if alz_pred[0] == 1:
            alz_diag = '⚠️ Sorry! You may have Alzheimer\'s. We recommend visiting a doctor.'
        else:
            alz_diag = '✅ Good News! You likely do not have Alzheimer\'s.'

    st.success(alz_diag)


# Load Random Forest Model
loaded_model = pickle.load(open('ecg_rf_model.sav', 'rb'))

if selected == 'ECG Analysis':

    st.image("ecg.jpg", width=400)
    st.title('Virtual ECG Analyzer - Random Forest Model')

    st.text("Precaution: Preliminary test only, please consult with Specialist")
    st.text("⚠️ Sorry! This service is under maintenance and is temporarily out of service.⚠️")

    




    
    
    
    
# user manual    
if selected == '📝 User Guidance 📝':
    st.title('HEALITICS -  User Manual 🧑‍⚕️‍⚕️🧑‍⚕️')
    st.markdown("______________________________________________________________________________________________________________________________")
    st.image("1.png", width=800)
    st.image("2.png", width=800)
    st.markdown("<p>💬  Enter the red coloured button to enter into the clinic.</p>", unsafe_allow_html=True)
    st.markdown("______________________________________________________________________________________________________________________________")
    st.markdown("<h4>-  Diabetes Test : </h4>", unsafe_allow_html=True)
    st.image("3.png", width=800)
    st.markdown("<p>💬  Row wise entry the following intents to test Diabetes : Number of Pregnancies, Glucose level, Blood Preasure, Skin thickness, Insulin level, Besal Metabolic Rate, Pedegry function, Age of the PAtient.</p>", unsafe_allow_html=True)
    st.markdown("<p>💬  Enter the Test button to know your tese result.</p>", unsafe_allow_html=True)
    st.markdown("______________________________________________________________________________________________________________________________")
    st.markdown("<h4>-  Heart Disease Test : </h4>", unsafe_allow_html=True)
    st.image("4.png", width=800)
    st.markdown("<p>💬  Row wise entry the following intents to test Diabetes : Age, Gender, Chest Pain Type,Resting Blood Preasure, Cholesterol, Fasting Blood Sugar, Resting ECG Rates, Thalachemia, exang, Oldpeak, Slope , etc. of the Patient.</p>", unsafe_allow_html=True)
    st.markdown("<p>💬  Enter the Test button to know your tese result.</p>", unsafe_allow_html=True)
    st.markdown("______________________________________________________________________________________________________________________________")
    st.markdown("<h4>-  Parkinson's Disorder Test : </h4>", unsafe_allow_html=True)
    st.image("5.png", width=800)
    st.markdown("<p>💬 Row wise entry all the intents to test Parkinson's Disorder.</p>", unsafe_allow_html=True)
    st.markdown("<p>💬  Parkinson's Disorder : Parkinson's disease is a movement disorder of the nervous system that worsens over time.</p>", unsafe_allow_html=True)
    st.markdown("<p> Symptoms start slowly. The first symptom may be a barely noticeable tremor in just one hand or sometimes a foot or the jaw. Tremor is common in Parkinson's disease. </p>", unsafe_allow_html=True)
    st.markdown("<p> Although Parkinson's disease can't be cured, medicines may help symptoms get better.</p>", unsafe_allow_html=True)
    st.markdown("<p>💬  Enter the Test button to know your tese result.</p>", unsafe_allow_html=True)
    st.markdown("______________________________________________________________________________________________________________________________")
    st.markdown("<h4>-  Kidney Stone Test : </h4>", unsafe_allow_html=True)
    st.image("6.png", width=800)
    st.markdown("<p>💬 Row wise entry all the intents to test Kidney Stone.</p>", unsafe_allow_html=True)
    st.markdown("<p>💬  Enter the Test button to know your tese result.</p>", unsafe_allow_html=True)
    st.markdown("______________________________________________________________________________________________________________________________")
    st.markdown("<h4>-  Thyroid Test : </h4>", unsafe_allow_html=True)
    st.image("7.png", width=800)
    st.markdown("<p>💬 Row wise entry all the intents to test Liver Serosisr.</p>", unsafe_allow_html=True)
    st.markdown("<p>💬  Enter the 'Predict Liver Disease' button to know your tese result.</p>", unsafe_allow_html=True)
    st.markdown("______________________________________________________________________________________________________________________________")
    st.markdown("<h4>-  Darmatology Test : </h4>", unsafe_allow_html=True)
    st.image("8.png", width=800)
    st.markdown("<p>💬 Row wise entry all the intents to test Darmatology Disorder.</p>", unsafe_allow_html=True)
    st.markdown("<p> Darmatology : It handles outer body disorders including Skin diseases, Skin Cancer, nail, Hair Disorders etc.</p>", unsafe_allow_html=True)
    st.markdown("<p>💬  Enter the Predict button to know your tese result.</p>", unsafe_allow_html=True)
    st.markdown("______________________________________________________________________________________________________________________________")
    st.markdown("<h4>-  Asthma Test : </h4>", unsafe_allow_html=True)
    st.image("9.png", width=800)
    st.markdown("<p>💬 Row wise entry all the intents to test Asthma Disorder.</p>", unsafe_allow_html=True)
    st.markdown("<p>💬  Enter the Test button to know your tese result.</p>", unsafe_allow_html=True)
    st.markdown("______________________________________________________________________________________________________________________________")
    st.markdown("<h4>-  Anemia Test : </h4>", unsafe_allow_html=True)
    st.image("10.png", width=800)
    st.markdown("<p>💬 Row wise entry all the intents to test Anemia.</p>", unsafe_allow_html=True)
    st.markdown("<p>💬  Enter the Test button to know your tese result.</p>", unsafe_allow_html=True)
    st.markdown("______________________________________________________________________________________________________________________________")
    st.markdown("<h4>-  Alzheimers Test : </h4>", unsafe_allow_html=True)
    st.image("11.png", width=800)
    st.markdown("<p>💬 Row wise entry all the intents to test Alzheimer Disorder.</p>", unsafe_allow_html=True)
    st.markdown("<p>💬  Enter the Test button to know your tese result.</p>", unsafe_allow_html=True)
    
    st.markdown("______________________________________________________________________________________________________________________________")
    st.markdown(
        "<p><br><strong>  👩‍⚕️  Precautions:</strong> This tool is for informational purposes only and is not a substitute for professional medical advice.<br><br></p>"
        "<strong>  👩‍⚕️  Disclaimer:</strong>  Priliminary test ues so, Consult with a specialized Doctor for better treatment.</p>",
        unsafe_allow_html=True
    )
    st.markdown("______________________________________________________________________________________________________________________________")
    st.markdown("<h4>--------------------- 🙏 Thank You Visit Again 🙏 --------------------------</h4>", unsafe_allow_html=True)
    
    
if selected == 'ℹ️ About Us ℹ️':
    st.title('HEALITICS -  🧑‍⚕️‍ About Us 🧑‍⚕️‍⚕️')
    st.markdown("______________________________________________________________________________________________________________________________")
    st.markdown("<p>💬   HEALITICS is a health-tech initiative driven by a passion for leveraging Artificial Intelligence and Machine Learning to improve healthcare accessibility and diagnostics. Our mission is to empower individuals with intelligent tools that assist in early disease detection, encouraging timely medical intervention..</p>", unsafe_allow_html=True)
    st.markdown("Developed by a team of tech enthusiasts and healthcare-focused developers, HEALITICS blends innovation with simplicity through its user-friendly web application built using Python and Streamlit.")
    st.markdown("We aim to bridge the gap between technology and healthcare by offering reliable, real-time predictive insights for multiple diseases—ultimately contributing to a healthier, more informed society.")
    st.markdown("🧑‍⚕️‍ I am SANCHAYAN GHOSH, a B.Tech CSE student and a passionate python and AIML developer. I have Developed this project as my semester project and in future I agree to davelope and maintain this web-app.")
    
    
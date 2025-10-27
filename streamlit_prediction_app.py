import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("🩺 Provider Fraud Prediction App")

# Upload CSV Files
st.sidebar.header("Upload CSV Files")
bf_file = st.sidebar.file_uploader("Beneficiary CSV", type="csv")
im_file = st.sidebar.file_uploader("Inpatient CSV", type="csv")
op_file = st.sidebar.file_uploader("Outpatient CSV", type="csv")
provider_file = st.sidebar.file_uploader("Provider CSV", type="csv")

# Feature Engineering Function
def create_provider_features(im_df, op_df, bf_df, training_feature_columns=None):
    #  Concatenate claims
    im_df['ClaimType'] = 'Inpatient'
    op_df['ClaimType'] = 'Outpatient'
    all_claims = pd.concat([im_df, op_df], ignore_index=True)

    # Convert date columns to datetime
    date_cols = ['ClaimStartDt','ClaimEndDt','AdmissionDt','DischargeDt']
    for col in date_cols:
        if col in all_claims.columns:
            all_claims[col] = pd.to_datetime(all_claims[col], errors='coerce')

    # Length of stay for inpatient only
    if 'AdmissionDt' in all_claims.columns and 'DischargeDt' in all_claims.columns:
        all_claims['LengthOfStay'] = (all_claims['DischargeDt'] - all_claims['AdmissionDt']).dt.days

    # Claim duration for all claims
    if 'ClaimStartDt' in all_claims.columns and 'ClaimEndDt' in all_claims.columns:
        all_claims['ClaimDuration'] = (all_claims['ClaimEndDt'] - all_claims['ClaimStartDt']).dt.days

    # Number of diagnosis/procedure codes
    diag_cols = [f'ClmDiagnosisCode_{i}' for i in range(1,11) if f'ClmDiagnosisCode_{i}' in all_claims.columns]
    proc_cols = [f'ClmProcedureCode_{i}' for i in range(1,7) if f'ClmProcedureCode_{i}' in all_claims.columns]
    all_claims['NumDiagCodes'] = all_claims[diag_cols].notna().sum(axis=1)
    all_claims['NumProcCodes'] = all_claims[proc_cols].notna().sum(axis=1)

    # One-hot encode claim type
    all_claims['InpatientFlag'] = (all_claims['ClaimType']=='Inpatient').astype(int)
    all_claims['OutpatientFlag'] = (all_claims['ClaimType']=='Outpatient').astype(int)

    # Physician count
    phys_cols = [c for c in ['AttendingPhysician','OperatingPhysician','OtherPhysician'] if c in all_claims.columns]
    all_claims['PhysicianSet'] = all_claims[phys_cols].apply(lambda x: set(x.dropna()), axis=1)
    all_claims['NumPhysicians'] = all_claims['PhysicianSet'].apply(len)

    # Select relevant claim-level columns
    selected_claims=['BeneID','ClaimID','Provider','AttendingPhysician','OperatingPhysician',
                     'InscClaimAmtReimbursed','DeductibleAmtPaid','DiagnosisGroupCode',
                     'LengthOfStay','ClaimDuration','NumDiagCodes','NumProcCodes',
                     'InpatientFlag','OutpatientFlag','NumPhysicians']
    selected_claims = [c for c in selected_claims if c in all_claims.columns]
    all_claims_selected = all_claims[selected_claims].copy()

    ## Beneficiary features
    if 'DOB' in bf_df.columns:
        bf_df['DOB'] = pd.to_datetime(bf_df['DOB'], errors='coerce')
    if 'DOD' in bf_df.columns:
        bf_df['DOD'] = pd.to_datetime(bf_df['DOD'], errors='coerce')

    bf_df['Age'] = bf_df.apply(lambda row: ((row['DOD'] if pd.notnull(row['DOD']) else pd.Timestamp.today()) - row['DOB']).days // 365 if pd.notnull(row['DOB']) else np.nan, axis=1)

    # Map RenalDiseaseIndicator
    if 'RenalDiseaseIndicator' in bf_df.columns:
        bf_df['RenalDiseaseIndicator'] = bf_df['RenalDiseaseIndicator'].map({'Y':1,'0':0})

    # Map chronic conditions 1->0, 2->1
    chronic_cols = [
        'ChronicCond_Alzheimer','ChronicCond_Heartfailure','ChronicCond_KidneyDisease',
        'ChronicCond_Cancer','ChronicCond_ObstrPulmonary','ChronicCond_Depression',
        'ChronicCond_Diabetes','ChronicCond_IschemicHeart','ChronicCond_Osteoporasis',
        'ChronicCond_rheumatoidarthritis','ChronicCond_stroke'
    ]
    for col in chronic_cols:
        if col in bf_df.columns:
            bf_df[col] = bf_df[col].map({1:0, 2:1})

    # Select relevant beneficiary columns
    selected_bf = [c for c in ['BeneID','Race','Age','Gender','RenalDiseaseIndicator',
                                'NoOfMonths_PartACov','NoOfMonths_PartBCov'] + chronic_cols
                   if c in bf_df.columns]
    bf_df_selected = bf_df[selected_bf].copy()

    ## Merge claim and beneficiary
    claims_with_bene = all_claims_selected.merge(bf_df_selected, on='BeneID', how='left')

    ## Aggregate per provider
    provider_agg = claims_with_bene.groupby('Provider').agg(
        # Claims & amounts
        n_claims = ('ClaimID','nunique'),
        reimb_sum = ('InscClaimAmtReimbursed','sum'),
        reimb_mean = ('InscClaimAmtReimbursed','mean'),
        ded_sum = ('DeductibleAmtPaid','sum'),
        los_mean = ('LengthOfStay','mean'),
        los_p95 = ('LengthOfStay', lambda x: np.nanpercentile(x,95)),
        bene_unique = ('BeneID','nunique'),
        attphy_unique = ('AttendingPhysician','nunique'),
        opphy_unique = ('OperatingPhysician','nunique'),
        dx_mean = ('NumDiagCodes','mean'),
        proc_mean = ('NumProcCodes','mean'),
        age_mean = ('Age','mean'),
        monthsA_mean = ('NoOfMonths_PartACov','mean'),
        monthsB_mean = ('NoOfMonths_PartBCov','mean'),

        # Chronic conditions proportions
        **{f'{col}_mean': (col,'mean') for col in chronic_cols},

        # Flags & demographics counts
        g1_count = ('Gender', lambda x: (x==1).sum()),
        g2_count = ('Gender', lambda x: (x==2).sum()),
        inpatient_count = ('InpatientFlag', 'sum'),
        outpatient_count = ('OutpatientFlag', 'sum'),
        race1_count = ('Race', lambda x: (x==1).sum()),
        race2_count = ('Race', lambda x: (x==2).sum()),
        race3_count = ('Race', lambda x: (x==3).sum()),
        race5_count = ('Race', lambda x: (x==5).sum()),
        renal_count = ('RenalDiseaseIndicator','sum')
    ).reset_index()

    ## Select only training features if provided
    if training_feature_columns is not None:
        X_features = provider_agg[training_feature_columns]
    else:
        X_features = provider_agg.drop(columns=['Provider'])

    return provider_agg


# Prediction workflow
if bf_file and im_file and op_file and provider_file:
    bf_df = pd.read_csv(bf_file)
    im_df = pd.read_csv(im_file)
    op_df = pd.read_csv(op_file)
    unseen_providers = pd.read_csv(provider_file)

    st.info("Running feature engineering and provider aggregation...")
    provider_features = create_provider_features(im_df, op_df, bf_df)
    
    provider_features = pd.merge(provider_features, unseen_providers, on='Provider', how='inner')
    provider_features = provider_features.fillna(0)

    # Load saved model
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Predict
    X_unseen = provider_features.drop(columns=['Provider'])
    y_pred = model.predict(X_unseen)
    y_prob = model.predict_proba(X_unseen)[:,1]
    y_prob = np.round(y_prob, 2) 

    provider_features['PredictedFraud'] = y_pred
    provider_features['FraudProbability'] = y_prob

    # Merge with unseen provider file to include all providers
    output_df = unseen_providers.merge(provider_features, on='Provider', how='left')
    output_df['PredictedFraud'] = output_df['PredictedFraud'].fillna(0).astype(int)
    output_df['FraudProbability'] = output_df['FraudProbability'].map(lambda x: f"{x:.2f}")

    st.write("Predictions for all providers:")
    st.dataframe(output_df[['Provider','PredictedFraud','FraudProbability']])

    st.write("Counts of predicted fraud vs non-fraud:")
    st.write(output_df['PredictedFraud'].value_counts())

    percentages = output_df['PredictedFraud'].value_counts(normalize=True) * 100
    st.write("Percentages:")
    st.dataframe(percentages.to_frame(name='Percentage (%)').style.format("{:.2f}"))

    st.download_button(
    "Download Predictions",
    output_df[['Provider','PredictedFraud','FraudProbability']].to_csv(index=False).encode('utf-8'),
    "unseen_provider_predictions.csv",
    "text/csv")



import streamlit as st
import pandas as pd
import numpy as np
import pickle
# import joblib

st.title("🩺 Provider Fraud Prediction App")

# Upload CSV Files
st.sidebar.header("Upload CSV Files")
bf_file = st.sidebar.file_uploader("Beneficiary CSV", type="csv")
im_file = st.sidebar.file_uploader("Inpatient CSV", type="csv")
op_file = st.sidebar.file_uploader("Outpatient CSV", type="csv")
provider_file = st.sidebar.file_uploader("Provider CSV", type="csv")

# Feature Engineering Function
def create_provider_features(im_df, op_df, bf_df):
    # Concatenate inpatient and outpatient claims
    im_df['ClaimType'] = 'Inpatient'
    op_df['ClaimType'] = 'Outpatient'
    all_claims = pd.concat([im_df, op_df], ignore_index=True)

    # Convert date columns
    date_cols = ['ClaimStartDt','ClaimEndDt','AdmissionDt','DischargeDt']
    for col in date_cols:
        all_claims[col] = pd.to_datetime(all_claims[col], errors='coerce')

    # Compute LOS and claim duration
    all_claims['LengthOfStay'] = (all_claims['DischargeDt'] - all_claims['AdmissionDt']).dt.days
    all_claims['ClaimDuration'] = (all_claims['ClaimEndDt'] - all_claims['ClaimStartDt']).dt.days

    # Count diagnosis and procedure codes
    diag_cols = [f'ClmDiagnosisCode_{i}' for i in range(1, 11)]
    proc_cols = [f'ClmProcedureCode_{i}' for i in range(1, 7)]
    all_claims['NumDiagCodes'] = all_claims[diag_cols].notna().sum(axis=1)
    all_claims['NumProcCodes'] = all_claims[proc_cols].notna().sum(axis=1)

    # Claim type flags
    all_claims['InpatientFlag'] = (all_claims['ClaimType'] == 'Inpatient').astype(int)
    all_claims['OutpatientFlag'] = (all_claims['ClaimType'] == 'Outpatient').astype(int)

    def create_claim_flag(row):
        if row['InpatientFlag']==1 and row['OutpatientFlag']==0: return 1
        elif row['InpatientFlag']==0 and row['OutpatientFlag']==1: return 2
        elif row['InpatientFlag']==1 and row['OutpatientFlag']==1: return 3
        else: return 0
    all_claims['ClaimFlag'] = all_claims.apply(create_claim_flag, axis=1)

    # Beneficiary info
    bf_df['DOB'] = pd.to_datetime(bf_df['DOB'], errors='coerce')
    bf_df['DOD'] = pd.to_datetime(bf_df['DOD'], errors='coerce')
    bf_df['Age'] = bf_df.apply(lambda row: ((row['DOD'] if pd.notnull(row['DOD']) else pd.Timestamp.today()) - row['DOB']).days // 365, axis=1)
    bf_df['RenalDiseaseIndicator'] = bf_df['RenalDiseaseIndicator'].map({'Y':1,'0':0})

    # Merge claims with beneficiary info
    claim_bf = all_claims.merge(bf_df, on='BeneID', how='left')


    # Aggregate per provider
    provider_agg = claim_bf.groupby("Provider").agg(
        n_claims=("ClaimID","nunique"),
        reimb_sum=("InscClaimAmtReimbursed","sum"),
        reimb_mean=("InscClaimAmtReimbursed","mean"),
        ded_sum=("DeductibleAmtPaid","sum"),
        los_mean=("LengthOfStay","mean"),
        los_p95=("LengthOfStay", lambda x: np.nanpercentile(x,95)),
        bene_unique=("BeneID","nunique"),
        attphy_unique=("AttendingPhysician","nunique"),
        opphy_unique=("OperatingPhysician","nunique"),
        dx_mean=("NumDiagCodes","mean"),
        proc_mean=("NumProcCodes","mean"),
        age_mean=("Age","mean"),
        monthsA_mean=("NoOfMonths_PartACov","mean"),
        monthsB_mean=("NoOfMonths_PartBCov","mean"),
        chronic_Alz_mean=("ChronicCond_Alzheimer","mean"),
        chronic_HF_mean=("ChronicCond_Heartfailure","mean"),
        chronic_KD_mean=("ChronicCond_KidneyDisease","mean"),
        chronic_Cancer_mean=("ChronicCond_Cancer","mean"),
        chronic_OBS_mean=("ChronicCond_ObstrPulmonary","mean"),
        chronic_Dep_mean=("ChronicCond_Depression","mean"),
        chronic_Diab_mean=("ChronicCond_Diabetes","mean"),
        chronic_IHD_mean=("ChronicCond_IschemicHeart","mean"),
        chronic_Osteo_mean=("ChronicCond_Osteoporasis","mean"),
        chronic_RA_mean=("ChronicCond_rheumatoidarthritis","mean"),
        chronic_Stroke_mean=("ChronicCond_stroke","mean"),
        IPReimb_mean=("IPAnnualReimbursementAmt","mean"),
        IPDed_mean=("IPAnnualDeductibleAmt","mean"),
        OPReimb_mean=("OPAnnualReimbursementAmt","mean"),
        OPDed_mean=("OPAnnualDeductibleAmt","mean"),
        g1_count=("Gender", lambda x:(x==1).sum()),
        g2_count=("Gender", lambda x:(x==2).sum()),
        ClaimFlag_mode=("ClaimFlag", lambda x: x.mode()[0] if not x.mode().empty else 0),
        race_mode=("Race", lambda x: x.mode()[0] if not x.mode().empty else 0),
        renal_mode=("RenalDiseaseIndicator","max")
    ).reset_index()

    return provider_agg

# Prediction workflow
if bf_file and im_file and op_file and provider_file:
    bf_df = pd.read_csv(bf_file)
    im_df = pd.read_csv(im_file)
    op_df = pd.read_csv(op_file)
    unseen_providers = pd.read_csv(provider_file)

    st.info("Running feature engineering and provider aggregation...")
    provider_features = create_provider_features(im_df, op_df, bf_df)
    
    provider_features=pd.merge(provider_features,unseen_providers,on='Provider',how='inner')
    provider_features=provider_features.fillna(0)

    # Load saved model
    # model = joblib.load("xgb_model.pkl")

    # Save model
    # with open('model.pkl', 'wb') as f:
    # pickle.dump(model, f)

    # Load model
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Predict
    X_unseen = provider_features.drop(columns=['Provider'])
    y_pred = model.predict(X_unseen)
    y_prob = model.predict_proba(X_unseen)[:,1]

    provider_features['PredictedFraud'] = y_pred
    provider_features['FraudProbability'] = y_prob

    # Merge with unseen provider file to include all providers
    output_df = unseen_providers.merge(provider_features, on='Provider', how='left')
    output_df['PredictedFraud'] = output_df['PredictedFraud'].fillna(0).astype(int)
    output_df['FraudProbability'] = output_df['FraudProbability'].fillna(0.0)

    st.write("Predictions for all providers:")
    st.dataframe(output_df[['Provider','PredictedFraud','FraudProbability']])

    st.write("Counts of predicted fraud vs non-fraud:")
    st.write(output_df['PredictedFraud'].value_counts())

    st.write("Percentages:")
    st.write(output_df['PredictedFraud'].value_counts(normalize=True)*100)

    st.download_button(
        "Download Predictions",
        output_df.to_csv(index=False).encode('utf-8'),
        "unseen_provider_predictions.csv",
        "text/csv"
    )




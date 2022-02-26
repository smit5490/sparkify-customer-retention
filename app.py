import streamlit as st
import pandas as pd
import shap
shap.initjs()
import joblib
import numpy as np
import matplotlib.pyplot as plt


# Draw a title and some text to the app:
col1, col2, col3, col4 = st.columns(4)
with col4:
    st.image("./images/sparkify.png")

'''
# Sparkify Churn Prediction Model

Use the sidebar on the left to filter for and select a User ID.
'''
st.info("""
        The model used to predict customer churn is a **Histogram-based Gradient Boosting Classifier**. See the 
        [Github Repository](https://github.com/smit5490/sparkify-customer-retention) for modeling information.
        """)

# Get data and model
train = pd.read_parquet("./data/train_data_full")
test = pd.read_parquet("./data/test_data_full")
data = pd.concat([train,test])
data = data.reset_index(drop=True)
gbc_model = joblib.load("./models/sklearn_gbc_full.pkl")
X_data = data.drop("churn", axis = 1)
y_test = data[["churn"]]
X_data_means = X_data.iloc[:,2:].mean()

# Get predictions
preds = gbc_model.predict_proba(X_data)[:,1]*100

# Get variable names
vars1 = gbc_model["preprocessing"].transformers_[0][1][1].get_feature_names_out().tolist()
vars2 = gbc_model["preprocessing"].transformers_[1][2]
vars3 = gbc_model["preprocessing"].transformers_[2][2]
var_names = vars1+vars2+vars3


# Sidebar
st.sidebar.markdown("## Filter User IDs:")
active_acct = st.sidebar.checkbox("Active Accounts Only")
likelihood = st.sidebar.slider("Churn Probability:",
                               min_value = int(preds.min()),
                               max_value = int(preds.max()),
                               value = (int(preds.min()),int(preds.max())))

preds = pd.concat([data,pd.Series(preds, name = "prob")], axis = 1)
preds = preds[(preds["prob"]>likelihood[0]) & (preds["prob"]<likelihood[1])]

if active_acct:
    preds = preds[preds["churn"]==0]

userId_ls = preds["userId"].tolist()


st.sidebar.write("Number of customers remaining: {}".format(preds.shape[0]))
st.sidebar.markdown("## Select User ID:")
userId = st.sidebar.selectbox("User ID", userId_ls)

run_button = True
if run_button:
    user_summary = X_data[X_data["userId"] == userId]
    user_index = X_data[X_data["userId"] == userId].index

    # make prediction
    gbc_prob = gbc_model.predict_proba(user_summary)[0][1]

    # Compute SHAP values for model:
    X_data_preprocessed = gbc_model['preprocessing'].transform(X_data)
    explainer = shap.Explainer(gbc_model['model'])
#    st.write(explainer.expected_value)
    # Compute shap value for selected user
    shap_values = explainer.shap_values(X_data_preprocessed, check_additivity=False)

    # Main Panel
    str_title = "Usage Summary for: {}".format(userId)
    st.markdown("<h2 style='text-align: center; color: black;'>{}</h2>".format(str_title), unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        active = y_test.iloc[user_index, :].values[0][0]
        active = str(np.where(active == 1, "Inactive", "Active"))
        st.metric(label = "Account Status:",
                  value = active,
                  )
    with col2:
        age = user_summary["tenure_days"].values[0]
        avg_age = X_data_means["tenure_days"]
        age_diff = age - avg_age
        st.metric(label = "Account Age (Days)",
                  value = round(age),
                  delta = round(age_diff))
    with col3:
        sub = user_summary["paid"].values[0]
        sub = str(np.where(sub==1,"Paid", "Free"))
        st.metric(label = "Subscription Type",
                  value = sub)
    with col4:
        thumbs_up_pct = user_summary["thumbs_up_pct"].values[0]
        mean_thumbs_up_pct = X_data_means["thumbs_up_pct"]
        thumbs_diff = thumbs_up_pct - mean_thumbs_up_pct
        st.metric(label = "Thumbs Up %",
                  value = str(round(thumbs_up_pct*100)),
                  delta = round(thumbs_diff*100))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        advert_rate = user_summary["advert_rate"].values[0]
        mean_advert_rate = X_data_means["advert_rate"]
        advert_diff = advert_rate - mean_advert_rate
        st.metric(label="Advert. Rate (per hour)",
                  value=str(round(advert_rate,2)),
                  delta = round(advert_diff,2))
    with col2:
        number_sessions = user_summary["session_count"].values[0]
        avg_number_sessions = X_data_means["session_count"]
        session_diff = number_sessions - avg_number_sessions

        st.metric(label="Number of Sessions",
                  value=number_sessions,
                  delta = round(session_diff))
    with col3:
        avg_session_length_hours = user_summary["avg_session_length_hours"].values[0]
        avg_avg_session_length_hours = X_data_means["avg_session_length_hours"]
        avg_session_length_hours_diff = avg_session_length_hours - avg_avg_session_length_hours
        st.metric(label="Average Session Length",
                  value=round(avg_session_length_hours),
                  delta = round(avg_session_length_hours_diff))
    with col4:
        non_int_rate = user_summary["non_song_interaction_rate"].values[0]
        avg_non_int_rate = X_data_means["non_song_interaction_rate"]
        non_int_rate_diff = non_int_rate - avg_number_sessions
        st.metric(label = "Non-song Int. Rate",
                  value = round(non_int_rate,2),
                  delta = round(non_int_rate_diff,2))

#    st.write("Other account features:")
#    st.write(user_summary.drop("gender",axis=1))

    prob_title = "Likelihood of Churn: {}%".format(round(gbc_prob * 100))
    color = np.where(gbc_prob < .50, "green","red")
    st.markdown("<h3 style='text-align: center; color: {};'>{}</h3>".format(color,prob_title), unsafe_allow_html=True)

    st.write(""" ## Drivers of Churn: """)
    st.write(" ### Force Plot")
    fig = shap.force_plot(explainer.expected_value,
                          shap_values[user_index,:],
                          X_data_preprocessed[user_index,:].round(4),
                          feature_names=var_names,
                          show=False,
                          matplotlib=True,
                          link='identity')
    st.pyplot(fig)
    # Force plot expander explanations
    with st.expander("More on force plots"):
        st.markdown("""
            The Force plot shows how each feature has contributed in increasing or decreasing the base value 
            (e.g. average class output of the test dataset) to the predicted value for the selected User ID.
            Those values are **log odds**: The SHAP values displayed are additive. Once the negative values (blue) are 
            substracted from the positive values (red), the distance from the base value to the output remains.
            """)
    st.write(" ### Feature Decision Plot")
    fig, ax = plt.subplots()

    ax = shap.decision_plot(explainer.expected_value,
                       shap_values[user_index, :],
                       feature_names=var_names,
#                       title = "Feature Decision Plot",
                       link = "logit")

    st.pyplot(fig)

    # Decision plot expander explanations
    with st.expander("More on decision plots"):
        st.markdown("""
         Just like the force plot, the decision plot shows how each feature has contributed in increasing or decreasing
          the base value (the grey line, aka. the average model output on the test dataset) to the predicted value for 
          the selected User ID.
    It also show the impact of less influential features more clearly.
    From SHAP documentation:
    - *The x-axis represents the model's output (e.g. probability of churn)*
    - *The plot is centered on the x-axis at explainer.expected_value (the base value). All SHAP values are relative to the model's expected value like a linear model's effects are relative to the intercept.*
    - *The y-axis lists the model's features. By default, the features are ordered by descending importance. The importance is calculated over the observations plotted. _This is usually different than the importance ordering for the entire dataset._ In addition to feature importance ordering, the decision plot also supports hierarchical cluster feature ordering and user-defined feature ordering.*
    - *Moving from the bottom of the plot to the top, SHAP values for each feature are added to the model's base value. This shows how each feature contributes to the overall prediction.*
    - *At the bottom of the plot, the observations converge at explainer.expected_value (the base value)*""")

    st.write("#### Variable Dependence Plot")
    fig, ax = plt.subplots()
    col1, col2 = st.columns(2)

    with col1:
        ind_var = st.selectbox("Independent Variable", var_names, index=8)
    with col2:
        int_var = st.selectbox("Interaction Variable", var_names, index = 48)

    shap.dependence_plot(ind_var,
                        shap_values,
                        X_data_preprocessed,
                        feature_names=var_names,
                        interaction_index=int_var,
                        ax = ax,
                        alpha=.25)

    st.pyplot(fig)
# Dependance plots expander explanations
with st.expander("More on dependence plots"):
     st.markdown("""
     From the SHAP documentation:
"*A dependence plot is a scatter plot that shows the effect a single feature has on the predictions made by the model.* 
- *Each dot is a single prediction (row) from the dataset.*
- *The x-axis is the value of the feature (from the X matrix).*
- *The y-axis is the SHAP value for that feature, which represents how much knowing that feature's value changes the output of the model for that sample's prediction. For this model the units are log-odd probabilities of churn.*
- *The color corresponds to a second feature that may have an interaction effect with the feature we are plotting (by default this second feature is chosen automatically). If an interaction effect is present between this other feature and the feature we are plotting it will show up as a distinct vertical pattern of coloring.*"
                """)
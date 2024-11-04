import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import pickle


def load_pickle_data():
    with open('App/model/svc_model.pkl', 'rb') as file:
        return pickle.load(file)


data = load_pickle_data()

df = data['dataframe']
y_test = data['y_test']
y_pred = data['y_pred']


def dashboard():
    st.title("Dashboard")

    st.divider()

    # ---------------------------------------------------------------------------
    # Histogram for Age
    df["Age"].hist(bins=15, grid=False, color='blue')

    plt.title('Histogram for Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    st.pyplot(plt.gcf())

    st.markdown(
        """
        - The mean of the age dataset was 48.0 years old.
        - The median of the age dataset was 47.5 years old.
        - Since the mean and the median were almost the same, the age dataset was symmetric.
        """
    )

    st.divider()

    # ---------------------------------------------------------------------------
    # BAR CHART for counts of positive and negative on features with 'Yes' values
    features = ["48 & Under", "Over 48", "Male", "Female", "Polyuria", "Polydipsia", "sudden weight loss", "weakness", "Polyphagia", "Genital thrush", "visual blurring",
                "Itching", "Irritability", "delayed healing", "partial paresis", "muscle stiffness", "Alopecia", "Obesity"]

    # get counts on age, male, female, features with answering 'Yes'
    age48down_pos = len(df.loc[(df['Age'] <= 48) & (df['class'] == 1)])
    age48down_neg = len(df.loc[(df['Age'] <= 48) & (df['class'] == 0)])
    age48up_pos = len(df.loc[(df['Age'] > 48) & (df['class'] == 1)])
    age48up_neg = len(df.loc[(df['Age'] > 48) & (df['class'] == 0)])
    mal_pos = len(df.loc[(df['Gender'] == 1) & (df['class'] == 1)])
    mal_neg = len(df.loc[(df['Gender'] == 1) & (df['class'] == 0)])
    fem_pos = len(df.loc[(df['Gender'] == 0) & (df['class'] == 1)])
    fem_neg = len(df.loc[(df['Gender'] == 0) & (df['class'] == 0)])
    uri_pos = len(df.loc[(df['Polyuria'] == 1) & (df['class'] == 1)])
    uri_neg = len(df.loc[(df['Polyuria'] == 1) & (df['class'] == 0)])
    dip_pos = len(df.loc[(df['Polydipsia'] == 1) & (df['class'] == 1)])
    dip_neg = len(df.loc[(df['Polydipsia'] == 1) & (df['class'] == 0)])
    swl_pos = len(df.loc[(df['sudden weight loss'] == 1) & (df['class'] == 1)])
    swl_neg = len(df.loc[(df['sudden weight loss'] == 1) & (df['class'] == 0)])
    wea_pos = len(df.loc[(df['weakness'] == 1) & (df['class'] == 1)])
    wea_neg = len(df.loc[(df['weakness'] == 1) & (df['class'] == 0)])
    pha_pos = len(df.loc[(df['Polyphagia'] == 1) & (df['class'] == 1)])
    pha_neg = len(df.loc[(df['Polyphagia'] == 1) & (df['class'] == 0)])
    gt_pos = len(df.loc[(df['Genital thrush'] == 1) & (df['class'] == 1)])
    gt_neg = len(df.loc[(df['Genital thrush'] == 1) & (df['class'] == 0)])
    vb_pos = len(df.loc[(df['visual blurring'] == 1) & (df['class'] == 1)])
    vb_neg = len(df.loc[(df['visual blurring'] == 1) & (df['class'] == 0)])
    itc_pos = len(df.loc[(df['Itching'] == 1) & (df['class'] == 1)])
    itc_neg = len(df.loc[(df['Itching'] == 1) & (df['class'] == 0)])
    irr_pos = len(df.loc[(df['Irritability'] == 1) & (df['class'] == 1)])
    irr_neg = len(df.loc[(df['Irritability'] == 1) & (df['class'] == 0)])
    dh_pos = len(df.loc[(df['delayed healing'] == 1) & (df['class'] == 1)])
    dh_neg = len(df.loc[(df['delayed healing'] == 1) & (df['class'] == 0)])
    pp_pos = len(df.loc[(df['partial paresis'] == 1) & (df['class'] == 1)])
    pp_neg = len(df.loc[(df['partial paresis'] == 1) & (df['class'] == 0)])
    ms_pos = len(df.loc[(df['muscle stiffness'] == 1) & (df['class'] == 1)])
    ms_neg = len(df.loc[(df['muscle stiffness'] == 1) & (df['class'] == 0)])
    alp_pos = len(df.loc[(df['Alopecia'] == 1) & (df['class'] == 1)])
    alp_neg = len(df.loc[(df['Alopecia'] == 1) & (df['class'] == 0)])
    obe_pos = len(df.loc[(df['Obesity'] == 1) & (df['class'] == 1)])
    obe_neg = len(df.loc[(df['Obesity'] == 1) & (df['class'] == 0)])

    y_pos = [age48down_pos, age48up_pos, mal_pos, fem_pos, uri_pos, dip_pos, swl_pos, wea_pos,
             pha_pos, gt_pos, vb_pos, itc_pos, irr_pos, dh_pos, pp_pos, ms_pos, alp_pos, obe_pos]
    y_neg = [age48down_neg, age48up_neg, mal_neg, fem_neg, uri_neg, dip_neg, swl_neg, wea_neg,
             pha_neg, gt_neg, vb_neg, itc_neg, irr_neg, dh_neg, pp_neg, ms_neg, alp_neg, obe_neg]

    width = 0.5
    plt.figure(figsize=(15, 7))
    plt.bar(features, y_pos, width, color='r')
    plt.bar(features, y_neg, width, bottom=y_pos, color='b')
    plt.xlabel('Feature')
    plt.ylabel('Count')
    plt.title('Counts of Positive and Negative On Features With \'Yes\' Value')
    plt.legend(['Positive', 'Negative'])
    plt.xticks(rotation=30)  # for xlabel not overlapping each other
    st.pyplot(plt.gcf())

    st.markdown(
        """
        - Age, male, itching, delayed healing, and alopecia(hair loss) were less likely to be a sign of diabetes.
        - Female, polyuria(excessive urination amount), polydipsia(excessive thirst), and irritability(mood swing) were more likely to be a sign of diabetes.
        """
    )

    st.divider()

    # ---------------------------------------------------------------------------
    # CORRELATION MATRIX HEATMAP
    fig, axs = plt.subplots(figsize=(15, 8))
    sns.heatmap(df.corr(), cmap='Greens', annot=True)
    plt.title('Correlation Matrix Heatmap', fontsize=20)
    st.pyplot(fig)

    st.markdown(
        """
        - Polyuria(excessive urination amount) and polydipsia(excessive thirst) were moderately correlated.
        - None of the features were highly correlated except itself.
        """
    )

    st.divider()

    # ---------------------------------------------------------------------------
    # CONFUSION MATRIX
    labels = ['Negative', 'Positive']
    cm = confusion_matrix(y_test.values, y_pred)
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=labels).plot()
    plt.title('Confusion Matrix of The Class')
    st.pyplot(plt.gcf())

    st.markdown(
        """
        - **Precision**: True Positive / (True Positive + False Positive) = 91 / (91 + 3) = **0.968**
        - **Recall**: True Positive / (True Positive + False Negative) = 91 / (91 + 3) = **0.968**
        - **Accuracy**: Correct Prediction / All Predictions = (59 + 91) / (59 + 3 + 3 + 91) = **0.962**
        """
    )


dashboard()

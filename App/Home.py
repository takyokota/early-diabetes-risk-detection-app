import streamlit as st


def homepage():
    st.set_page_config(page_title="Early Stage Diabetes Risk Prediction App")
    st.title(":blue[Early Diabetes Risk Prediction App]")

    st.divider()

    st.markdown(
        """
        Diabetes has been one of the major health concerns in society for a long time \
        because it increases the risk of other health issues including heart attack, stroke, \
        kidney failure, and vision loss. There are a significant number of people who do not know \
        that they have diabetes. The earlier detection of diabetes helps them to prevent developing \
        other diseases caused by diabetes with the right treatments in the earlier stage.\
        This application predicts the risk of diabetes. 
        """
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.write("*To understand data:*")
        st.page_link("pages/Dashboard.py", label="Jump To Dashboard", icon="▶")

    with col2:
        st.write("*To check if you might have diabetes:*")
        st.page_link("pages/Application.py",
                     label="Jump To Application", icon="▶")


homepage()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("experiment_15.csv")
        return df
    except FileNotFoundError:
        st.error("Error: 'experiment_15.csv' not found. Make sure the file is in the correct directory.")
        return None

df = load_data()

if df is not None:
    st.title("Feature Engineering and Data Analysis")

    # Display raw data
    st.subheader("Raw Dataset")
    st.dataframe(df.head())

    # Feature Engineering
    st.subheader("Feature Engineering")
    new_features_df = pd.DataFrame()
    new_features_df['X1_PositionError'] = df['X1_CommandPosition'] - df['X1_ActualPosition']
    new_features_df['Y1_PositionError'] = df['Y1_CommandPosition'] - df['Y1_ActualPosition']
    new_features_df['Z1_PositionError'] = df['Z1_CommandPosition'] - df['Z1_ActualPosition']
    
    st.dataframe(new_features_df.head())

    # Visualization
    st.subheader("Feature Distributions")
    selected_feature = st.selectbox("Select a feature for distribution plot:", new_features_df.columns)
    fig, ax = plt.subplots()
    sns.histplot(new_features_df[selected_feature], kde=True, ax=ax)
    st.pyplot(fig)

    # Boxplot for outliers
    st.subheader("Box Plot")
    fig, ax = plt.subplots()
    new_features_df.boxplot(column=[selected_feature], ax=ax)
    st.pyplot(fig)

    # Scatter Plot
    st.subheader("Scatter Plot: Position Error vs. Velocity Error")
    fig, ax = plt.subplots()
    ax.scatter(new_features_df['X1_PositionError'], new_features_df['Y1_PositionError'], alpha=0.5)
    ax.set_xlabel('X1 Position Error')
    ax.set_ylabel('Y1 Position Error')
    st.pyplot(fig)

    st.success("Feature Engineering and EDA completed!")

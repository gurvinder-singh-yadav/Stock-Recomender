import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import datetime
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

today = str(datetime.date.today())



@st.cache
def train_model1(name:str = "grow"):
    fund_path = "data/{}/{}/funds.csv".format(name ,today)
    df = pd.read_csv(fund_path)
    stock_matrix_UII = df.pivot_table(index='funds_name', columns='Name', values='Assets(Rs_Cr.)')
    stock_matrix_UII.to_csv('models/model1.csv')
    return stock_matrix_UII.columns

@st.cache
def collFiltering(name : str, n:int = 10): 
    path = "models/model1.csv" 
    if not os.path.exists(path):
        train_model1()
    stock_matrix_UII = pd.read_csv('models/model1.csv')
    learned_stocks = stock_matrix_UII.columns[1:]
    if name not in learned_stocks:
        return "Stock Cannot be Predicted"
    
    stock_matrix_UII.set_index('funds_name')
    stock_name = name
    similar_to_stock = stock_matrix_UII.corrwith(stock_matrix_UII.loc[:, stock_name]).dropna()
    corr_stock = pd.DataFrame(similar_to_stock, columns=['Correlation'])
    corr_stock.dropna(inplace=True)
    res = corr_stock.sort_values(by='Correlation', ascending=False).reset_index()
    return res["index"].tolist()[:n]

@st.cache
def train_model2(name:str = "grow"):
    fund_path = "data/{}/{}/funds.csv".format(name ,today)
    df = pd.read_csv(fund_path)
    stock_matrix_UII = df.pivot_table(index='funds_name', columns='Name', values='Assets(Rs_Cr.)')
    stock_matrix_UII.to_csv('models/model2.csv')
    return stock_matrix_UII.columns

@st.cache
def cosine_simi(name: str, n:int = 10):
    path = 'models/model2.csv'
    # Load the user-item matrix
    if not os.path.exists(path):
        train_model2()
    df = pd.read_csv(path, index_col=0)
    learned_stocks = df.columns[1:]

    if name not in learned_stocks:
        return "Stock Cannot be Predicted"

    # Replace missing values with 0
    df = df.fillna(0)

    # Calculate the cosine similarity between columns
    cosine_sim = cosine_similarity(df.T)

    # Get the name of the target column
    target_col = name

    # Get the index of the target column
    target_col_idx = df.columns.get_loc(target_col)

    # Get the similarity scores between the target column and all other columns
    sim_scores = list(enumerate(cosine_sim[target_col_idx]))

    # Sort the columns based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 10 similar columns
    top_cols = [df.columns[i] for i, score in sim_scores if i != target_col_idx][:10]
        
    return top_cols

st.set_page_config(
    page_title="Stock Recommendations",
    layout="wide",
    initial_sidebar_state="auto"
)

st.title("Stock Recommendations")

st.sidebar.markdown("Main Page")

@st.cache
def load_data(name: str, nrows:int = 50):
    fund_path = "data/{}/{}/funds.csv".format(name ,today)
    df = pd.read_csv(fund_path, nrows=nrows)
    return df

st.subheader("Individual Stock Information of Mutual Funds listed on grow")
st.dataframe(load_data("grow"))

@st.cache
def get_stock_volume(name:str, n:int = 10):
    fund_path = "data/{}/{}/funds.csv".format(name ,today)
    df = pd.read_csv(fund_path)
    df = df[["Name", "Assets(Rs_Cr.)"]]
    df = df.groupby("Name").aggregate(sum).reset_index()
    df = df[["Name", "Assets(Rs_Cr.)"]].sort_values(by = "Assets(Rs_Cr.)", ascending=False)
    df = df.set_index("Name")
    df = df.head(n)
    return df

@st.cache
def get_stocks(name:str, n:int = 10):
    fund_path = "data/{}/{}/funds.csv".format(name ,today)
    df = pd.read_csv(fund_path)
    return df["Name"].unique().tolist()[:n]


st.bar_chart(get_stock_volume("grow"), height=1000)


option = st.selectbox(
    "Get me most similar stocks of the given stock",
    get_stocks("grow")
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model 1")
    # st.text("Description")
    st.text_area("Description", "Correlation is another measure of similarity that is commonly used in recommendation systems. Correlation measures the linear relationship between two variables. In the context of a recommendation system, the variables represent the user's past behavior and the items being recommended. The correlation between the two variables can be positive or negative, and the closer the correlation is to 1 (in absolute value), the stronger the relationship between the variables is considered to be.", height=200)
    st.subheader("Results : ")
    for stock in collFiltering(option):
        st.markdown("-  "+stock)

with col2:
    st.subheader("Model 2")
    st.text_area("Description", "Cosine similarity is a measure of similarity between two vectors in a multi-dimensional space. In the context of a recommendation system, the vectors represent the user's past behavior (e.g., items they have viewed or purchased) and the items being recommended. The cosine similarity between two vectors is calculated as the cosine of the angle between them. The closer the cosine similarity is to 1, the more similar the two vectors are considered to be.", height=200)
    st.subheader("Results : ")
    for stock in cosine_simi(option):
        st.markdown("-  "+stock)


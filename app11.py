import streamlit as st
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.llms import OpenAI 
import os 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
import pandas as pd
import plotly.graph_objects as go
from langchain_google_genai import GoogleGenerativeAI
import os
from langchain.prompts import PromptTemplate
import json
import pathlib
import pandas as pd
import PyPDF2
import fitz
import pdfplumber
from PIL import Image
import io
import fitz 
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI


# Load your dataset



df = pd.read_csv("contract_data_preprocessed1.csv")

# Create a navigation bar on the left side
menu = ["Analysis", "Q&A"]
st.title("Contract Analysis")
qqq=st.text_input('enter apikey')
os.environ['OPENAI_API_KEY']=qqq
os.environ["GOOGLE_API_KEY"] = "AIzaSyCeXPjXmwt-S2PVRxD9SkA-XQZZN1lzOgw"

# Sidebar radio button
choice = st.sidebar.radio("Choose an option", menu)
# Function to display question details

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text
 
def generate_contract(input_text):
    prompt = f"Generate a contract similar to the following text:\n\n{input_text}\n\nPlease create a professional IT service Agreement that is based on the example provided and is of minimum 6 pages in length."
    
 
    message = HumanMessage(content=prompt)
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    response=llm([message])
 
    content = response.dict()
 
    content= content["content"]
    print("here")
    # Write content to a text file
    with open('contract.txt', 'w') as file:
        file.write(content)
 
    print("Content has been written to 'contract_review.txt' file.")
 
    return response


def new_func():
        # Path to the input PDF file
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Extract text from the uploaded PDF
        extracted_text = extract_text_from_pdf(uploaded_file)
        
        # Generate the contract
        contract = generate_contract(extracted_text)
        
        # Display the contract
        edited_contract = st.text_area("Edit the contract", value=contract, height=300)
    
    # Optionally, you can use the edited contract further in your application
        if st.button("Save Contract"):
            st.write("The contract has been saved.")
            # Implement the logic to save the edited contract

        # Display the edited contract
        st.write("Edited Contract:")
        st.write(edited_contract)

        #st.write(contract)
        #st.write("Contract generated and saved to 'generated_contract.txt'")

   


def display_question():
    # Create a dropdown to select from the 'file' column
    selected_file = st.selectbox("Select File:", df["file"].unique())

    # Filter the DataFrame based on the selected file
    filtered_df = df[df["file"] == selected_file]

    # Display the remaining columns and their values
    st.write("## File Details:")
    for column in filtered_df.columns:
        if column != "file":  # Skip the 'file' column itself
            st.write(f"**Question:** {column}")
            st.write(f"**Answer:** {filtered_df.iloc[0][column]}")
            st.write("---")  # Separator between questions

# Function to display analysis
def display_analysis():
    # Assuming df is your DataFrame with the 'Year' column already extracted
    # Count the occurrences of each year
    df['Agreement Date'] = pd.to_datetime(df['Agreement Date'], errors='coerce')

    # Extract year from the datetime column
    df['Year'] = df['Agreement Date'].dt.year
    year_counts = df['Year'].value_counts().sort_index()

    # Create a bar plot using Plotly
    fig = go.Figure([go.Bar(x=year_counts.index, y=year_counts.values)])

    # Update layout
    fig.update_layout(
        title='Year-wise Contracts Count',
        xaxis=dict(title='Year', tickmode='array', tickvals=year_counts.index, ticktext=year_counts.index),
        yaxis=dict(title='Count'),
        bargap=0.1
    )

    # Display the plot using Streamlit
    st.plotly_chart(fig)

    df1=df[df['Year']==2019]
    second_party_counts = df1['Second Party'].value_counts()

    # Create a donut chart using Plotly
    fig = go.Figure(go.Pie(labels=second_party_counts.index, values=second_party_counts.values, hole=0.3))

    # Update layout
    fig.update_layout(
        title='Second Party Occurrences Percentagewise for year 2019',
        legend_title='Second Party',
        showlegend=True,
        width=800,  # Set the width
    height=600
    )

    # Display the plot using Streamlit
    st.plotly_chart(fig)
    df1=df[df['Year']==2019]
    second_party_counts = df1['first_party_location'].value_counts()

    # Create a donut chart using Plotly
    fig = go.Figure(go.Pie(labels=second_party_counts.index, values=second_party_counts.values, hole=0.3))

    # Update layout
    fig.update_layout(
        title='First Party Location Percentagewise for year 2019',
        legend_title='Second Party',
        showlegend=True,
        width=800,  # Set the width
    height=600
    )

    # Display the plot using Streamlit
    st.plotly_chart(fig)

    
    import plotly.express as px

    # Count occurrences of each name
    df1=df[df['Year']==2019]
    name_counts = df1['Name'].value_counts().reset_index()
    name_counts.columns = ['Name', 'Count']

    # Plotting
    fig = px.bar(name_counts, x='Name', y='Count', title='Count of Aggrement Names in year 2019')
    st.plotly_chart(fig)



    agent_try = create_pandas_dataframe_agent(ChatOpenAI(temperature=0,model="gpt-4"),df,verbose=True,agent_type=AgentType.OPENAI_FUNCTIONS)
    user_input = st.text_input("Enter your text:")

    if st.button("Submit"):
        out=agent_try(user_input)
        st.write(out['output'])

llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
# Function to display Q&A


def display_qna1():

    llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()

        try:
            # Save the uploaded PDF file temporarily
            with open("temp_upload.pdf", "wb") as f:
                f.write(file_bytes)

            # Process the uploaded PDF file
            text, images, tables = info_from_pdf("temp_upload.pdf")

            # Add OCR processing for images
            try:
                if len(images) != 0:
                    reader = easyocr.Reader(['en'])
                    for img in images:
                        result = reader.readtext(img)
                        for (bbox, text_image, prob) in result:
                            text += text_image + '\n'
            except Exception as e:
                st.error(f"An error occurred during OCR: {e}")

            # Format tables into text
            formatted_tables = format_tables(tables)
            text += formatted_tables

            question = st.text_input("Enter your question:")

            if st.button("Submit"):
                answer = get_answer(question, text)
                st.write(answer)

        except Exception as e:
            st.error(f"An error occurred: {e}")

        finally:
            # Delete the temporary uploaded file
            if os.path.exists("temp_upload.pdf"):
                os.remove("temp_upload.pdf")

def info_from_pdf(file_path):
    text = ""
    images = []
    tables = []

    # Extract text from PDF
    with open(file_path, "rb") as pdf_file:
        reader = fitz.open(file_path)
        for page_number in range(len(reader)):
            page = reader.load_page(page_number)
            text += page.get_text()

    # Extract images from PDF
    doc = fitz.open(file_path)
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        image_list = page.get_images(full=True)
        for image_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            image_file_path = f"image_{len(images)}.png"
            image.save(image_file_path)
            images.append(image_file_path)

    # Extract tables from PDF
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted_tables = page.extract_tables()
            if extracted_tables:
                tables.extend(extracted_tables)

    return text, images, tables

def format_tables(tables):
    formatted_tables = "\n\nTables:\n\n"
    for table in tables:
        formatted_tables += str(table) + "\n"
    return formatted_tables

def get_answer(question, context):
    template = """
    You are an expert in the field of Contracts data management. You are giving advice on a contract provided as context. You are truthful and never lie. Never make up facts and if you are not 100 percent sure, reply with why you cannot answer in a truthful way. Give a detailed answer to the question using information from the provided contexts.
    Context : {context}
    Question : {question}
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    answer=chain.invoke({"context": context, "question": question})
    st.write(answer)
    return 

def display_qna():
    
    folder_path = r"D:\Tellius Project\contract_streamlit\full_contract_pdf"
    pdf_files = os.listdir(folder_path)[:10]  # Get the first 10 PDF files in the folder

    question = st.text_input("Enter your text:")

    if st.button("Submit"):
        for pdf_file in pdf_files:
            st.write(pdf_file)
            file_path = os.path.join(folder_path, pdf_file)

            text, images, tables = info_from_pdf(file_path)

            try:
                if len(images) != 0:
                    reader = easyocr.Reader(['en'])
                    for img in images:
                        result = reader.readtext(img)
                        for (bbox, text_image, prob) in result:
                            text += text_image + '\n'
            except Exception as e:
                # Handle the exception here
                print(f"An error occurred: {e}")

            formatted_tables = format_tables(tables)
            text += formatted_tables

            st.write(get_answer(question, text))

def info_from_pdf(file_path):
    text = ""
    images = []
    tables = []

    # Extract text from PDF
    with open(file_path, "rb") as pdf_file:
        reader = fitz.open(file_path)
        for page_number in range(len(reader)):
            page = reader.load_page(page_number)
            text += page.get_text()

    # Extract images from PDF
    doc = fitz.open(file_path)
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        image_list = page.get_images(full=True)
        for image_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            image_file_path = f"image_{len(images)}.png"
            image.save(image_file_path)
            images.append(image_file_path)

    # Extract tables from PDF
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted_tables = page.extract_tables()
            if extracted_tables:
                tables.extend(extracted_tables)

    return text, images, tables

def format_tables(tables):
    formatted_tables = "\n\nTables:\n\n"
    for table in tables:
        formatted_tables += str(table) + "\n"
    return formatted_tables

def get_answer(question, context):
    template = """
    You are an expert in the field of Contracts data management. You are giving advice on a contract provided as context. You are truthful and never lie. Never make up facts and if you are not 100 percent sure, reply with why you cannot answer in a truthful way. Give a detailed answer to the question using information from the provided contexts.
    Context : {context}
    Question : {question}
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm

    return chain.invoke({"context": context, "question": question})

# Depending on the choice, display the appropriate content
#if choice == "new":
 #   new_func()
if choice == "Analysis":
    display_analysis()
elif choice == "Q&A":
    #option = st.checkbox('All Data')

    #if option:
     #   display_qna()
    #else:
    display_qna1()


from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import plotly.graph_objects as go
from gensim.models import FastText
import streamlit as st
import PyPDF2
import spacy
import re


def google_API():
    GOOGLE_API_KEY="GOOGLE_API_KEY"
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }
    genai.configure(api_key=GOOGLE_API_KEY)
    AImodel = genai.GenerativeModel("models/gemini-1.5-pro",generation_config=generation_config)
    return AImodel

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def preprocess(data):
    nlp = spacy.load("en_core_web_lg",disable=["ner","parser","attribute_ruler"])
    data = data.lower()
    data = re.sub(r"[^0-9a-zA-Z\+]+", " ",data)
    doc = nlp(data)
    tokens = [token for token in doc]
    filtered_tokens = [token for token in tokens if not token.is_stop]
    return filtered_tokens

def candidate_personal_info(text, AImodel):
    query = f"""Task: get the name, phone number, email id of candidate from following resume data and return it in dictionary format with keys as "name", "phone_number" and "email_id" respectively.Don't provide any code and other infomation. data: {text[:50]}"""
    response = AImodel.generate_content(query)
    per_info = re.sub(r"\n", " ",response.text)
    per_info = re.findall(r"""(?<=:\s")[^"]*(?=")|(?<=:\s)\d+""",per_info)
    print(per_info)
    return per_info
    
def Candidate_skills(text,AImodel):
    query = f"""Task: get the programming languages, librarires and frameworks, Technologies and skills from given data of a resume about a person and combine those all categories values in a list and give in the format Like ["value1","value2",...]. Don't provide any code and other infomation. Data: {text}"""
    response = AImodel.generate_content(query)
    skills = re.findall(r"\"([\w -]+)\"",response.text)
    print(skills)
    return skills

def Job_skills(text,AImodel):
    query = f"""Task: List the skills needed for the job described below. Combine all skill categories into a Python list and present them accordingly. Job Description: {text}"""
    response = AImodel.generate_content(query)
    skills_required = re.findall(r"\"([\w -]+)\"",response.text)
    print(skills_required)
    return skills_required

def create_gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'font': {'size': 24, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "green", 'thickness': 0.4},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 100], 'color': 'rgba(0,0,0,0)'}],
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.75,
                'value': 100
            }
        }
    ))
    
    # Customize the gauge
    fig.update_layout(
        paper_bgcolor="rgba(14, 17, 23, 1)",
        plot_bgcolor="rgba(14, 17, 23, 1)",
        font={'color': "white", 'family': "Arial"},
        height=230,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    # Add static percentage texts
    fig.add_annotation(text="0%", x=0.2, y=0.1, showarrow=False, font=dict(size=14, color="white"), xanchor="center")
    fig.add_annotation(text="100%", x=0.8, y=0.1, showarrow=False, font=dict(size=14, color="white"), xanchor="center")
    
    # Add score in the center
    fig.add_annotation(
        text=f"{score}%",
        x=0.5,
        y=0.3,
        showarrow=False,
        font=dict(size=30, color="white"),
        xanchor="center",
        yanchor="middle"
    )
    
    return fig


st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
        font-weight: bold;
        text-align: center;
    }
    .personal-info{
        font-size:30px !important;
        font-weight: bold;            
    }
    .quote {
        font-size:20px;
        font-style: italic;
        text-align: center;
        color: #556;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)


st.markdown('<p class="big-font">ATS System</p>', unsafe_allow_html=True)
st.markdown('<p class="quote">"Tailor Your Talent, Track Your Success"</p>', unsafe_allow_html=True)

# Create an input box for the prompt
job_desc_prompt = st.text_area("Enter Job Description:",height=150)
print(job_desc_prompt)

# Create a file uploader for PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Initialize a session state variable to track if the file has been newly uploaded
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

if uploaded_file is not None:
    # Check if this is a new file upload
    if not st.session_state.file_uploaded:
        st.success("File added successfully!")
        st.session_state.file_uploaded = True
    
    # Read the PDF and extract text
    pdf_text = extract_text_from_pdf(uploaded_file)
    cleaned_pdf_text = preprocess(pdf_text)
    print(cleaned_pdf_text)

    #calling google API
    AImodel = google_API()
    
    if st.button("Check Score"):
        #loading vec_model
        vec_model = FastText.load("E:\\Nikhil\\VScode\\ML\\models\\ATS_model2.vec")

        #extracting candidate personal info 
        per_info = candidate_personal_info(cleaned_pdf_text,AImodel)
            
        st.markdown('<p class="personal-info">Candidate Personal Information:</p>', unsafe_allow_html=True)
        st.text(f"Name: {per_info[0]}")
        st.text(f"Phone Number: {per_info[1]}")
        st.text(f"Mail: {per_info[2]}")

        st.markdown('<p class="personal-info">Your CV Score:</p>', unsafe_allow_html=True)
        
        #extracting candidate skills
        cand_skills = Candidate_skills(cleaned_pdf_text,AImodel)

        #extracting job skills
        job_skills = Job_skills(job_desc_prompt,AImodel)

        #CV score
        cand_skill_vec = vec_model.wv.get_mean_vector(cand_skills).reshape(1, -1)
        job_skill_vec = vec_model.wv.get_mean_vector(job_skills).reshape(1, -1)

        CV_score = (cosine_similarity(cand_skill_vec,job_skill_vec)[0][0]*100).round(2)
        print(CV_score)

        st.plotly_chart(create_gauge_chart(CV_score), use_container_width=True)



else:
    # Reset the file_uploaded state when no file is uploaded
    st.session_state.file_uploaded = False
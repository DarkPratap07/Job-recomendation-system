from flask import Flask, render_template, request
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import folium
from geopy.geocoders import Nominatim

app = Flask(__name__)

# ===========================
# Resume Parser Functionality
# ===========================
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_skills(text):
    skills_list = [
        "python", "java", "sql", "machine learning", "data analysis", "deep learning",
        "nlp", "flask", "django", "excel", "pandas", "numpy", "c++", "react", "node.js",
        "cloud", "aws", "azure", "linux", "javascript", "html", "css", "r", "git", "github"
    ]
    text = text.lower()
    extracted = [skill for skill in skills_list if skill in text]
    return extracted

# ===========================
# Job Matcher Functionality
# ===========================
def get_matching_jobs(resume_text):
    jobs = pd.read_csv("jobs.csv")
    vectorizer = TfidfVectorizer(stop_words="english")
    job_vectors = vectorizer.fit_transform(jobs["industry"].fillna(""))
    resume_vector = vectorizer.transform([resume_text])
    similarity_scores = cosine_similarity(resume_vector, job_vectors).flatten()
    jobs["similarity"] = similarity_scores
    matched_jobs = jobs[jobs["similarity"] > 0].sort_values(by="similarity", ascending=False).head(5)
    return matched_jobs

# ===========================
# Folium Map Generator
# ===========================
def generate_map(matched_jobs):
    geolocator = Nominatim(user_agent="job_recommendation_app")
    latitudes, longitudes = [], []

    # Geocode each job location
    for location in matched_jobs["location"]:
        try:
            geo_location = geolocator.geocode(location)
            if geo_location:
                latitudes.append(geo_location.latitude)
                longitudes.append(geo_location.longitude)
            else:
                latitudes.append(None)
                longitudes.append(None)
        except Exception as e:
            print(f"Geocoding error: {e}")
            latitudes.append(None)
            longitudes.append(None)

    matched_jobs["latitude"] = latitudes
    matched_jobs["longitude"] = longitudes

    # Create folium map centered on India
    job_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB positron")

    # Add job markers to the map
    for _, row in matched_jobs.dropna(subset=["latitude", "longitude"]).iterrows():
        popup_html = f"""
            <strong>{row['job_title']}</strong><br>
            <em>{row['company']}</em><br>
            <span>{row['location']}</span>
        """
        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color="blue", icon="briefcase", prefix="fa")
        ).add_to(job_map)

    return job_map._repr_html_()


# ===========================
# Flask Routes
# ===========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_resume():
    uploaded_file = request.files["resume"]
    if uploaded_file and uploaded_file.filename.endswith(".pdf"):
        resume_text = extract_text_from_pdf(uploaded_file)
        extracted_skills = extract_skills(resume_text)
        matching_jobs = get_matching_jobs(resume_text)
        folium_map_html = generate_map(matching_jobs)
        return render_template("results.html", skills=extracted_skills, jobs=matching_jobs, folium_map=folium_map_html)
    return "Please upload a valid PDF file."

if __name__ == "__main__":
    app.run(debug=True)

from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.conf import settings
import pandas as pd
import random
import os
from pbl.pdf_generator import render_to_pdf

def index(request):
    csv_path = os.path.join(settings.BASE_DIR, 'project4', 'movie_metadata.csv')
    if not os.path.exists(csv_path):
        return HttpResponse(f"Error: Dataset not found at {csv_path}", status=500)
    
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Drop rows with duplicate or missing titles, clean titles
    df = df.dropna(subset=['movie_title'])
    df['movie_title_clean'] = df['movie_title'].apply(lambda x: str(x).strip().replace('\xa0', ''))
    df = df.drop_duplicates(subset=['movie_title_clean'])
    
    # Check if we have at least 20 movies
    if len(df) < 20:
        return HttpResponse("Error: Dataset has less than 20 unique movies.", status=500)
        
    # Sample 20 unique movies
    sampled_indices = random.sample(range(len(df)), 20)
    sampled_df = df.iloc[sampled_indices]
    
    movies = []
    for _, row in sampled_df.iterrows():
        # Get values with fallbacks for missing/NaN
        title = row['movie_title_clean']
        year = int(row['title_year']) if not pd.isnull(row['title_year']) else "Unknown Year"
        director = str(row['director_name']).strip() if not pd.isnull(row['director_name']) else "Unknown Director"
        genres_list = [g.strip() for g in str(row['genres']).split('|') if g.strip()] if not pd.isnull(row['genres']) else []
        imdb = float(row['imdb_score']) if not pd.isnull(row['imdb_score']) else 0.0
        duration = int(row['duration']) if not pd.isnull(row['duration']) else "Unknown Duration"
        
        movies.append({
            'title': title,
            'year': year,
            'director': director,
            'genres': genres_list,
            'imdb_score': imdb,
            'duration': duration,
        })
        
    # Split into Design 1 (5 pairs = 10 movies) and Design 2 (10 movies)
    design1_movies = movies[:10]
    design2_movies = movies[10:20]
    
    # Format Design 1 as 5 pairs
    design1_pairs = []
    for i in range(0, 10, 2):
        design1_pairs.append({
            'pair_index': i // 2 + 1,
            'movie1': design1_movies[i],
            'movie2': design1_movies[i+1],
        })
        
    # Add content menu items exactly as in Project 3
    content_menu_items = [
        {"label": "Design 1: Pairwise Selection", "href": "#design-1-section"},
        {"label": "Design 2: Ranking Interface", "href": "#design-2-section"},
    ]
        
    context = {
        'design1_pairs': design1_pairs,
        'design2_movies': design2_movies,
        'content_menu_items': content_menu_items,
    }
    
    return render(request, "project4/index.html", context)

def report(request):
    # Print statement in Python as requested by the user
    print("\n[PYTHON LOG] Download PDF report clicked! Rendering HTML-to-PDF...")
    
    return render_to_pdf(
        template_src='project4/report_pdf.html',
        context_dict={},
        filename='movie_recommender_feature_report.pdf'
    )

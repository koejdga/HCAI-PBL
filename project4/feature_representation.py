import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class MovieFeatureMatrix:
    """
    Numeric movie feature matrix plus the metadata needed by the study UI.
    """
    movies: list
    features: pd.DataFrame
    feature_columns: list
    feature_labels: dict
    assumptions: list
    warnings: list


def print_missing_values(df, verbose=True):
    """
    Prints the number of missing values for each column in the dataframe.
    """
    if not verbose:
        return
    print("\n--- Missing Values Per Column in Dataset ---")
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        print(f"  {col}: {count}")
    print("--------------------------------------------")


def get_top_genres(df, top_n=20, verbose=True):
    """
    Extracts all genres, counts their frequencies, prints top N and discarded ones,
    and returns a list of the top N genres.
    """
    all_genres = []
    # Drop missing values (though genres column has 0 missing values in this dataset)
    genres_series = df['genres'].dropna()
    for g_str in genres_series:
        # Split piped genres and strip whitespace
        all_genres.extend([g.strip() for g in g_str.split('|') if g.strip()])
    
    genre_counts = pd.Series(all_genres).value_counts()
    
    top_genres = list(genre_counts.head(top_n).index)
    discarded_genres = list(genre_counts.tail(max(0, len(genre_counts) - top_n)).index)
    
    if verbose:
        print(f"\n--- Genre Processing (Top {top_n}) ---")
        print(f"Top {len(top_genres)} Genres Retained:")
        for idx, genre in enumerate(top_genres, 1):
            print(f"  {idx}. {genre} (count: {genre_counts[genre]})")
        print(f"\nDiscarded Genres (grouped into 'feature_genre_other'):")
        for idx, genre in enumerate(discarded_genres, 1):
            print(f"  {idx}. {genre} (count: {genre_counts[genre]})")
        print("-------------------------------------")
    
    return top_genres


def get_top_directors(df, top_n=50, verbose=True):
    """
    Extracts director counts, prints the top N, and returns a list of top N directors.
    """
    # Exclude missing values from frequency counts
    director_counts = df['director_name'].dropna().value_counts()
    
    top_directors = list(director_counts.head(top_n).index)
    
    if verbose:
        print(f"\n--- Director Processing (Top {top_n}) ---")
        print(f"Top {len(top_directors)} Directors Retained (sample of top 10):")
        for idx, director in enumerate(top_directors[:10], 1):
            print(f"  {idx}. {director} (count: {director_counts[director]})")
        print(f"Total directors retained: {len(top_directors)}")
        print("-----------------------------------------")
    
    return top_directors

def compute_scaler_params(df):
    """
    Computes min, max, and median values for normalization and imputation.
    These parameters are computed from the full dataset to ensure consistent
    scaling when transforming individual rows.
    """
    params = {}
    
    # Imputation medians
    params['median_year'] = df['title_year'].median()
    params['median_duration'] = df['duration'].median()
    
    # Fill missing values temporarily with median to compute proper range
    year_filled = df['title_year'].fillna(params['median_year'])
    params['min_year'] = year_filled.min()
    params['max_year'] = year_filled.max()
    
    params['min_imdb'] = df['imdb_score'].min()
    params['max_imdb'] = df['imdb_score'].max()
    
    duration_filled = df['duration'].fillna(params['median_duration'])
    params['min_duration'] = duration_filled.min()
    params['max_duration'] = duration_filled.max()
    
    # Log-transformed popularity parameters (handles skewness in counts)
    log_voted = np.log1p(df['num_voted_users'])
    params['min_log_voted'] = log_voted.min()
    params['max_log_voted'] = log_voted.max()
    
    log_likes = np.log1p(df['movie_facebook_likes'])
    params['min_log_likes'] = log_likes.min()
    params['max_log_likes'] = log_likes.max()
    
    log_cast = np.log1p(df['cast_total_facebook_likes'])
    params['min_log_cast'] = log_cast.min()
    params['max_log_cast'] = log_cast.max()
    
    return params

def transform_row(row, top_genres, top_directors, scaler_params, year_encoding='minmax'):
    """
    Transforms a single row of the original dataset into a feature representation dictionary.
    
    Parameters:
    - row: pandas Series or dict-like representing a single movie row.
    - top_genres: list of top N genres.
    - top_directors: list of top N directors.
    - scaler_params: dict containing scaling and imputation parameters.
    - year_encoding: 'minmax' or 'binning'.
    
    Returns:
    - A dictionary representing the feature vector.
    """
    # Convert row to dict if it is a Series for easy and consistent lookup
    if isinstance(row, pd.Series):
        r = row.to_dict()
    else:
        r = dict(row)
        
    features = {}
    
    # 1. Movie title identifier (useful for mapping predictions, not in feature vector U(x) = w^T x)
    features['movie_title'] = str(r.get('movie_title', '')).strip()
    
    # 2. Impute and encode Year
    year = r.get('title_year')
    if pd.isnull(year) or year is None:
        year = scaler_params['median_year']
    
    if year_encoding == 'minmax':
        denom = (scaler_params['max_year'] - scaler_params['min_year'])
        features['feature_year_normalized'] = (year - scaler_params['min_year']) / denom if denom != 0 else 0.0
    elif year_encoding == 'binning':
        features['feature_era_pre_1990'] = 1.0 if year < 1990 else 0.0
        features['feature_era_1990s'] = 1.0 if 1990 <= year < 2000 else 0.0
        features['feature_era_2000s'] = 1.0 if 2000 <= year < 2010 else 0.0
        features['feature_era_2010s'] = 1.0 if year >= 2010 else 0.0
    else:
        raise ValueError("year_encoding must be either 'minmax' or 'binning'")
        
    # 3. Impute and scale numerical values
    imdb = r.get('imdb_score')
    if pd.isnull(imdb) or imdb is None:
        imdb = 6.0  # reasonable default score
    denom_imdb = (scaler_params['max_imdb'] - scaler_params['min_imdb'])
    imdb_norm = (imdb - scaler_params['min_imdb']) / denom_imdb if denom_imdb != 0 else 0.0
    features['feature_imdb_score_normalized'] = imdb_norm
    
    duration = r.get('duration')
    if pd.isnull(duration) or duration is None:
        duration = scaler_params['median_duration']
    denom_dur = (scaler_params['max_duration'] - scaler_params['min_duration'])
    duration_norm = (duration - scaler_params['min_duration']) / denom_dur if denom_dur != 0 else 0.0
    features['feature_duration_normalized'] = duration_norm
    
    # Popularity (log-transformed and min-max scaled)
    voted = r.get('num_voted_users', 0)
    if pd.isnull(voted) or voted is None:
        voted = 0
    log_voted = np.log1p(voted)
    denom_voted = (scaler_params['max_log_voted'] - scaler_params['min_log_voted'])
    voted_norm = (log_voted - scaler_params['min_log_voted']) / denom_voted if denom_voted != 0 else 0.0
    features['feature_num_voted_users_normalized'] = voted_norm
    
    likes = r.get('movie_facebook_likes', 0)
    if pd.isnull(likes) or likes is None:
        likes = 0
    log_likes = np.log1p(likes)
    denom_likes = (scaler_params['max_log_likes'] - scaler_params['min_log_likes'])
    likes_norm = (log_likes - scaler_params['min_log_likes']) / denom_likes if denom_likes != 0 else 0.0
    features['feature_movie_facebook_likes_normalized'] = likes_norm
    
    cast = r.get('cast_total_facebook_likes', 0)
    if pd.isnull(cast) or cast is None:
        cast = 0
    log_cast = np.log1p(cast)
    denom_cast = (scaler_params['max_log_cast'] - scaler_params['min_log_cast'])
    cast_norm = (log_cast - scaler_params['min_log_cast']) / denom_cast if denom_cast != 0 else 0.0
    features['feature_cast_total_facebook_likes_normalized'] = cast_norm
    
    # 4. Encode Genres (Top N + Other)
    genres_val = r.get('genres', '')
    if pd.isnull(genres_val) or genres_val is None:
        genres_val = ''
    row_genres = [g.strip() for g in str(genres_val).split('|') if g.strip()]
    
    has_other_genre = False
    for genre in top_genres:
        if genre in row_genres:
            features[f'feature_genre_{genre}'] = 1.0
        else:
            features[f'feature_genre_{genre}'] = 0.0
            
    for genre in row_genres:
        if genre not in top_genres:
            has_other_genre = True
            break
    features['feature_genre_other'] = 1.0 if has_other_genre else 0.0
    
    # 5. Encode Director (Top N + Other)
    director = r.get('director_name')
    if pd.isnull(director) or director is None:
        director = ''
    director = str(director).strip()
    
    for dir_name in top_directors:
        if director == dir_name:
            features[f'feature_director_{dir_name}'] = 1.0
        else:
            features[f'feature_director_{dir_name}'] = 0.0
            
    if director != '' and director not in top_directors:
        features['feature_director_other'] = 1.0
    else:
        features['feature_director_other'] = 0.0
        
    # 6. Encode Country & Language & Content Rating (Top content ratings + Other)
    country = r.get('country', '')
    features['feature_is_usa'] = 1.0 if country == 'USA' else 0.0
    
    language = r.get('language', '')
    features['feature_is_english'] = 1.0 if language == 'English' else 0.0
    
    rating = r.get('content_rating')
    if pd.isnull(rating) or rating is None:
        rating = ''
    rating = str(rating).strip()
    for r_type in ['R', 'PG-13', 'PG', 'G']:
        features[f'feature_rating_{r_type}'] = 1.0 if rating == r_type else 0.0
    features['feature_rating_other'] = 1.0 if rating not in ['R', 'PG-13', 'PG', 'G'] else 0.0
    
    # 7. Feature Interactions
    features['feature_interact_indie_gem'] = imdb_norm * (1.0 - voted_norm)
    features['feature_interact_blockbuster'] = imdb_norm * voted_norm
    
    return features

def process_dataset(df, year_encoding='minmax', top_n_genres=21, top_n_directors=50, verbose=True):
    """
    Takes the full original dataset and returns the full modified dataset containing features.
    The original dataset is not modified.
    
    Parameters:
    - df: The original movie metadata DataFrame.
    - year_encoding: 'minmax' or 'binning' for the year column.
    - top_n_genres: number of top genres to keep.
    - top_n_directors: number of top directors to keep.
    
    Returns:
    - A modified DataFrame containing only the movie identifier and the feature representation columns.
    """
    # 1. Print missing values for the original dataset
    print_missing_values(df, verbose=verbose)
    
    # 2. Get top genres and directors
    top_genres = get_top_genres(df, top_n=top_n_genres, verbose=verbose)
    top_directors = get_top_directors(df, top_n=top_n_directors, verbose=verbose)
    
    # 3. Compute scaling and imputation parameters
    scaler_params = compute_scaler_params(df)
    
    # 4. Transform all rows
    feature_rows = []
    for idx, row in df.iterrows():
        transformed = transform_row(
            row, 
            top_genres, 
            top_directors, 
            scaler_params, 
            year_encoding=year_encoding
        )
        feature_rows.append(transformed)
        
    features_df = pd.DataFrame(feature_rows)
    return features_df


def clean_movie_title(value):
    """
    Normalizes titles from the IMDB CSV so they are stable session identifiers.
    """
    if pd.isnull(value):
        return ""
    return str(value).replace("\xa0", " ").strip()


def get_feature_columns(features_df):
    """
    Returns the numeric columns used in U(x) = w^T x.
    """
    return [col for col in features_df.columns if col.startswith("feature_")]


def humanize_feature_name(feature_name):
    """
    Converts backend feature names into short labels for explanations.
    """
    replacements = {
        "feature_year_normalized": "release year",
        "feature_imdb_score_normalized": "IMDB score",
        "feature_duration_normalized": "duration",
        "feature_num_voted_users_normalized": "viewer votes",
        "feature_movie_facebook_likes_normalized": "movie likes",
        "feature_cast_total_facebook_likes_normalized": "cast popularity",
        "feature_is_usa": "USA production",
        "feature_is_english": "English language",
        "feature_interact_indie_gem": "well-rated niche movie",
        "feature_interact_blockbuster": "well-rated popular movie",
    }
    if feature_name in replacements:
        return replacements[feature_name]
    if feature_name.startswith("feature_genre_"):
        return f"genre: {feature_name.removeprefix('feature_genre_').replace('_', ' ')}"
    if feature_name.startswith("feature_director_"):
        return f"director: {feature_name.removeprefix('feature_director_').replace('_', ' ')}"
    if feature_name.startswith("feature_rating_"):
        return f"content rating: {feature_name.removeprefix('feature_rating_').replace('_', ' ')}"
    if feature_name.startswith("feature_era_"):
        return f"era: {feature_name.removeprefix('feature_era_').replace('_', ' ')}"
    return feature_name.removeprefix("feature_").replace("_", " ")


def coerce_numeric_features(features_df, feature_columns=None):
    """
    Ensures all model features are finite numeric values.
    """
    if feature_columns is None:
        feature_columns = get_feature_columns(features_df)

    numeric_features = features_df.copy()
    numeric_features[feature_columns] = numeric_features[feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric_features[feature_columns] = numeric_features[feature_columns].replace([np.inf, -np.inf], np.nan)
    numeric_features[feature_columns] = numeric_features[feature_columns].fillna(0.0)
    return numeric_features


def movie_metadata_from_row(row, movie_id):
    """
    Builds the small movie object shared by the study UI and recommendation output.
    """
    genres = []
    if not pd.isnull(row.get("genres")):
        genres = [genre.strip() for genre in str(row.get("genres")).split("|") if genre.strip()]

    return {
        "movie_id": movie_id,
        "title": movie_id,
        "year": int(row["title_year"]) if not pd.isnull(row.get("title_year")) else "Unknown Year",
        "director": str(row["director_name"]).strip() if not pd.isnull(row.get("director_name")) else "Unknown Director",
        "genres": genres,
        "imdb_score": float(row["imdb_score"]) if not pd.isnull(row.get("imdb_score")) else 0.0,
        "duration": int(row["duration"]) if not pd.isnull(row.get("duration")) else "Unknown Duration",
    }


def build_movie_feature_matrix(df, year_encoding="minmax", top_n_genres=21, top_n_directors=50):
    """
    Cleans the IMDB movie metadata and returns numeric vectors plus UI metadata.
    """
    source_count = len(df)
    working_df = df.copy()
    working_df["movie_id"] = working_df["movie_title"].apply(clean_movie_title)
    working_df = working_df[working_df["movie_id"] != ""]
    after_title_drop = len(working_df)
    working_df = working_df.drop_duplicates(subset=["movie_id"], keep="first")

    features_df = process_dataset(
        working_df,
        year_encoding=year_encoding,
        top_n_genres=top_n_genres,
        top_n_directors=top_n_directors,
        verbose=False,
    )
    features_df["movie_id"] = working_df["movie_id"].values
    features_df["movie_title"] = working_df["movie_id"].values

    feature_columns = get_feature_columns(features_df)
    features_df = coerce_numeric_features(features_df, feature_columns)

    movies = [
        movie_metadata_from_row(row, movie_id)
        for movie_id, (_, row) in zip(features_df["movie_id"].tolist(), working_df.iterrows())
    ]

    warnings = []
    missing_title_count = source_count - after_title_drop
    duplicate_count = after_title_drop - len(working_df)
    if missing_title_count:
        warnings.append(f"{missing_title_count} rows without movie titles were removed.")
    if duplicate_count:
        warnings.append(f"{duplicate_count} duplicate movie titles were removed.")
    if not feature_columns:
        warnings.append("No usable numeric feature columns were extracted.")

    assumptions = [
        "Movie titles are used as session-local identifiers.",
        "Missing numeric values are median-imputed before normalization.",
        "High-cardinality genres and directors are represented with top categories plus an 'other' fallback.",
        "Feature vectors are content-based; no personal data or historic user ratings are stored.",
    ]

    feature_labels = {name: humanize_feature_name(name) for name in feature_columns}

    return MovieFeatureMatrix(
        movies=movies,
        features=features_df,
        feature_columns=feature_columns,
        feature_labels=feature_labels,
        assumptions=assumptions,
        warnings=warnings,
    )

if __name__ == "__main__":
    # If run as a script, process the local movie_metadata.csv and print basic info
    import os
    csv_path = "./project4/movie_metadata.csv"
    if os.path.exists(csv_path):
        df_orig = pd.read_csv(csv_path)
        print("Dataset loaded successfully.")
        
        # Test default minmax year encoding
        df_features = process_dataset(df_orig, year_encoding='minmax')
        print(f"\nFinal feature DataFrame shape: {df_features.shape}")
        
        # Verify no missing values in features
        null_counts = df_features.isnull().sum().sum()
        print(f"Total missing values in features DataFrame: {null_counts}")
        
        # Print a sample movie's features
        sample = df_features.iloc[0]
        print("\nSample movie features (Avatar):")
        for k, v in list(sample.items())[:15]:
            print(f"  {k}: {v}")
        print("  ...")
    else:
        print(f"Could not find {csv_path} at active path.")

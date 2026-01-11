import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ================================
# 1. Page Config & "Premium" CSS
# ================================
st.set_page_config(
    page_title="Nexus | Universal Game Mind",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for that "High-End App" look
st.markdown("""
<style>
    /* Dark Space Background */
    .stApp {
        background: linear-gradient(to bottom right, #0f0c29, #302b63, #24243e);
        color: white;
    }

    /* Glassmorphism Card Style */
    .game-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .game-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.5);
        border-color: #a855f7; /* Purple Glow */
    }

    /* Typography */
    .card-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 5px;
        color: #f3f4f6;
    }
    
    .card-meta {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-bottom: 12px;
        font-style: italic;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-right: 8px;
    }
    
    .badge-video {
        background: linear-gradient(45deg, #2563eb, #06b6d4);
        color: white;
        box-shadow: 0 0 10px rgba(6, 182, 212, 0.4);
    }
    
    .badge-board {
        background: linear-gradient(45deg, #d97706, #f59e0b);
        color: white;
        box-shadow: 0 0 10px rgba(245, 158, 11, 0.4);
    }
    
    .match-score {
        float: right;
        background-color: #10b981;
        color: #064e3b;
        font-weight: bold;
        padding: 2px 8px;
        border-radius: 8px;
        font-size: 0.8rem;
    }
    
    /* Hero Header */
    .hero-container {
        text-align: center;
        padding: 40px 20px;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 20px;
        margin-bottom: 30px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(eee, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# 2. Universal Data Loader (Strict Mode)
# ================================
@st.cache_resource
def load_data_strict():
    master_list = []
    status_log = []

    # --- Load Video Games ---
    try:
        vg_df = pd.read_csv("merged_data.csv", on_bad_lines='warn')
        vg_df = vg_df[['Title', 'Popular Tags', 'Game Description']].copy()
        vg_df.rename(columns={'Popular Tags': 'Tags', 'Game Description': 'Description'}, inplace=True)
        vg_df['Type'] = 'Video Game'
        master_list.append(vg_df)
        status_log.append("✅ Video Games Loaded")
    except FileNotFoundError:
        status_log.append("❌ Video Games (merged_data.csv) Missing")

    # --- Load Board Games ---
    try:
        bgg_df = pd.read_csv("BGG_Data_Set.csv", encoding="ISO-8859-1", on_bad_lines='skip')
        # Combine Mechanics + Domains for robust tags
        bgg_df['Tags'] = bgg_df['Mechanics'].fillna('') + ", " + bgg_df['Domains'].fillna('')
        bgg_df = bgg_df[['Name', 'Tags']].copy()
        bgg_df.rename(columns={'Name': 'Title'}, inplace=True)
        # Create synthetic description
        bgg_df['Description'] = "Tabletop experience featuring: " + bgg_df['Tags']
        bgg_df['Type'] = 'Board Game'
        master_list.append(bgg_df)
        status_log.append("✅ Board Games Loaded")
    except FileNotFoundError:
        status_log.append("❌ Board Games (BGG_Data_Set.csv) Missing")

    if not master_list:
        return None, None, None, status_log

    # --- Merge ---
    df = pd.concat(master_list, ignore_index=True)
    df['Tags'] = df['Tags'].fillna('Unknown')
    df['Description'] = df['Description'].fillna('')
    
    # Combined features for AI
    df['combined_features'] = df['Tags'] + " " + df['Description']
    
    # Vectorize
    tfidf = TfidfVectorizer(stop_words='english', dtype='float32')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    # Index Map
    df.drop_duplicates(subset='Title', keep='first', inplace=True)
    indices = pd.Series(df.index, index=df['Title']).drop_duplicates()
    
    return df, tfidf_matrix, indices, status_log

# Load Data
df, tfidf_matrix, indices, status_log = load_data_strict()

# ================================
# 3. Recommendation Logic
# ================================
def get_recommendations(title, filter_type='All', top_n=10):
    if title not in indices:
        return []
    
    idx = indices[title]
    
    # Calculate Similarity
    cosine_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix)
    sim_scores = list(enumerate(cosine_scores[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Grab top 100 to allow for filtering
    sim_scores = sim_scores[1:101] 
    game_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores] # Keep the score value
    
    # Fetch rows
    results = df.iloc[game_indices].copy()
    results['Similarity'] = scores # Add score to dataframe
    
    # Filter
    if filter_type == 'Video Games Only':
        results = results[results['Type'] == 'Video Game']
    elif filter_type == 'Board Games Only':
        results = results[results['Type'] == 'Board Game']
        
    return results.head(top_n)

# ================================
# 4. Streamlit UI
# ================================

# --- Sidebar ---
if df is not None:
    with st.sidebar:
        st.header("🤖 Nexus Settings")
        st.markdown("---")
        
        # Search
        all_titles = sorted(df['Title'].astype(str).unique())
        selected_game = st.selectbox("🎯 Select a Game:", all_titles)
        
        # Filter
        filter_option = st.radio("Show me:", ["All", "Video Games Only", "Board Games Only"])
        
        num_recs = st.slider("Quantity:", 4, 12, 6)
        
        st.markdown("---")
        st.markdown("**System Status:**")
        for status in status_log:
            if "✅" in status:
                st.success(status)
            else:
                st.error(status)
        
        if st.button("🚀 Launch Analysis", type="primary", use_container_width=True):
            st.session_state['show_results'] = True

    # --- Main Area ---
    
    # Hero Header
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">UNIVERSAL GAME NEXUS</h1>
        <p style="font-size: 1.2rem; color: #d1d5db;">
            Bridging the gap between Digital and Tabletop realities.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if 'show_results' in st.session_state and st.session_state['show_results']:
        st.markdown(f"### 🧬 Analyzing DNA of: *{selected_game}*")
        st.markdown("---")
        
        with st.spinner('Calculating vector alignments...'):
            results = get_recommendations(selected_game, filter_option, num_recs)
        
        if not results.empty:
            cols = st.columns(2)
            
            for i, (_, row) in enumerate(results.iterrows()):
                col = cols[i % 2]
                
                # Dynamic Logic
                badge_class = "badge-video" if row['Type'] == 'Video Game' else "badge-board"
                icon = "💻" if row['Type'] == 'Video Game' else "🎲"
                match_score = int(row['Similarity'] * 100)
                
                # Tags cleanup
                tags = str(row['Tags']).split(',')[:4]
                tags_str = ", ".join(tags)
                
                with col:
                    st.markdown(f"""
                    <div class="game-card">
                        <div>
                            <span class="badge {badge_class}">{icon} {row['Type']}</span>
                            <span class="match-score">{match_score}% Match</span>
                        </div>
                        <div class="card-title" style="margin-top:10px;">{i+1}. {row['Title']}</div>
                        <div class="card-meta">Tags: {tags_str}</div>
                        <p style="font-size: 0.9rem; color: #e5e7eb; line-height: 1.5;">
                            {str(row['Description'])[:140]}...
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No matches found within the current multiverse parameters.")
else:
    st.error("CRITICAL ERROR: No data found. Please place CSV files in the directory.")
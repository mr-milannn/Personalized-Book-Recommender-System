import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from collections import defaultdict
import re
from models import CollaborativeFilteringSVD
from models import ContentBasedFiltering
from models import HybridRecommenderSystem

# Page configuration
st.set_page_config(
    page_title="📚 BookWise - Personalized Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .book-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .metric-card {
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .sidebar-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1e88e5;
        color: #1e3d59;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_models():
    """Load the trained models and data"""
    try:
        with open('recommender_models.pkl', 'rb') as f:
            models_data = pickle.load(f)
        return (
            models_data['cf_model'],
            models_data['cb_model'],
            models_data['hybrid_model'],
            models_data['books_df']
        )
    except FileNotFoundError:
        st.error("❌ Model file 'recommender_models.pkl' not found! Please ensure you've trained and saved your models.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

# Load models
cf_model, cb_model, hybrid_model, books_df = load_models()

def display_book_card(book, show_score=False, score=None):
    """Display a book in a nice card format"""
    with st.container():
        col1, col2 = st.columns([1, 4])
        
        with col1:
            # Placeholder for book cover (you could add actual covers later)
            generic_image = "https://cdn-icons-png.flaticon.com/512/29/29302.png"
            st.image(generic_image, width=100)
        
        with col2:
            st.markdown(f"**{book.get('title', 'Unknown Title')}**")
            st.markdown(f"*by {book.get('authors', 'Unknown Author')}*")
            
            if 'average_rating' in book and book['average_rating']:
                rating = float(book['average_rating'])
                stars = "⭐" * int(rating) + "☆" * (5 - int(rating))
                st.markdown(f"{stars} {rating:.1f}/5.0")
            
            if show_score and score:
                st.markdown(f"🎯 **Recommendation Score:** {score:.2f}")
                
            if 'original_publication_year' in book and book['original_publication_year']:
                st.markdown(f"📅 Published: {int(book['original_publication_year'])}")
        
        st.markdown("---")

def search_books(query, books_df, n_results=10):
    """Search books by title or author"""
    if not query:
        return []
    
    query = query.lower()
    mask = (
        books_df['title'].str.lower().str.contains(query, na=False, regex=False) |
        books_df['authors'].str.lower().str.contains(query, na=False, regex=False)
    )
    
    results = books_df[mask].head(n_results)
    return results.to_dict('records')

def get_trending_books(books_df, n_books=10):
    """Get trending books based on ratings"""
    # Sort by average rating and filter books with decent ratings
    trending = books_df[
        (books_df['average_rating'] >= 4.0) & 
        (books_df['ratings_count'] >= 100)
    ].nlargest(n_books, 'average_rating')
    
    return trending.to_dict('records')

def get_random_books(books_df, n_books=5):
    """Get random book recommendations"""
    return books_df.sample(n_books).to_dict('records')

def predict_user_rating(user_id, book_id, cf_model):
    """Predict rating for a user-book pair"""
    try:
        rating = cf_model.predict_rating(user_id, book_id)
        return max(1.0, min(5.0, rating))
    except:
        return 3.0

# Sidebar
with st.sidebar:
    st.markdown("## 🎛️ Navigation")
    
    page = st.selectbox(
        "Choose a feature:",
        [
            "🏠 Home",
            "👤 User Recommendations", 
            "🔍 Search Books",
            "📖 Similar Books",
            "🏷️ Tag-based Discovery",
            "🔥 Trending Books",
            "🎲 Random Discovery",
            "🎯 Rating Predictor",
            "📊 Analytics Dashboard"
        ]
    )
    
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-info">
    <h4>💡 How it works</h4>
    <p>This app uses machine learning to recommend books based on:</p>
    <ul>
    <li>🤝 Collaborative Filtering</li>
    <li>📝 Content-based Filtering</li>
    <li>🔄 Hybrid Approach</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📈 Dataset Stats")
    st.metric("Total Books", f"{len(books_df):,}")
    st.metric("Unique Authors", f"{books_df['authors'].nunique():,}")
    if 'user_mapping' in dir(cf_model):
        st.metric("Total Users", f"{len(cf_model.user_mapping):,}")

# Main content area
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">📚 BookWise</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">Your Personalized Book Recommendation System</h3>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
        <h3>🤝 Collaborative Filtering</h3>
        <p>Get recommendations based on users with similar reading preferences</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
        <h3>📝 Content-Based</h3>
        <p>Discover books similar to ones you've enjoyed based on content features</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
        <h3>🔄 Hybrid System</h3>
        <p>Best of both worlds - combining multiple recommendation techniques</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("## 📊 Quick Stats")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_rating = books_df['average_rating'].mean()
        st.markdown(f"""
        <div class="metric-card">
        <h3>{avg_rating:.2f}</h3>
        <p>Average Rating</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        top_year = books_df['original_publication_year'].mode().iloc[0]
        st.markdown(f"""
        <div class="metric-card">
        <h3>{int(top_year)}</h3>
        <p>Most Popular Year</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        max_ratings = books_df['ratings_count'].max()
        st.markdown(f"""
        <div class="metric-card">
        <h3>{int(max_ratings):,}</h3>
        <p>Most Rated Book</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        high_rated = len(books_df[books_df['average_rating'] >= 4.0])
        st.markdown(f"""
        <div class="metric-card">
        <h3>{high_rated:,}</h3>
        <p>Books Rated 4.0+</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample recommendations
    st.markdown("---")
    st.markdown("## 🎲 Sample Recommendations")
    sample_books = get_random_books(books_df, 3)
    
    for book in sample_books:
        display_book_card(book)

elif page == "👤 User Recommendations":
    st.markdown("# 👤 Personalized Recommendations")
    st.markdown("Get book recommendations based on a user's reading history")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if hasattr(cf_model, 'user_mapping') and cf_model.user_mapping:
            available_users = list(cf_model.user_mapping.keys())
            user_id = st.selectbox(
                "Select User ID:",
                options=available_users[:100],  # Limit for performance
                help="Choose a user ID to get personalized recommendations"
            )
            
            n_recs = st.slider("Number of recommendations:", 1, 20, 10)
            
            if st.button("Get Recommendations", type="primary"):
                with st.spinner("🔍 Finding perfect books for you..."):
                    try:
                        recommendations = hybrid_model.recommend_books(
                            user_id=user_id, 
                            n_recommendations=n_recs
                        )
                        
                        if recommendations:
                            st.success(f"Found {len(recommendations)} recommendations!")
                            
                            with col2:
                                st.markdown(f"### 📚 Recommendations for User {user_id}")
                                for i, rec in enumerate(recommendations, 1):
                                    st.markdown(f"#### {i}. Recommendation")
                                    display_book_card(
                                        rec, 
                                        show_score=True, 
                                        score=rec.get('hybrid_score', 0)
                                    )
                        else:
                            st.warning("No recommendations found for this user.")
                            
                    except Exception as e:
                        st.error(f"Error getting recommendations: {str(e)}")
        else:
            st.warning("No user data available in the collaborative filtering model.")

elif page == "🔍 Search Books":
    st.markdown("# 🔍 Search Books")
    st.markdown("Search for books by title or author")
    
    search_query = st.text_input(
        "Enter book title or author name:",
        placeholder="e.g., Harry Potter, Stephen King, Pride and Prejudice"
    )
    
    if search_query:
        with st.spinner("🔍 Searching..."):
            results = search_books(search_query, books_df, 20)
            
            if results:
                st.success(f"Found {len(results)} results for '{search_query}'")
                
                for book in results:
                    display_book_card(book)
            else:
                st.warning(f"No books found for '{search_query}'. Try different keywords!")
    
    # Popular searches suggestion
    if not search_query:
        st.markdown("### 💡 Try searching for:")
        suggestions = ["Harry Potter", "Stephen King", "Jane Austen", "Science Fiction", "Romance"]
        cols = st.columns(len(suggestions))
        
        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                if st.button(suggestion):
                    st.experimental_rerun()

elif page == "📖 Similar Books":
    st.markdown("# 📖 Find Similar Books")
    st.markdown("Discover books similar to ones you've enjoyed")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        book_search = st.text_input(
            "Search for a book:",
            placeholder="Enter book title"
        )
        
        if book_search:
            search_results = search_books(book_search, books_df, 10)
            
            if search_results:
                selected_book_title = st.selectbox(
                    "Select a book:",
                    options=[book['title'] for book in search_results]
                )
                
                selected_book = next(book for book in search_results if book['title'] == selected_book_title)
                
                n_similar = st.slider("Number of similar books:", 1, 15, 8)
                
                if st.button("Find Similar Books", type="primary"):
                    with st.spinner("🔍 Finding similar books..."):
                        try:
                            similar_books = cb_model.get_similar_books(
                                selected_book['book_id'], 
                                n_similar
                            )
                            
                            if similar_books:
                                with col2:
                                    st.markdown(f"### 📚 Books similar to '{selected_book_title}'")
                                    
                                    # Show selected book first
                                    st.markdown("#### 🎯 Selected Book:")
                                    display_book_card(selected_book)
                                    
                                    st.markdown("#### 📖 Similar Books:")
                                    for book in similar_books:
                                        display_book_card(book)
                            else:
                                st.warning("No similar books found.")
                                
                        except Exception as e:
                            st.error(f"Error finding similar books: {str(e)}")

elif page == "🏷️ Tag-based Discovery":
    st.markdown("# 🏷️ Discover Books by Tags")
    st.markdown("Explore books based on genres, themes, and tags")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎯 Enter Tags/Genres")
        
        # Predefined popular tags
        popular_tags = [
            "fiction", "fantasy", "romance", "mystery", "science-fiction",
            "historical-fiction", "thriller", "young-adult", "non-fiction",
            "biography", "horror", "adventure", "comedy", "drama"
        ]
        
        # Tag selection
        selected_tags = st.multiselect(
            "Choose from popular tags:",
            options=popular_tags,
            default=[]
        )
        
        custom_tags = st.text_input(
            "Or enter custom tags (comma-separated):",
            placeholder="e.g., space opera, dragons, magic"
        )
        
        # Combine tags
        all_tags = selected_tags.copy()
        if custom_tags:
            all_tags.extend([tag.strip() for tag in custom_tags.split(',')])
        
        tag_string = ' '.join(all_tags)
        
        n_recs = st.slider("Number of recommendations:", 1, 20, 10)
        
        if st.button("Discover Books", type="primary") and tag_string:
            with st.spinner("🔍 Finding books matching your interests..."):
                try:
                    recommendations = cb_model.recommend_by_tags(tag_string, n_recs)
                    
                    if recommendations:
                        with col2:
                            st.markdown(f"### 📚 Books for tags: {', '.join(all_tags)}")
                            
                            for book in recommendations:
                                display_book_card(book)
                                if 'tags' in book and book['tags']:
                                    st.markdown(f"🏷️ **Tags:** {book['tags'][:100]}...")
                                st.markdown("---")
                    else:
                        st.warning("No books found for the specified tags.")
                        
                except Exception as e:
                    st.error(f"Error finding books: {str(e)}")
    
    if not tag_string:
        with col2:
            st.markdown("### 💡 How to use:")
            st.markdown("""
            1. Select from popular tags or enter your own
            2. Combine multiple tags for better results
            3. Click 'Discover Books' to find matches
            
            **Example combinations:**
            - fantasy + magic + dragons
            - historical fiction + romance
            - science fiction + space opera
            """)

elif page == "🔥 Trending Books":
    st.markdown("# 🔥 Trending Books")
    st.markdown("Discover popular and highly-rated books")
    
    tab1, tab2, tab3 = st.tabs(["📈 Highest Rated", "🔥 Most Popular", "📅 Recent Releases"])
    
    with tab1:
        st.markdown("### ⭐ Highest Rated Books")
        trending_books = get_trending_books(books_df, 15)
        
        if trending_books:
            for book in trending_books:
                display_book_card(book)
        else:
            st.warning("No trending books data available.")
    
    with tab2:
        st.markdown("### 📊 Most Rated Books")
        most_rated = books_df.nlargest(15, 'ratings_count')
        
        for _, book in most_rated.iterrows():
            display_book_card(book.to_dict())
    
    with tab3:
        st.markdown("### 📅 Recent Publications")
        recent_books = books_df[
            books_df['original_publication_year'] >= 2000
        ].nlargest(15, 'original_publication_year')
        
        for _, book in recent_books.iterrows():
            display_book_card(book.to_dict())

elif page == "🎲 Random Discovery":
    st.markdown("# 🎲 Random Book Discovery")
    st.markdown("Serendipitous book recommendations for adventurous readers")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🎲 Discover Random Books", type="primary", use_container_width=True):
            with st.spinner("🎲 Finding random gems..."):
                random_books = get_random_books(books_df, 8)
                
                st.markdown("### 🎯 Your Random Discoveries")
                for book in random_books:
                    display_book_card(book)
        
        st.markdown("---")
        st.markdown("### 🎯 Filtered Random Discovery")
        
        # Filters for random discovery
        min_rating = st.slider("Minimum rating:", 1.0, 5.0, 3.5, 0.1)
        min_year = st.slider("Published after year:", 1800, 2023, 1990)
        
        if st.button("🎲 Discover with Filters", use_container_width=True):
            with st.spinner("🔍 Finding filtered random books..."):
                filtered_books = books_df[
                    (books_df['average_rating'] >= min_rating) &
                    (books_df['original_publication_year'] >= min_year)
                ]
                
                if len(filtered_books) > 0:
                    random_filtered = filtered_books.sample(min(8, len(filtered_books)))
                    
                    st.markdown("### 🎯 Your Filtered Random Discoveries")
                    for _, book in random_filtered.iterrows():
                        display_book_card(book.to_dict())
                else:
                    st.warning("No books match your criteria. Try adjusting the filters.")

elif page == "🎯 Rating Predictor":
    st.markdown("# 🎯 Rating Predictor")
    st.markdown("Predict how much a user might like a specific book")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if hasattr(cf_model, 'user_mapping') and cf_model.user_mapping:
            # User selection
            available_users = list(cf_model.user_mapping.keys())
            selected_user = st.selectbox(
                "Select User ID:",
                options=available_users[:50],  # Limit for performance
            )
            
            # Book search
            book_search = st.text_input("Search for a book:", placeholder="Enter book title")
            
            if book_search:
                search_results = search_books(book_search, books_df, 10)
                
                if search_results:
                    selected_book_title = st.selectbox(
                        "Select a book:",
                        options=[book['title'] for book in search_results]
                    )
                    
                    selected_book = next(book for book in search_results if book['title'] == selected_book_title)
                    
                    if st.button("Predict Rating", type="primary"):
                        with st.spinner("🔮 Predicting rating..."):
                            predicted_rating = predict_user_rating(
                                selected_user, 
                                selected_book['book_id'], 
                                cf_model
                            )
                            
                            with col2:
                                st.markdown("### 🎯 Prediction Result")
                                
                                # Display book info
                                display_book_card(selected_book)
                                
                                # Show prediction
                                stars = "⭐" * int(predicted_rating) + "☆" * (5 - int(predicted_rating))
                                st.markdown(f"""
                                <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin: 1rem 0;">
                                <h2>Predicted Rating</h2>
                                <h1>{stars}</h1>
                                <h2>{predicted_rating:.1f}/5.0</h2>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Interpretation
                                if predicted_rating >= 4.0:
                                    st.success("🎉 This user will likely love this book!")
                                elif predicted_rating >= 3.0:
                                    st.info("👍 This user will probably enjoy this book.")
                                else:
                                    st.warning("🤔 This book might not be the best fit for this user.")
        else:
            st.warning("Rating prediction requires collaborative filtering data.")

elif page == "📊 Analytics Dashboard":
    st.markdown("# 📊 Analytics Dashboard")
    st.markdown("Insights and statistics about the book dataset")
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Books", f"{len(books_df):,}")
    with col2:
        st.metric("Avg Rating", f"{books_df['average_rating'].mean():.2f}")
    with col3:
        st.metric("Unique Authors", f"{books_df['authors'].nunique():,}")
    with col4:
        if hasattr(cf_model, 'user_mapping'):
            st.metric("Users in System", f"{len(cf_model.user_mapping):,}")
        else:
            st.metric("Year Range", f"{int(books_df['original_publication_year'].min())}-{int(books_df['original_publication_year'].max())}")
    
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["📈 Ratings", "📅 Publications", "👥 Popular Books"])
    
    with tab1:
        # Rating distribution
        fig = px.histogram(
            books_df, 
            x='average_rating', 
            nbins=30,
            title="Distribution of Average Book Ratings",
            labels={'average_rating': 'Average Rating', 'count': 'Number of Books'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Ratings count vs average rating scatter
        sample_books = books_df.sample(min(1000, len(books_df)))  # Sample for performance
        fig2 = px.scatter(
            sample_books,
            x='ratings_count',
            y='average_rating',
            title="Ratings Count vs Average Rating",
            labels={'ratings_count': 'Number of Ratings', 'average_rating': 'Average Rating'},
            opacity=0.6
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        # Publications by year
        year_counts = books_df['original_publication_year'].value_counts().sort_index()
        year_counts_recent = year_counts[year_counts.index >= 1950]
        
        fig = px.line(
            x=year_counts_recent.index,
            y=year_counts_recent.values,
            title="Books Published by Year (1950+)",
            labels={'x': 'Year', 'y': 'Number of Books'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top publication years
        top_years = books_df['original_publication_year'].value_counts().head(10)
        fig2 = px.bar(
            x=top_years.values,
            y=[str(int(year)) for year in top_years.index],
            orientation='h',
            title="Top 10 Most Productive Years",
            labels={'x': 'Number of Books', 'y': 'Year'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        # Most rated books
        top_rated_books = books_df.nlargest(15, 'ratings_count')
        
        fig = px.bar(
            top_rated_books,
            x='ratings_count',
            y=[title[:30] + '...' if len(title) > 30 else title for title in top_rated_books['title']],
            orientation='h',
            title="Most Rated Books",
            labels={'ratings_count': 'Number of Ratings', 'y': 'Book Title'}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top authors by number of books
        author_counts = books_df['authors'].value_counts().head(10)
        fig2 = px.bar(
            x=author_counts.values,
            y=author_counts.index,
            orientation='h',
            title="Most Prolific Authors",
            labels={'x': 'Number of Books', 'y': 'Author'}
        )
        st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
📚 BookWise - Powered by Machine Learning | Built with Streamlit<br>
<small>Discover your next favorite book with personalized recommendations</small>
</div>
""", unsafe_allow_html=True)
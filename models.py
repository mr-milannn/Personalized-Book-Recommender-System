import numpy as np
from sklearn.decomposition import TruncatedSVD

class CollaborativeFilteringSVD:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_item_matrix = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
    def prepare_matrix(self, ratings_df):
        unique_users = ratings_df['user_id'].unique()
        unique_items = ratings_df['book_id'].unique()
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        user_item_matrix = np.zeros((n_users, n_items))
        
        for _, row in ratings_df.iterrows():
            user_idx = self.user_mapping[row['user_id']]
            item_idx = self.item_mapping[row['book_id']]
            user_item_matrix[user_idx, item_idx] = row['rating']
        
        self.user_item_matrix = user_item_matrix
        return user_item_matrix
    
    def fit(self, ratings_df):
        print("Preparing user-item matrix...")
        matrix = self.prepare_matrix(ratings_df)
        
        print("Fitting SVD model...")
        self.svd.fit(matrix)
        
        print(f"SVD model fitted with {self.n_components} components")
        print(f"Explained variance ratio: {self.svd.explained_variance_ratio_.sum():.3f}")
        
    def predict_rating(self, user_id, book_id):
        if user_id not in self.user_mapping or book_id not in self.item_mapping:
            return 3.0
        
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[book_id]
        
        user_factors = self.svd.transform(self.user_item_matrix[user_idx:user_idx+1])
        item_factors = self.svd.components_[:, item_idx]
        
        predicted_rating = np.dot(user_factors[0], item_factors)
        return max(1, min(5, predicted_rating))
    
    def recommend_books(self, user_id, books_df, n_recommendations=10):
        if user_id not in self.user_mapping:
            popular_books = books_df.sample(n_recommendations)
            return popular_books[['book_id', 'title', 'authors']].to_dict('records')
        
        user_idx = self.user_mapping[user_id]
        user_ratings = self.user_item_matrix[user_idx]
        unrated_books = np.where(user_ratings == 0)[0]
        
        predictions = []
        for item_idx in unrated_books:
            book_id = self.reverse_item_mapping[item_idx]
            pred_rating = self.predict_rating(user_id, book_id)
            predictions.append((book_id, pred_rating))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_book_ids = [book_id for book_id, _ in predictions[:n_recommendations]]
        
        recommended_books = books_df[books_df['book_id'].isin(top_book_ids)]
        return recommended_books[['book_id', 'title', 'authors']].to_dict('records')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentBasedFiltering:
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.books_df = None

    def fit(self, books_df):
        """Fit the TF-IDF model"""
        self.books_df = books_df.copy()
        print("Fitting TF-IDF model...")
        self.tfidf_matrix = self.tfidf.fit_transform(books_df['combined_features'].fillna(''))
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    def get_similar_books(self, book_id, n_recommendations=10):
        """Get similar books based on content"""
        try:
            book_idx = self.books_df[self.books_df['book_id'] == book_id].index[0]
        except IndexError:
            return []

        book_vector = self.tfidf_matrix[book_idx]
        similarities = linear_kernel(book_vector, self.tfidf_matrix).flatten()

        similar_indices = similarities.argsort()[::-1][1:n_recommendations+1]
        similar_books = self.books_df.iloc[similar_indices]

        return similar_books[['book_id', 'title', 'authors']].to_dict('records')

    def recommend_by_tags(self, tags, n_recommendations=10):
        """Recommend books based on tags"""
        tag_vector = self.tfidf.transform([tags])
        similarities = linear_kernel(tag_vector, self.tfidf_matrix).flatten()

        top_indices = similarities.argsort()[::-1][:n_recommendations]
        recommended_books = self.books_df.iloc[top_indices]

        return recommended_books[['book_id', 'title', 'authors', 'tags']].to_dict('records')

from collections import defaultdict


class HybridRecommenderSystem:
    def __init__(self, cf_model, cb_model, cf_weight=0.6, cb_weight=0.4):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight

    def recommend_books(self, user_id=None, book_id=None, tags=None, books_df=None, n_recommendations=10):
        """
        Hybrid recommendation combining collaborative and content-based filtering.
        Requires books_df for collaborative filtering.
        """
        recommendations = []

        # Collaborative filtering recommendations
        if user_id and hasattr(self.cf_model, 'user_mapping') and user_id in self.cf_model.user_mapping:
            cf_recs = self.cf_model.recommend_books(user_id, books_df, n_recommendations * 2)
            for rec in cf_recs:
                rec['source'] = 'collaborative'
                rec['score'] = self.cf_weight
            recommendations.extend(cf_recs)

        # Content-based recommendations
        if book_id:
            cb_recs = self.cb_model.get_similar_books(book_id, n_recommendations)
            for rec in cb_recs:
                rec['source'] = 'content_based'
                rec['score'] = self.cb_weight
            recommendations.extend(cb_recs)

        # Tag-based recommendations
        if tags:
            tag_recs = self.cb_model.recommend_by_tags(tags, n_recommendations)
            for rec in tag_recs:
                rec['source'] = 'tag_based'
                rec['score'] = self.cb_weight * 0.8
            recommendations.extend(tag_recs)

        # Remove duplicates and combine scores
        book_scores = defaultdict(float)
        book_info = {}

        for rec in recommendations:
            b_id = rec['book_id']
            book_scores[b_id] += rec['score']
            if b_id not in book_info:
                book_info[b_id] = rec

        # Sort by combined score
        sorted_books = sorted(book_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top recommendations
        final_recommendations = []
        for b_id, score in sorted_books[:n_recommendations]:
            rec = book_info[b_id].copy()
            rec['hybrid_score'] = score
            final_recommendations.append(rec)

        return final_recommendations

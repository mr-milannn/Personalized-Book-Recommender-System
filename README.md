📚 Personalized Book Recommendation System

A feature-rich recommendation engine built using the Goodbooks-10K dataset, combining Collaborative Filtering, Content-Based Filtering, and a Hybrid Model to deliver personalized book recommendations. The project includes a Streamlit app for an interactive user experience and a comparative evaluation of different approaches.

🚀 Features

🔍 Search & Discover – Search books by title/author and get tailored recommendations.

🤝 Collaborative Filtering – Recommends books based on user–item interactions (SVD/NMF).

📑 Content-Based Filtering – Uses TF-IDF & cosine similarity on book metadata.

🔗 Hybrid Model – Combines collaborative & content-based for improved accuracy.

📊 Evaluation Metrics – Accuracy, Precision, Recall, and F1 Score with visualizations.

🎨 Streamlit App – Intuitive UI with charts, interactive search, and recommendation results.

🗂️ Dataset

We use the Goodbooks-10K dataset, which contains:

10,000 books

53,000 users

6 million ratings

Metadata including title, author, and genres

Goodbooks-10K Dataset

🏗️ Tech Stack

Python (Pandas, NumPy, Scikit-learn, Surprise, NMF/SVD)

Natural Language Processing (TF-IDF, Cosine Similarity)

Streamlit (UI/UX for recommendations)

Matplotlib / Seaborn / Plotly (Visualization)

📊 Models Implemented

Collaborative Filtering

Matrix factorization (SVD, NMF)

User–Item similarity

Content-Based Filtering

TF-IDF vectorization on book descriptions

Cosine similarity for recommendations

Hybrid Recommendation System

Weighted combination of collaborative & content-based results

📈 Results & Evaluation

Comparative study between models

Metrics used: Accuracy, Precision, Recall, F1 Score

Visual graphs to showcase performance

🖥️ Streamlit App Demo

Upload or search books

View top-N recommendations

Interactive charts to visualize user preferences

Real-time personalized suggestions

# Run the app
locally- streamlit run app.py
web- https://personalized-book-recommender-system.streamlit.app/


📌 Future Improvements

Add deep learning–based recommenders (Neural CF, Autoencoders)

Integration with a book API (Google Books, Goodreads)

User profiles & login system for long-term personalization

🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

📜 License

MIT License

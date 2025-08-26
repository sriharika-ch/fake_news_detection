# fake_news_detection
📰 Fake News Detection using Machine Learning:

This project is a Fake News Detection System built using Python, Scikit-learn, and Streamlit. It aims to classify news articles as either REAL or FAKE, helping to combat misinformation.

The project leverages Natural Language Processing (NLP) techniques with TF-IDF vectorization and a PassiveAggressiveClassifier model to achieve accurate classification.

⚙️ Features:

✅ Interactive Streamlit Web App: Enter any news article text and instantly check if it's REAL or FAKE.

✅ Machine Learning Pipeline: Uses TF-IDF for text feature extraction and Passive Aggressive Classifier for classification.

✅ Model Persistence: The trained model and vectorizer are stored using joblib for faster reloading.

✅ Model Performance Metrics: Evaluated using Accuracy, Confusion Matrix, and ROC Curve.

📂 Project Structure:
.
├── fake_news.ipynb          
├── fake_or_real_news.csv    
├── model.pkl               
├── vectorizer.pkl           
├── app.py                   
└── README.md                

🧠 Model Training & Evaluation:

Dataset: Fake and Real news dataset from Kaggle

Preprocessing:

Model: PassiveAggressiveClassifier

Evaluation Metrics:

Accuracy Score

Confusion Matrix

ROC Curve

📊 Results:

Achieved an accuracy of ~X% (replace with your actual result).

The confusion matrix and ROC curve show that the model performs well in distinguishing between fake and real news.

💡 Future Improvements:

Add support for deep learning models (LSTMs, Transformers).

Deploy the model on cloud platforms (Heroku/Streamlit Cloud/AWS).

Enhance dataset with more recent and diverse news sources.

Acknowledgements:

Dataset from Kaggle

Libraries: pandas, scikit-learn, streamlit, joblib, matplotlib

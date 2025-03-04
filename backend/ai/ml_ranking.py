import datetime
import math
import os
import threading
from typing import List, Dict
import psycopg2
import pandas as pd
import numpy as np
import asyncio
from sklearn.linear_model import LinearRegression
import pickle
import logging
from sqlalchemy import create_engine

# PostgreSQL Configuration
POSTGRES_DB = os.getenv("POSTGRES_DB", "lumidoc_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "lumidoc_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "lumidoc_pass")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# Database URL with psycopg driver
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
MODEL_PATH = "ranking_model.pkl"

# ✅ Create a SQLAlchemy Engine
engine = create_engine(DATABASE_URL)

class RankingModel:
    def __init__(self):
       """Initialize and load ranking model with file lock."""
       self.lock = threading.Lock()  # Ensure safe file access
       self.model = None
       self.load_model()
       
    def load_model(self):
        """Load trained ranking model safely."""
        with self.lock:  # Prevent concurrent read/write issues
            try:
                if os.path.exists(MODEL_PATH):
                    with open(MODEL_PATH, "rb") as f:
                        self.model = pickle.load(f)
                    logging.info("Ranking model loaded successfully.")
                else:
                    logging.warning("Ranking model not found. Training new model.")
                    self.train_ranking_model()
            except Exception as e:
                logging.error(f"Error loading ranking model: {str(e)}")
                self.model = None  # Prevent using a corrupted model


    def fetch_interaction_data(self):
        """Fetch user interaction data for ML model training."""
        try:
            with engine.connect() as conn:  # Use SQLAlchemy engine
                df = pd.read_sql("""
                    SELECT distinct retrieval_count, likes, dislikes, time_decay, relevance_score 
                    FROM search_interactions
                """, conn)
            return df
        except Exception as e:
            logging.error(f"Error fetching interaction data: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame if failure

    def train_ranking_model(self):
        """Train ML model for ranking adjustment with exception handling."""
        try:
            data = self.fetch_interaction_data()
            
            if data.empty:
                logging.warning("No training data available. Skipping model training.")
                return

            X = data[["retrieval_count", "likes", "dislikes", "time_decay"]]
            y = data["relevance_score"]

            model = LinearRegression()
            model.fit(X, y)

            # Safely save the model
            with self.lock:
                with open(MODEL_PATH, "wb") as f:
                    pickle.dump(model, f)
                self.model = model  # Update model in memory

            logging.info("ML Model retrained & saved!")

        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")

    def rank_results(self, results):
        """Re-rank search results using ML model."""
        if not results or not self.model:
            return results

        try:
            feature_data = pd.DataFrame(results)[["retrieval_count", "likes", "dislikes", "time_decay"]]
            predicted_scores = self.model.predict(feature_data)
            
            for i, result in enumerate(results):
                result["final_score"] = predicted_scores[i]

            """if self.model:
                features = np.array([[doc["retrieval_count"], doc["likes"], doc["dislikes"], doc["time_decay"]] for doc in results_list])
                scores = self.model.predict(features)
                for i, doc in enumerate(results_list):
                    doc["ranking_score"] = scores[i]
            else:
                # Apply simple heuristic scoring if ML model is unavailable
                for doc in results_list:
                    doc["ranking_score"] = doc["retrieval_count"] * 0.3 + doc["likes"] * 0.4 - doc["dislikes"] * 0.3 + doc["time_decay"] * 0.1

            return sorted(results_list, key=lambda x: x["ranking_score"], reverse=True)
            """

            return sorted(results, key=lambda x: x["final_score"], reverse=True)
        
        except Exception as e:
            logging.error(f"Ranking model failed: {str(e)}")
            return results  # Return unranked results if ML ranking fails

    async def periodic_training(self):
        """Periodically retrain the model every 24 hours (async)."""
        while True:
            try:
                self.train_ranking_model()
            except Exception as e:
                logging.error(f"Periodic training failed: {str(e)}")
            await asyncio.sleep(86400)  # 24 hours

    def compute_decay(last_used: datetime.datetime, decay_factor=0.05):
        """
        Compute time decay weight based on last usage time.
        Uses an exponential decay function to reduce relevance of old documents.
        """
        if not last_used:
            return 1.0  # Default score for new documents

        days_since_last_used = (datetime.datetime.utcnow() - last_used).days
        return math.exp(-decay_factor * days_since_last_used)

    def score_document(self,doc):
        """
        Compute an initial score before ML ranking.
        Scores based on retrieval count, likes/dislikes, and time decay.
        """
        retrieval_score = doc["retrieval_count"] * 0.3  # Weight retrieval count
        like_score = doc["likes"] * 0.5  # Weight user likes
        dislike_penalty = doc["dislikes"] * -0.3  # Penalize dislikes
        decay_factor = self.compute_decay(doc["last_used"])  # Apply time decay

        return (retrieval_score + like_score + dislike_penalty) * decay_factor
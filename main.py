import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import random


#  Product Catalog

products = pd.DataFrame({
    "Product": ["Minimalist Watch", "Leather Wallet", "Headphones", "Urban Backpack", "Running Shoes", "Coffee Mugs"],
    "Category": ["Accessories", "Accessories", "Electronics", "Travel", "Sports", "Home"],
    "Popularity": [500, 300, 800, 450, 600, 700]  # sales count
})


#  User–Item Ratings

ratings = pd.DataFrame({
    "User": ["A", "B", "C", "D", "E"],
    "Minimalist Watch": [5, 4, 2, 5, 4],
    "Leather Wallet":   [3, 0, 4, 3, 5],
    "Headphones":       [0, 5, 5, 0, 4],
    "Urban Backpack":   [4, 3, 9, 4, 7],
    "Running Shoes":    [0, 0, 4, 8, 5],
    "Coffee Mugs":      [5, 4, 2, 5, 6]
}).set_index("User")


#  Recommendation Engine

class EcommerceRecommender:

    def __init__(self, ratings, products):
        self.ratings = ratings
        self.products = products
        self.sim_matrix = pd.DataFrame(
            cosine_similarity(ratings),
            index=ratings.index,
            columns=ratings.index
        )

    def collaborative(self, user, top_n=3):
        sims = self.sim_matrix[user].sort_values(ascending=False).drop(user)
        sims = sims[sims > 0.5]
        if sims.empty: return []
        unrated = self.ratings.loc[user][self.ratings.loc[user]==0].index
        preds = {}
        for item in unrated:
            rated = self.ratings.loc[sims.index, item]
            rated = rated[rated>0]
            if not rated.empty:
                weights = sims.loc[rated.index]
                preds[item] = (rated*weights).sum()/weights.sum()
        return sorted(preds.items(), key=lambda x:x[1], reverse=True)[:top_n]

    def content_based(self, product_name, top_n=3):
        vecs = pd.get_dummies(self.products["Category"])
        sim = cosine_similarity(vecs)
        idx = self.products[self.products["Product"]==product_name].index[0]
        scores = list(enumerate(sim[idx]))
        scores = sorted(scores, key=lambda x:x[1], reverse=True)[1:top_n+1]
        return [(self.products.iloc[i]["Product"], float(s)) for i,s in scores]

    def popular(self, top_n=3):
        return self.products.sort_values("Popularity", ascending=False).head(top_n)[["Product","Popularity"]].values.tolist()

    def explore(self, top_n=2):
        return [(p,1) for p in random.sample(list(self.products["Product"]), top_n)]


#  Pie Chart Utility

def plot_pie(data, title, filename):
    if not data:
        print("No data to plot.")
        return None
    items = [d[0] for d in data]
    scores = [d[1] for d in data]
    plt.figure(figsize=(6,6))
    plt.pie(scores, labels=items, autopct="%1.1f%%", startangle=140)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved chart: {filename}")
    return filename

#  Interactive User Choice

engine = EcommerceRecommender(ratings, products)

print("Available users:", list(ratings.index))

chosen_user = input("Enter user ID (A–E): ").strip().upper()

if chosen_user not in ratings.index:
    print("Invalid user. Please choose from:", list(ratings.index))
else:
    # Collaborative recommendations
    cf_recs = engine.collaborative(chosen_user)
    plot_pie(cf_recs, f"Collaborative Recommendations for User {chosen_user}", f"cf_user{chosen_user}.png")

    # Popular products
    pop_recs = engine.popular()
    plot_pie(pop_recs, "Popular Products", "popular.png")

    # Exploration picks
    explore_recs = engine.explore()
    plot_pie(explore_recs, "Exploration Picks", "explore.png")

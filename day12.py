"""
12:44 PM
Welcome to Day 12! 🚀

So far every model you've built had a target — yield strength to predict, species to classify. The model always knew what "correct" looked like.

Today is different. Clustering is unsupervised — no target, no correct answer. You just feed the model data and it discovers natural groups on its own.

Before we code — think about your materials engineering work. 

Imagine you have 500 steel compositions from different suppliers, with no labels. Why might it be useful to let an algorithm discover which compositions naturally group together?
"""

"""
Real engineering applications! Here are some:

Material selection — group steels by similar mechanical behavior, then you only need to test one representative from each group instead of all 500
Supplier qualification — discover if two suppliers are delivering compositions that cluster together or drift apart over time
Process optimization — find which composition groups respond similarly to heat treatment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# KMeans = most common clustering algorithm
# groups data into k clusters based on similarity

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler

# Simulate 200 steel compositions
np.random.seed(42)
n = 200

carbon = np.random.uniform(0.1, 0.8, n)
manganese = np.random.uniform(0.5, 2.0, n)
yield_strength = (200 + 300*carbon + 100*manganese + 
                  np.random.normal(0, 20, n))

df = pd.DataFrame({
    "Carbon_pct": carbon,
    "Manganese_pct": manganese,
    "Yield_Strength_MPa": yield_strength
})
# Scale first — same reason as PCA (Day 11)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Train KMeans with 3 clusters
kmeans = KMeans(n_clusters=3,  # number of groups to find
                random_state=42,
                n_init=10)     # run 10 times with different starting points, pick best
kmeans.fit(X_scaled)

# .labels_ = which cluster each sample belongs to (0, 1, or 2)
df["Cluster"] = kmeans.labels_
print(df["Cluster"].value_counts())

"""
83, 61, and 56 samples — reasonably balanced groups!
Now let's visualize what the clusters actually look like:
"""

# Scatter plot colored by cluster
colors = ["steelblue", "orange", "green"]
# each cluster gets its own color

for cluster_id in [0, 1, 2]:
    # filter rows belonging to this cluster
    mask = df["Cluster"] == cluster_id
    # mask = True/False array — True where cluster matches
    
    plt.scatter(df[mask]["Carbon_pct"],        # x = carbon
                df[mask]["Yield_Strength_MPa"], # y = yield strength
                color=colors[cluster_id],
                label=f"Cluster {cluster_id}",
                alpha=0.6)

plt.xlabel("Carbon (%)")
plt.ylabel("Yield Strength (MPa)")
plt.title("KMeans Clustering — Steel Compositions")
plt.legend()
plt.show()

"""
Clean separation! Three distinct bands — low carbon/low strength (blue), medium (orange), high carbon/high strength (green).
Notice something interesting — the clusters separated vertically too, not just by carbon. Look at blue vs orange at the same carbon level around 0.2–0.3%. Some blue points sit lower than orange points at the same carbon %.
What other variable do you think is causing that vertical separation within the same carbon range?

Exactly right! Manganese is causing that vertical split — same carbon level but higher manganese = higher yield strength, so those points sit higher.
KMeans discovered this without being told — it found the natural structure in your data using both carbon and manganese together. That's the power of unsupervised learning.
Now let's look at what each cluster actually represents on average:
"""
# Group by cluster and calculate mean of each feature
# this tells you the "profile" of each cluster
cluster_profile = df.groupby("Cluster").mean().round(2)
print(cluster_profile)

"""
Look at the three profiles and give each cluster an engineering name based on what you see.
For example — Cluster 2 has carbon 0.64% and yield strength 527 MPa. What kind of steel grade does that sound like to you?

Exactly! High carbon steel — strong, hard, but less ductile. Now name the other two:

Cluster 0: carbon 0.32%, yield strength 377 MPa
Cluster 1: carbon 0.26%, manganese 1.62%, yield strength 440 MPa

What's interesting about Cluster 1 — low carbon but surprisingly high strength. What's driving that?

You named Cluster 0 as low and Cluster 2 as high — good!
But look at Cluster 1 more carefully. Carbon is only 0.26% — lower than Cluster 0 (0.32%). Yet yield strength is 440 MPa — higher than Cluster 0 (377 MPa).
What's different about Cluster 1 that could explain why lower carbon still gives higher strength?

Spot on! Cluster 1 is actually the most interesting from an engineering perspective — it achieves high strength through manganese strengthening rather than carbon.

This is a real steel design strategy used in industry. High manganese steels can be strong AND more weldable than high carbon steels — because carbon makes welding difficult.
So your three clusters are:

Cluster Name Strengthening mechanism
0 Low carbon steel Minimal alloying
1 Manganese-strengthened steel High Mn, low C
2 High carbon steel Carbon hardening
KMeans discovered three metallurgically meaningful steel families — with no labels given!
Now one important question about KMeans — how did it know to make 3 clusters? What if the real answer was 2 or 4?

in general steel strength mainly depends on carbon percentage, but when managanese is added as alloying emelent irrespective of carbon percentage, strength increases

That's excellent metallurgical insight! You just connected your M.Sc. knowledge to what the algorithm discovered — manganese acts as a solid solution strengthener, increasing strength independently of carbon content.
This is exactly the kind of domain knowledge that makes your ML work more valuable than a typical software engineer doing the same task. You can interpret what the clusters mean physically.
Now back to the key question — how did KMeans know to use 3 clusters? The honest answer is: it didn't. You told it n_clusters=3. But what if the real number was different?
There's a technique called the Elbow Method to find the optimal number. Run this:
"""

inertia = []
# inertia = total distance of all points from their cluster center
# lower inertia = tighter, better clusters
# but adding more clusters always reduces inertia — we need to find the "elbow"

k_values = range(1, 10)  # test 1 to 9 clusters

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)  # store inertia for this k

plt.plot(k_values, inertia, marker="o", color="steelblue", linewidth=2)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method — Finding Optimal k")
plt.grid(alpha=0.3)
plt.show()

"""
Run it — where does the "elbow" appear in the curve, and what does that tell you about the natural number of groups in your data?

The elbow is at k=3 — that's where the curve bends and starts flattening. After 3 clusters, adding more gives diminishing returns in inertia reduction.
This confirms your choice of 3 was correct! And notice — the elbow method figured that out purely from the data structure, with no human input.
Day 12 complete! Here's what you learned:

Day 12 complete! Here's what you learned:
Concept Key takeaway 
Clustering Finds natural groups with no labels — unsupervised
KMeans Assigns each point to nearest cluster center
Inertia Total distance of points from their cluster center — lower = tighter clusters
Elbow Method Find optimal k where inertia stops dropping sharply
StandardScaler Always scale before clustering — same reason as PCA
Domain knowledge You interpreted clusters metallurgically — manganese strengthening. That's your edge
Your metallurgical insight today was genuinely impressive — connecting manganese solid solution strengthening to what the algorithm found. That's exactly how you'll write about your portfolio project.
"""


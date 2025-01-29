{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "class KMeans:\n",
    "    def __init__(self, k, max_iters=100, tol=1e-4):\n",
    "        self.k = k\n",
    "        self.max_iters = max_iters\n",
    "        self.tol = tol\n",
    "        self.centroids = None\n",
    "    \n",
    "    def fit(self, X):\n",
    "        \"\"\"Fits the KMeans model to the data.\"\"\"\n",
    "        n_samples, n_features = X.shape\n",
    "        \n",
    "        # Randomly initialize centroids\n",
    "        np.random.seed(42)\n",
    "        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]\n",
    "        \n",
    "        for _ in range(self.max_iters):\n",
    "            # Assign clusters\n",
    "            clusters = self._assign_clusters(X)\n",
    "            \n",
    "            # Compute new centroids\n",
    "            new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(self.k)])\n",
    "            \n",
    "            # Check for convergence\n",
    "            if np.linalg.norm(self.centroids - new_centroids) < self.tol:\n",
    "                break\n",
    "            \n",
    "            self.centroids = new_centroids\n",
    "    \n",
    "    def _assign_clusters(self, X):\n",
    "        \"\"\"Assigns each sample to the nearest centroid.\"\"\"\n",
    "        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)\n",
    "        return np.argmin(distances, axis=1)\n",
    "    \n",
    "    def predict(self, X):\n",
    "        \"\"\"Predicts the cluster for new data points.\"\"\"\n",
    "        return self._assign_clusters(X)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

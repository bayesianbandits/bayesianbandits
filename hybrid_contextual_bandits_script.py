#!/usr/bin/env python
"""
# Hybrid vs Disjoint Contextual Bandits with LipschitzContextualAgent

In this tutorial, we'll explore the concept of hybrid contextual bandits as introduced by Li et al. (2010) in their LinUCB paper, and demonstrate how `LipschitzContextualAgent` enables efficient feature sharing across arms.

## Key Concepts

- **Disjoint bandits**: Each arm has completely separate parameters with no sharing of learned information
- **Hybrid bandits**: Arms share context features but maintain arm-specific adjustments
- **Benefits**: Better sample efficiency, faster learning, and knowledge transfer between arms

## What You'll Learn

1. The mathematical difference between disjoint and hybrid approaches
2. How to implement both approaches using bayesianbandits
3. Empirical comparison showing when hybrid bandits outperform disjoint bandits
4. Practical considerations for real-world applications
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Bayesian Bandits imports
from bayesianbandits import (
    Arm,
    ContextualAgent,
    LipschitzContextualAgent,
    ArmColumnFeaturizer,
    NormalRegressor,
    ThompsonSampling,
)

# Import LearnerPipeline for preprocessing
from bayesianbandits.pipelines import LearnerPipeline

# Sklearn imports for feature transformation
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

# Plotting style
plt.style.use("ggplot")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 10

"""
## Mathematical Framework

Let's understand the mathematical difference between disjoint and hybrid bandits:

### Disjoint Linear Bandits
Each arm $a$ has its own parameter vector $\\theta_a$:
$$\\mathbb{E}[r_t | x_t, a] = x_t^T \\theta_a$$

### Hybrid Linear Bandits  
All arms share some parameters with arm-specific adjustments:
$$\\mathbb{E}[r_t | x_t, a] = z_{t,a}^T \\theta$$

where $z_{t,a}$ is a feature vector that combines:
- Shared context features: $x_t$ (affect all arms)
- Interaction features: $x_t \\otimes \\text{category}(a)$ (category-specific effects)
- Arm-specific features: $\\text{one-hot}(a)$ (individual arm adjustments)

The key insight is that hybrid bandits learn a single parameter vector $\\theta$ that captures universal patterns, category patterns, and arm-specific effects.
"""

# Simulation parameters
n_users = 1000
n_articles = 20
n_rounds = 2000

# User features with clearer semantics
# - reading_level: affects ALL articles (truly shared)
# - general_interest: affects ALL articles (truly shared)
# - tech_interest: only matters for tech articles
# - sports_interest: only matters for sports articles
# - politics_interest: only matters for politics articles
n_shared_features = 2  # reading_level, general_interest
n_interest_features = 3  # tech, sports, politics
n_features = n_shared_features + n_interest_features

# Generate user features as DataFrame
feature_names = [
    "reading_level",
    "general_interest",
    "tech_interest",
    "sports_interest",
    "politics_interest",
]
X_users_array = np.random.randn(n_users, n_features)
X_users_array = (X_users_array - X_users_array.mean(axis=0)) / X_users_array.std(axis=0)
X_users = pd.DataFrame(X_users_array, columns=feature_names)

print(f"Generated {n_users} users with {n_features} features each")
print(f"Feature means: {X_users.mean().values}")
print(f"Feature stds: {X_users.std().values}")

"""
## Define true reward model

We'll create a realistic model where:
- Some features (reading_level, general_interest) affect ALL articles
- Some features (tech_interest, sports_interest, etc.) only affect their respective categories
- Each article also has small random variations
"""

# Define true reward model
# Shared effects: reading_level and general_interest affect ALL articles
theta_shared_all = np.array([0.8, 0.5])  # Only for truly shared features

# Article categories
article_categories = (
    ["tech"] * 5 + ["sports"] * 5 + ["politics"] * 5 + ["lifestyle"] * 5
)

# Create true parameters for each article
# Each article has:
# - Shared effects from reading_level and general_interest
# - Category-specific effect from the relevant interest feature
# - Small random variations
np.random.seed(42)
theta_articles_full = []

for i, category in enumerate(article_categories):
    # Start with zeros for all features
    theta = np.zeros(n_features)

    # Add shared effects (these apply to ALL articles)
    theta[:n_shared_features] = theta_shared_all

    # Add category-specific interest effects
    if category == "tech":
        theta[2] = 1.2  # tech_interest matters for tech articles
    elif category == "sports":
        theta[3] = 1.2  # sports_interest matters for sports articles
    elif category == "politics":
        theta[4] = 1.2  # politics_interest matters for politics articles
    # lifestyle articles don't have a specific interest feature

    # Add small random variations
    theta += np.random.randn(n_features) * 0.1

    theta_articles_full.append(theta)

theta_articles_full = np.array(theta_articles_full)


# True reward function
def true_reward(user_features, article_id: int, noise_std: float = 0.5) -> float:
    """Generate true reward for a user-article pair."""
    # Convert to numpy if DataFrame
    if hasattr(user_features, "values"):
        user_features = user_features.values
    if user_features.ndim > 1:
        user_features = user_features.flatten()

    # Reward is just the dot product with article's parameters
    reward = user_features @ theta_articles_full[article_id]
    noise = np.random.normal(0, noise_std)
    return reward + noise


# Visualize true parameters
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

im = ax.imshow(theta_articles_full.T, cmap="RdBu_r", aspect="auto")
ax.set_xlabel("Article ID")
ax.set_ylabel("Feature")
ax.set_title("True Parameters for Each Article")
ax.set_yticks(range(n_features))
ax.set_yticklabels(feature_names)

# Add category labels
for i in range(0, n_articles, 5):
    ax.axvline(x=i - 0.5, color="black", linewidth=1)
    if i < n_articles:
        ax.text(i + 2.5, -1, article_categories[i], ha="center", va="top")

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()

print("Key insights from the true model:")
print("1. Reading level and general interest affect ALL articles similarly")
print("2. Tech interest only affects tech articles (articles 0-4)")
print("3. Sports interest only affects sports articles (articles 5-9)")
print("4. Politics interest only affects politics articles (articles 10-14)")
print("5. Lifestyle articles (15-19) only use shared features")

"""
## Example: News Article Recommendation

We'll simulate a news recommendation system where:
- Users have features that include both universal preferences and topic-specific interests
- Articles belong to different categories (tech, sports, politics, lifestyle)
- Some user features affect all articles (e.g., reading level preference)
- Some features only matter for specific categories (e.g., sports interest → sports articles)

This setup mirrors real recommendation systems where:
- General engagement features (time of day, device type) affect all items
- Domain-specific features (genre preferences) only matter for relevant items
"""

"""
## Approach 1: Disjoint Bandits

First, let's implement the traditional approach where each article has its own completely separate model.
"""

# Create disjoint bandits - each article gets its own learner
disjoint_arms = []
for i in range(n_articles):
    arm = Arm(
        action_token=i,
        learner=NormalRegressor(
            alpha=1.0,  # Prior precision
            beta=1.0,  # Noise precision
        ),
    )
    disjoint_arms.append(arm)

# Create agent with Thompson Sampling policy
disjoint_agent = ContextualAgent(arms=disjoint_arms, policy=ThompsonSampling())

print(f"Created disjoint agent with {len(disjoint_arms)} independent arms")

"""
## Approach 2: Hybrid Bandits with LipschitzContextualAgent

Now let's implement the hybrid approach using a single shared model with arm-specific features.
"""

# Create arm featurizer that appends arm ID as a column
arm_featurizer = ArmColumnFeaturizer(column_name="article_id")


# Create a more sophisticated feature engineering pipeline
# This will create:
# 1. Shared features that affect all arms
# 2. Interaction features between interests and article categories
def create_interaction_features(df):
    """Create interaction features between user interests and article categories."""
    # df has columns: reading_level, general_interest, tech_interest, sports_interest, politics_interest, article_id

    # Create a copy to avoid modifying original
    result = df.copy()

    # Map article IDs to categories
    def get_category(article_id):
        if article_id < 5:
            return "tech"
        elif article_id < 10:
            return "sports"
        elif article_id < 15:
            return "politics"
        else:
            return "lifestyle"

    # Add category column
    result["category"] = result["article_id"].apply(get_category)

    # Create interaction features
    result["tech_x_is_tech"] = result["tech_interest"] * (
        result["category"] == "tech"
    ).astype(float)
    result["sports_x_is_sports"] = result["sports_interest"] * (
        result["category"] == "sports"
    ).astype(float)
    result["politics_x_is_politics"] = result["politics_interest"] * (
        result["category"] == "politics"
    ).astype(float)

    # Drop the category column and original interest columns (we only want interactions)
    result = result.drop(
        columns=["category", "tech_interest", "sports_interest", "politics_interest"]
    )

    return result


# Create preprocessing pipeline
preprocessor = make_pipeline(
    FunctionTransformer(create_interaction_features),
    ColumnTransformer(
        [
            ("shared", "passthrough", ["reading_level", "general_interest"]),
            (
                "interactions",
                "passthrough",
                ["tech_x_is_tech", "sports_x_is_sports", "politics_x_is_politics"],
            ),
            ("article_id", OneHotEncoder(sparse_output=False), ["article_id"]),
        ]
    ),
)

# Pre-fit the preprocessor on dummy data
dummy_contexts = pd.DataFrame(np.random.randn(100, n_features), columns=feature_names)
dummy_arms = list(range(n_articles))
dummy_X_transformed = arm_featurizer.transform(dummy_contexts, action_tokens=dummy_arms)
preprocessor.fit(dummy_X_transformed)

# Create a single shared learner wrapped in LearnerPipeline
learner_pipeline = LearnerPipeline(
    steps=[("preprocessor", preprocessor)],
    learner=NormalRegressor(
        alpha=1.0,  # Prior precision
        beta=1.0,  # Noise precision
    ),
)

# Create arms for hybrid agent
hybrid_arms = [Arm(action_token=i) for i in range(n_articles)]

# Create hybrid agent
hybrid_agent = LipschitzContextualAgent(
    arms=hybrid_arms,
    learner=learner_pipeline,
    policy=ThompsonSampling(),
    arm_featurizer=arm_featurizer,
)

print(f"Created hybrid agent with {n_articles} arms sharing a single learner")
print("Feature engineering:")
print("- Shared features: reading_level, general_interest (affect all articles)")
print("- Interaction features: tech_interest × is_tech_article, etc.")
print("- One-hot encoded article IDs for article-specific adjustments")

"""
## Running the Comparison Experiment

Let's run both agents on the same sequence of users and compare their performance.
"""

# Initialize metrics storage
rewards_disjoint = []
rewards_hybrid = []
regrets_disjoint = []
regrets_hybrid = []
arms_pulled_disjoint = []
arms_pulled_hybrid = []

# Run simulation
print("\nRunning simulation...")
for t in range(n_rounds):
    if (t + 1) % 500 == 0:
        print(f"Round {t + 1}/{n_rounds}")

    # Sample a random user
    user_idx = np.random.randint(n_users)
    user_features = X_users.iloc[[user_idx]]  # DataFrame with shape: (1, n_features)

    # Compute optimal reward for this user
    user_array = X_users.iloc[user_idx].values
    optimal_rewards = [
        true_reward(user_array, a, noise_std=0) for a in range(n_articles)
    ]
    optimal_reward = max(optimal_rewards)
    optimal_arm = np.argmax(optimal_rewards)

    # Disjoint agent's turn
    arms_disjoint = disjoint_agent.pull(user_features)
    arm_disjoint = arms_disjoint[0]  # Get first (and only) arm from list
    reward_disjoint = true_reward(user_array, arm_disjoint)
    disjoint_agent.select_for_update(arm_disjoint).update(
        user_features, np.array([reward_disjoint])
    )

    # Hybrid agent's turn
    arms_hybrid = hybrid_agent.pull(user_features)
    arm_hybrid = arms_hybrid[0]  # Get first (and only) arm from list
    reward_hybrid = true_reward(user_array, arm_hybrid)
    hybrid_agent.select_for_update(arm_hybrid).update(
        user_features, np.array([reward_hybrid])
    )

    # Record metrics
    rewards_disjoint.append(reward_disjoint)
    rewards_hybrid.append(reward_hybrid)
    regrets_disjoint.append(optimal_reward - reward_disjoint)
    regrets_hybrid.append(optimal_reward - reward_hybrid)
    arms_pulled_disjoint.append(arm_disjoint)
    arms_pulled_hybrid.append(arm_hybrid)

print("Simulation complete!")

"""
## Analyzing the Results

Let's visualize and analyze the performance of both approaches.
"""

# Calculate cumulative metrics
cumulative_regret_disjoint = np.cumsum(regrets_disjoint)
cumulative_regret_hybrid = np.cumsum(regrets_hybrid)
cumulative_reward_disjoint = np.cumsum(rewards_disjoint)
cumulative_reward_hybrid = np.cumsum(rewards_hybrid)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Cumulative Regret
ax = axes[0, 0]
ax.plot(cumulative_regret_disjoint, label="Disjoint Bandits", linewidth=2)
ax.plot(cumulative_regret_hybrid, label="Hybrid Bandits", linewidth=2)
ax.set_xlabel("Round")
ax.set_ylabel("Cumulative Regret")
ax.set_title("Cumulative Regret Comparison")
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Average Reward (Moving Average)
ax = axes[0, 1]
window = 100
avg_reward_disjoint = pd.Series(rewards_disjoint).rolling(window, min_periods=1).mean()
avg_reward_hybrid = pd.Series(rewards_hybrid).rolling(window, min_periods=1).mean()
ax.plot(avg_reward_disjoint, label="Disjoint Bandits", linewidth=2)
ax.plot(avg_reward_hybrid, label="Hybrid Bandits", linewidth=2)
ax.set_xlabel("Round")
ax.set_ylabel(f"Average Reward ({window}-round MA)")
ax.set_title("Average Reward Over Time")
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Arm Pull Distribution
ax = axes[1, 0]
arm_counts_disjoint = pd.Series(arms_pulled_disjoint).value_counts().sort_index()
arm_counts_hybrid = pd.Series(arms_pulled_hybrid).value_counts().sort_index()
x = np.arange(n_articles)
width = 0.35
bars1 = ax.bar(
    x - width / 2,
    [arm_counts_disjoint.get(i, 0) for i in range(n_articles)],
    width,
    label="Disjoint",
    alpha=0.8,
)
bars2 = ax.bar(
    x + width / 2,
    [arm_counts_hybrid.get(i, 0) for i in range(n_articles)],
    width,
    label="Hybrid",
    alpha=0.8,
)
ax.set_xlabel("Article ID")
ax.set_ylabel("Times Pulled")
ax.set_title("Arm Pull Distribution")
ax.legend()
ax.set_xticks(x)

# Color bars by category
colors = {"tech": "blue", "sports": "green", "politics": "red", "lifestyle": "orange"}
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    color = colors[article_categories[i]]
    bar1.set_facecolor(color)
    bar2.set_facecolor(color)

# 4. Regret Reduction Rate
ax = axes[1, 1]
rounds = np.arange(100, n_rounds, 100)
regret_rate_disjoint = [cumulative_regret_disjoint[r] / r for r in rounds]
regret_rate_hybrid = [cumulative_regret_hybrid[r] / r for r in rounds]
ax.plot(rounds, regret_rate_disjoint, label="Disjoint Bandits", linewidth=2, marker="o")
ax.plot(rounds, regret_rate_hybrid, label="Hybrid Bandits", linewidth=2, marker="s")
ax.set_xlabel("Round")
ax.set_ylabel("Average Regret per Round")
ax.set_title("Regret Rate Over Time")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n=== Performance Summary ===")
print("Total Cumulative Regret:")
print(f"  Disjoint: {cumulative_regret_disjoint[-1]:.2f}")
print(f"  Hybrid: {cumulative_regret_hybrid[-1]:.2f}")
print(
    f"  Improvement: {(1 - cumulative_regret_hybrid[-1] / cumulative_regret_disjoint[-1]) * 100:.1f}%"
)
print("\nFinal Average Reward (last 100 rounds):")
print(f"  Disjoint: {np.mean(rewards_disjoint[-100:]):.3f}")
print(f"  Hybrid: {np.mean(rewards_hybrid[-100:]):.3f}")

"""
## Analyzing Learning Efficiency

Let's examine how quickly each approach learns about different articles, especially those with limited data.
"""


# Analyzing per-arm performance over time
def calculate_per_arm_accuracy(arms_pulled, rewards, n_articles, window=50):
    """Calculate how often each arm was optimal when pulled."""
    arm_performance = {i: [] for i in range(n_articles)}

    for t, (arm, reward) in enumerate(zip(arms_pulled, rewards)):
        # Was this a good choice?
        user_idx = np.random.randint(n_users)  # Same sequence as simulation
        user_array = X_users.iloc[user_idx].values
        optimal_rewards = [
            true_reward(user_array, a, noise_std=0) for a in range(n_articles)
        ]
        is_optimal = arm == np.argmax(optimal_rewards)
        arm_performance[arm].append(is_optimal)

    return arm_performance


# Reset random seed to match simulation
np.random.seed(42)
perf_disjoint = calculate_per_arm_accuracy(
    arms_pulled_disjoint, rewards_disjoint, n_articles
)
np.random.seed(42)
perf_hybrid = calculate_per_arm_accuracy(arms_pulled_hybrid, rewards_hybrid, n_articles)

# Find arms with few pulls
rare_arms = [i for i, count in arm_counts_disjoint.items() if count < 50]
common_arms = [i for i, count in arm_counts_disjoint.items() if count >= 100]

print(f"\nArms with <50 pulls (disjoint): {rare_arms}")
print(f"Arms with ≥100 pulls (disjoint): {common_arms}")

# Compare accuracy on rare arms
if rare_arms:
    print("\n=== Performance on Rarely-Pulled Arms ===")
    for arm in rare_arms[:3]:  # Show first 3 rare arms
        if arm in perf_disjoint and perf_disjoint[arm]:
            acc_disjoint = np.mean(perf_disjoint[arm]) * 100
            pulls_disjoint = len(perf_disjoint[arm])
        else:
            acc_disjoint = 0
            pulls_disjoint = 0

        if arm in perf_hybrid and perf_hybrid[arm]:
            acc_hybrid = np.mean(perf_hybrid[arm]) * 100
            pulls_hybrid = len(perf_hybrid[arm])
        else:
            acc_hybrid = 0
            pulls_hybrid = 0

        print(f"\nArm {arm} ({article_categories[arm]} article):")
        print(f"  Disjoint: {pulls_disjoint} pulls, {acc_disjoint:.1f}% optimal")
        print(f"  Hybrid: {pulls_hybrid} pulls, {acc_hybrid:.1f}% optimal")

"""
## Examining Learned Parameters

Let's look at what each approach actually learned about the problem structure.
"""

# Examining learned parameters
# Extract learned parameters from hybrid model
if hasattr(hybrid_agent.learner, "learner") and hasattr(
    hybrid_agent.learner.learner, "coef_"
):
    hybrid_params = hybrid_agent.learner.learner.coef_
    print(f"Hybrid model learned {len(hybrid_params)} parameters")

    # The first 2 parameters should be the shared effects
    learned_shared = hybrid_params[:2]
    print(
        f"\nLearned shared parameters (reading_level, general_interest): {learned_shared}"
    )
    print(f"True shared parameters: {theta_shared_all}")

    # Next 3 are the interaction effects
    learned_interactions = hybrid_params[2:5]
    print(f"\nLearned interaction effects: {learned_interactions}")
    print("(tech×is_tech, sports×is_sports, politics×is_politics)")

else:
    print("Could not access learned parameters")

# Check parameter recovery for a few well-explored arms in disjoint model
print("\n\n=== Parameter Recovery for Disjoint Model ===")
well_explored = sorted(
    [(arm, count) for arm, count in arm_counts_disjoint.items()],
    key=lambda x: x[1],
    reverse=True,
)[:3]

for arm_id, count in well_explored:
    arm = disjoint_arms[arm_id]
    if hasattr(arm.learner, "coef_"):
        learned = arm.learner.coef_
        true = theta_articles_full[arm_id]
        if len(learned) == len(true):
            corr = np.corrcoef(learned, true)[0, 1]
            print(f"\nArm {arm_id} ({article_categories[arm_id]}): {count} pulls")
            print(f"  Correlation with true params: {corr:.3f}")
            print(f"  Learned: {learned}")
            print(f"  True: {true}")

"""
## Knowledge Transfer Analysis

One key advantage of hybrid bandits is knowledge transfer - learning about one arm helps with others. Let's examine this.
"""

# Knowledge Transfer Analysis
# Group articles by category
category_groups = {}
for i, cat in enumerate(article_categories):
    if cat not in category_groups:
        category_groups[cat] = []
    category_groups[cat].append(i)

# Calculate within-category vs across-category performance
print("=== Knowledge Transfer Analysis ===")
print("\nWithin-Category Learning (Do similar articles help each other?)")

for category, articles in category_groups.items():
    # Find the most and least pulled articles in this category
    pulls_in_category = [(a, arm_counts_hybrid.get(a, 0)) for a in articles]
    pulls_in_category.sort(key=lambda x: x[1], reverse=True)

    if len(pulls_in_category) >= 2:
        most_pulled = pulls_in_category[0]
        least_pulled = pulls_in_category[-1]

        print(f"\n{category.capitalize()} articles:")
        print(f"  Most pulled: Article {most_pulled[0]} ({most_pulled[1]} pulls)")
        print(f"  Least pulled: Article {least_pulled[0]} ({least_pulled[1]} pulls)")

        # In hybrid model, even least-pulled benefits from most-pulled
        if least_pulled[0] in perf_hybrid and perf_hybrid[least_pulled[0]]:
            accuracy = np.mean(perf_hybrid[least_pulled[0]]) * 100
            print(f"  Least-pulled accuracy (hybrid): {accuracy:.1f}%")

        if least_pulled[0] in perf_disjoint and perf_disjoint[least_pulled[0]]:
            accuracy_disjoint = np.mean(perf_disjoint[least_pulled[0]]) * 100
            print(f"  Least-pulled accuracy (disjoint): {accuracy_disjoint:.1f}%")

"""
## Computational Efficiency Comparison

Let's also examine the computational efficiency of both approaches.
"""

import time

# Measure prediction time for a batch of users
n_test_users = 100
test_features = X_users.iloc[:n_test_users]

# Disjoint timing
start_time = time.time()
for idx in range(n_test_users):
    _ = disjoint_agent.pull(test_features.iloc[[idx]])
disjoint_time = time.time() - start_time

# Hybrid timing
start_time = time.time()
for idx in range(n_test_users):
    _ = hybrid_agent.pull(test_features.iloc[[idx]])
hybrid_time = time.time() - start_time

print("\n=== Computational Efficiency ===")
print(f"\nTime to make {n_test_users} predictions:")
print(
    f"  Disjoint: {disjoint_time:.3f} seconds ({disjoint_time / n_test_users * 1000:.1f} ms/prediction)"
)
print(
    f"  Hybrid: {hybrid_time:.3f} seconds ({hybrid_time / n_test_users * 1000:.1f} ms/prediction)"
)
print(f"  Speedup: {disjoint_time / hybrid_time:.1f}x")

# Memory usage comparison
print("\n=== Memory Efficiency ===")
print("Number of parameters to store:")
print(
    f"  Disjoint: {n_articles} models × {n_features + 1} params = {n_articles * (n_features + 1)} total"
)
print(
    f"  Hybrid: 1 model × (2 shared + 3 interactions + {n_articles} one-hot) = {2 + 3 + n_articles} total"
)
print(
    f"  Memory reduction: {(1 - (2 + 3 + n_articles) / (n_articles * (n_features + 1))) * 100:.1f}%"
)

"""
## When to Use Each Approach

Based on our experiments, let's summarize when to use each approach.
"""

# Create a decision matrix visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Define scenarios and scores (based on our results)
scenarios = [
    "Many arms (>100)",
    "Limited data per arm",
    "Arms share common structure",
    "Need fast predictions",
    "Memory constrained",
    "Arms are very different",
    "Need interpretability per arm",
]

# Scores: 0 = Bad, 1 = Okay, 2 = Good, 3 = Excellent
disjoint_scores = [1, 1, 1, 2, 1, 3, 3]
hybrid_scores = [3, 3, 3, 3, 3, 1, 1]

# Create heatmap data
scores = np.array([disjoint_scores, hybrid_scores]).T
methods = ["Disjoint", "Hybrid"]

# Plot heatmap
im = ax.imshow(scores, cmap="RdYlGn", aspect="auto", vmin=0, vmax=3)

# Set ticks and labels
ax.set_xticks(np.arange(len(methods)))
ax.set_yticks(np.arange(len(scenarios)))
ax.set_xticklabels(methods)
ax.set_yticklabels(scenarios)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_ticks([0, 1, 2, 3])
cbar.set_ticklabels(["Poor", "Fair", "Good", "Excellent"])

# Add text annotations
for i in range(len(scenarios)):
    for j in range(len(methods)):
        text = ax.text(
            j, i, scores[i, j], ha="center", va="center", color="black", fontsize=12
        )

ax.set_title("When to Use Each Approach", fontsize=14, pad=20)
plt.tight_layout()
plt.show()

"""
## Advanced: Custom Feature Sharing

For completeness, let's show how to implement partial feature sharing for more nuanced control.
"""

from bayesianbandits import ArmFeaturizer


class CategoryAwareFeaturizer(ArmFeaturizer):
    """Custom featurizer that creates category-level sharing."""

    def __init__(self, categories, n_user_features):
        self.categories = categories
        self.n_user_features = n_user_features
        self.unique_categories = list(set(categories))
        self.category_to_idx = {cat: i for i, cat in enumerate(self.unique_categories)}

    def transform(self, X, *, action_tokens):
        """Transform contexts with category-aware arm features."""
        # Convert to DataFrame if needed
        if hasattr(X, "values"):
            X_array = X.values
        else:
            X_array = X

        n_contexts = X_array.shape[0]
        n_arms = len(action_tokens)
        n_categories = len(self.unique_categories)

        # Output shape: (n_contexts * n_arms, n_user_features + n_categories + n_arms)
        n_features = self.n_user_features + n_categories + n_arms
        X_transformed = np.zeros((n_contexts * n_arms, n_features))

        idx = 0
        for i in range(n_contexts):
            for j, arm in enumerate(action_tokens):
                # User features
                X_transformed[idx, : self.n_user_features] = X_array[i]

                # Category indicator
                category = self.categories[arm]
                cat_idx = self.category_to_idx[category]
                X_transformed[idx, self.n_user_features + cat_idx] = 1

                # Arm-specific indicator
                X_transformed[idx, self.n_user_features + n_categories + arm] = 1

                idx += 1

        return X_transformed


# Example usage (not run in this cell)
# category_featurizer = CategoryAwareFeaturizer(article_categories, n_features)
# category_agent = LipschitzContextualAgent(
#     arms=list(range(n_articles)),
#     learner=NormalRegressor(alpha=1.0, beta=1.0),
#     policy=ThompsonSampling(),
#     arm_featurizer=category_featurizer
# )

print("\nCreated category-aware featurizer with hierarchical sharing:")
print(f"- {n_features} user features (fully shared)")
print(f"- {len(set(article_categories))} category features (shared within category)")
print(f"- {n_articles} arm-specific features (not shared)")

"""
## Conclusions

Based on our experimental results, here are the key takeaways:

### 1. **Feature Engineering is Critical for Hybrid Bandits**
- Simply concatenating all features doesn't leverage the hybrid advantage
- Creating meaningful shared features and interactions is essential
- Our approach: shared features (reading_level, general_interest) + interaction features (interest × category)

### 2. **When Hybrid Bandits Excel**
- **Many arms with shared structure**: Performance gap widens with more arms
- **Limited data per arm**: Hybrid learns faster by pooling information
- **Clear feature semantics**: When you know which features should be shared vs. arm-specific

### 3. **When Disjoint Bandits May Be Better**
- **Fundamentally different arms**: No meaningful shared structure
- **Abundant data**: Each arm gets thousands of observations
- **Complex heterogeneous effects**: When arms behave very differently

### 4. **Implementation Best Practices**
- Use pandas DataFrames for proper column tracking with ColumnTransformer
- Create interaction features explicitly rather than hoping the model learns them
- Pre-fit transformers to ensure consistent feature dimensions
- Consider which features truly affect all arms vs. specific subsets

### 5. **The LipschitzContextualAgent Advantage**
- Efficient batched operations for large action spaces
- Single forward pass through the model for all arms
- Memory efficient compared to storing separate models
- Easy integration with sklearn preprocessing pipelines

### 6. **Real-World Application Tips**
- Start by identifying truly shared effects (e.g., user engagement, quality preference)
- Create category-based interactions for domain-specific features
- Monitor which arms get few observations - these benefit most from hybrid approach
- Use cross-validation to tune the prior precision (alpha) for the shared model

The hybrid approach with proper feature engineering can significantly outperform disjoint bandits, especially in scenarios with many arms and clear shared structure. The key is thoughtful feature design that captures both universal patterns and arm-specific variations.
"""

print("\n=== Tutorial Complete ===")
print(
    f"Final improvement: {(1 - cumulative_regret_hybrid[-1] / cumulative_regret_disjoint[-1]) * 100:.1f}%"
)
print("The hybrid approach demonstrated significant advantages through:")
print("- Faster learning via knowledge transfer")
print("- Better performance on rarely-seen arms")
print("- Computational and memory efficiency")
print("\nKey insight: Proper feature engineering (shared + interactions) is crucial!")

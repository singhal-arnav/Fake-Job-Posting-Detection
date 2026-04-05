
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud


def load_data(filepath: str) -> pd.DataFrame:
    """Load the job-postings CSV and return a DataFrame."""
    return pd.read_csv(filepath)


def print_overview(df: pd.DataFrame) -> None:
    """Print shape, dtypes, and summary statistics for the dataset."""
    print("=== Dataset Overview ===")
    df.info()
    print("\n", df.describe())


def print_class_distribution(df: pd.DataFrame) -> None:
    """Print absolute and relative counts for the target variable."""
    print("=== Class Distribution ===")
    print(df["fraudulent"].value_counts())
    print(df["fraudulent"].value_counts(normalize=True))


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Bar chart of fraudulent vs. real job postings."""
    df["fraudulent"].value_counts().plot(kind="bar")
    plt.title("Class Distribution")
    plt.show()


def print_missing_values(df: pd.DataFrame) -> None:
    """Print columns sorted by number of missing values (descending)."""
    print("=== Missing Values ===")
    print(df.isnull().sum().sort_values(ascending=False))


def plot_missing_heatmap(df: pd.DataFrame) -> None:
    """Seaborn heatmap showing missingness across the dataset."""
    sns.heatmap(df.isnull(), cbar=False)
    plt.show()


def add_description_length(df: pd.DataFrame) -> pd.DataFrame:
    """Add a character-count column for the job description."""
    df = df.copy()
    df["desc_length"] = df["description"].astype(str).apply(len)
    return df


def print_length_by_class(df: pd.DataFrame) -> None:
    """Print mean description length split by fraud label."""
    print("=== Mean Description Length by Class ===")
    print(df.groupby("fraudulent")["desc_length"].mean())


def plot_length_distribution(df: pd.DataFrame) -> None:
    """Histogram of description lengths, coloured by fraud label."""
    sns.histplot(data=df, x="desc_length", hue="fraudulent", bins=50)
    plt.title("Description Length Distribution")
    plt.show()



def get_words(text: str) -> list:
    """Tokenise a string into lowercase word tokens."""
    return re.findall(r"\w+", str(text).lower())


def compute_word_frequencies(df: pd.DataFrame) -> tuple:
    """
    Return (real_word_counter, fake_word_counter) from description column.
    """
    real_words, fake_words = [], []
    for _, row in df.iterrows():
        words = get_words(row["description"])
        if row["fraudulent"] == 0:
            real_words.extend(words)
        else:
            fake_words.extend(words)
    return Counter(real_words), Counter(fake_words)


def print_top_words(df: pd.DataFrame, top_n: int = 20) -> None:
    """Print the most common words for real and fake postings."""
    real_counter, fake_counter = compute_word_frequencies(df)
    print("=== Top Words – Real Postings ===")
    print(real_counter.most_common(top_n))
    print("\n=== Top Words – Fake Postings ===")
    print(fake_counter.most_common(top_n))


def plot_wordcloud(df: pd.DataFrame, label: int = 1) -> None:
    """
    Generate a word cloud for the given fraud label.
    label=1 → fake postings; label=0 → real postings.
    """
    title = "Fake Jobs WordCloud" if label == 1 else "Real Jobs WordCloud"
    text = " ".join(df[df["fraudulent"] == label]["description"].astype(str))
    wc = WordCloud(width=800, height=400).generate(text)
    plt.imshow(wc)
    plt.axis("off")
    plt.title(title)
    plt.show()

def print_employment_type_counts(df: pd.DataFrame) -> None:
    """Print value counts for the employment_type column."""
    print("=== Employment Type Counts ===")
    print(df["employment_type"].value_counts())


def plot_employment_by_fraud(df: pd.DataFrame) -> None:
    """Grouped bar chart of employment type split by fraud label."""
    sns.countplot(data=df, x="employment_type", hue="fraudulent")
    plt.xticks(rotation=45)
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of Pearson correlations among all numeric columns."""
    sns.heatmap(df.corr(numeric_only=True), annot=True)
    plt.show()


def print_top_bigrams(df: pd.DataFrame, top_n: int = 20) -> None:
    """Print the most frequent bi-grams across all descriptions."""
    vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words="english")
    X = vectorizer.fit_transform(df["description"].astype(str))
    sum_words = X.sum(axis=0)
    words_freq = [
        (word, sum_words[0, idx])
        for word, idx in vectorizer.vocabulary_.items()
    ]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    print("=== Top Bi-grams ===")
    print(words_freq[:top_n])


if __name__ == "__main__":

    # 1. Load data
    df = load_data("merged_job_postings.csv")

    # 2. Dataset overview
    print_overview(df)

    # 3. Class distribution
    print_class_distribution(df)
    plot_class_distribution(df)

    # 4. Missing values
    print_missing_values(df)
    plot_missing_heatmap(df)

    # 5. Text length analysis
    df = add_description_length(df)
    print_length_by_class(df)
    plot_length_distribution(df)

    # 6. Word frequency analysis
    print_top_words(df)

    # 7. Word clouds
    plot_wordcloud(df, label=1)   # fake postings
    plot_wordcloud(df, label=0)   # real postings

    # 8. Categorical feature analysis
    print_employment_type_counts(df)
    plot_employment_by_fraud(df)

    # 9. Correlation heatmap
    plot_correlation_heatmap(df)

    # 10. Bi-gram analysis
    print_top_bigrams(df)

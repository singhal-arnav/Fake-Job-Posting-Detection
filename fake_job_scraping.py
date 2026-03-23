# ============================================================
# FAKE JOB DETECTION - WEB SCRAPING
# Scrapes real job postings from RemoteOK (public API)
# and merges with Kaggle dataset
# ============================================================

import requests
import pandas as pd
import time
import re
from bs4 import BeautifulSoup

# ============================================================
# METHOD 1: RemoteOK Public API (Recommended — no blocking)
# Fetches real, live remote job postings
# ============================================================

def scrape_remoteok(max_jobs=200):
    """
    Scrapes jobs from RemoteOK's public API.
    All jobs here are REAL (fraudulent = 0).
    """
    url = "https://remoteok.com/api"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }

    print("Fetching jobs from RemoteOK API...")
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching RemoteOK: {e}")
        return pd.DataFrame()

    jobs = data[1:]  # First item is metadata, skip it
    records = []

    for job in jobs[:max_jobs]:
        # Clean HTML tags from description
        desc_raw = job.get("description", "") or ""
        desc_clean = BeautifulSoup(desc_raw, "html.parser").get_text(separator=" ")
        desc_clean = re.sub(r"\s+", " ", desc_clean).strip()

        records.append({
            "title":           job.get("position", ""),
            "location":        job.get("location", ""),
            "company_profile": job.get("company", ""),
            "description":     desc_clean,
            "requirements":    "",
            "benefits":        "",
            "employment_type": "Full-time",
            "required_experience": "",
            "required_education":  "",
            "industry":        ", ".join(job.get("tags", [])),
            "function":        "",
            "fraudulent":      0   # ← These are verified real jobs
        })

        time.sleep(0.2)  # Be polite to the server

    print(f"Scraped {len(records)} jobs from RemoteOK.")
    return pd.DataFrame(records)


# ============================================================
# METHOD 2: Indeed-style HTML Scraper (Backup / Demo)
# Note: Indeed blocks bots heavily; use with caution.
# For academic use only. Check robots.txt before running.
# ============================================================

def scrape_indeed_sample(query="software engineer", location="remote", pages=2):
    """
    Demonstrates HTML scraping logic.
    May be blocked by Indeed — use RemoteOK API above for reliability.
    """
    base_url = "https://www.indeed.com/jobs"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }

    records = []

    for page in range(pages):
        params = {
            "q": query,
            "l": location,
            "start": page * 10
        }
        print(f"Scraping Indeed page {page + 1}...")

        try:
            response = requests.get(base_url, headers=headers, params=params, timeout=15)
            soup = BeautifulSoup(response.text, "html.parser")

            # Indeed job cards
            job_cards = soup.find_all("div", class_="job_seen_beacon")
            if not job_cards:
                print("  No cards found — Indeed may be blocking the request.")
                break

            for card in job_cards:
                title_tag   = card.find("h2", class_="jobTitle")
                company_tag = card.find("span", class_="companyName")
                loc_tag     = card.find("div", class_="companyLocation")
                desc_tag    = card.find("div", class_="job-snippet")

                records.append({
                    "title":           title_tag.get_text(strip=True)   if title_tag   else "",
                    "location":        loc_tag.get_text(strip=True)     if loc_tag     else "",
                    "company_profile": company_tag.get_text(strip=True) if company_tag else "",
                    "description":     desc_tag.get_text(strip=True)    if desc_tag    else "",
                    "requirements":    "",
                    "benefits":        "",
                    "employment_type": "",
                    "required_experience": "",
                    "required_education":  "",
                    "industry":        "",
                    "function":        "",
                    "fraudulent":      0
                })

            time.sleep(2)  # Wait between pages to avoid getting blocked

        except Exception as e:
            print(f"  Error on page {page + 1}: {e}")
            break

    print(f"Scraped {len(records)} jobs from Indeed.")
    return pd.DataFrame(records)


# ============================================================
# STEP 3: MERGE WITH KAGGLE DATASET
# ============================================================

def merge_datasets(kaggle_path="fake_job_postings.csv", scraped_df=None):
    """
    Loads the Kaggle dataset and merges with scraped data.
    Ensures column alignment before merging.
    """
    print("\nLoading Kaggle dataset...")
    kaggle_df = pd.read_csv(kaggle_path)
    print(f"Kaggle dataset: {kaggle_df.shape[0]} rows")

    if scraped_df is None or scraped_df.empty:
        print("No scraped data to merge. Using Kaggle dataset only.")
        return kaggle_df

    print(f"Scraped dataset: {scraped_df.shape[0]} rows")

    # Keep only columns that exist in Kaggle dataset
    common_cols = [c for c in scraped_df.columns if c in kaggle_df.columns]
    scraped_aligned = scraped_df[common_cols]

    # Combine
    combined_df = pd.concat([kaggle_df, scraped_aligned], ignore_index=True)

    # Fill all NaN with empty string (for text columns)
    for col in combined_df.columns:
        if combined_df[col].dtype == object:
            combined_df[col] = combined_df[col].fillna("")
        else:
            combined_df[col] = combined_df[col].fillna(0)

    print(f"\nMerged dataset: {combined_df.shape[0]} rows")
    print("Fraud distribution in merged dataset:")
    print(combined_df["fraudulent"].value_counts())

    # Save merged dataset
    combined_df.to_csv("merged_job_postings.csv", index=False)
    print("\nSaved merged dataset to: merged_job_postings.csv")

    return combined_df


# ============================================================
# MAIN — Run everything
# ============================================================
if __name__ == "__main__":

    # --- SCRAPE ---
    # Use RemoteOK (reliable, no blocking)
    scraped_df = scrape_remoteok(max_jobs=300)

    # Optional: save scraped data separately for inspection
    if not scraped_df.empty:
        scraped_df.to_csv("scraped_jobs.csv", index=False)
        print("Scraped data saved to: scraped_jobs.csv")
        print(scraped_df[["title", "company_profile", "location", "fraudulent"]].head(5))

    # --- MERGE ---
    combined_df = merge_datasets(
        kaggle_path="fake_job_postings.csv",
        scraped_df=scraped_df
    )

    print("\nDone! Use 'merged_job_postings.csv' as input for your model training.")
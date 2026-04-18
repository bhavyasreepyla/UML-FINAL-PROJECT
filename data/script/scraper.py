"""
Scrape article text from a list of URLs, saving results to CSV.

This script is designed to be robust to interruptions: it saves progress after each article, and can be safely re-run to pick up where it left off.

Usage:
    1. Ensure you have the required libraries by running: *fill in*
    2. Place your input CSV (with a "URL" column) in the same directory as this script, named "ML_untagged_data-TITLES.csv".
    3. Run the script: `python scraper.py`
    4. Output will be saved to "article_output.csv", and any failures will be logged to "scrape_failures.csv".
    5. If you need to stop the script, simply interrupt it (Ctrl+C); it will finish the current article and save progress.

Notes:
    - The script uses a checkpoint file ("scrape_checkpoint.txt") to track completed URLs, allowing it to resume without duplication.
    - Adjust the BATCH_SIZE and REQUEST_DELAY constants as needed to balance speed and server load.
    - Ensure that the input CSV has a column named "URL" with the article URLs to scrape.
"""

import requests
from bs4 import BeautifulSoup
import sys
import os
import time
import signal
import pandas as pd

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

COOKIES = {
    "fp.user": '{"provider":"web-direct","id_token":"eyJhbGciOiJSUzI1NiIsImtpZCI6IjNDRDJCOEUxQkQ0MzM5MDI1QUVBNzgzMTFDREQyMzM0MTYzMjY0MUVSUzI1NiIsInR5cCI6IkpXVCIsIng1dCI6IlBOSzQ0YjFET1FKYTZuZ3hITjBqTkJZeVpCNCJ9.eyJuYmYiOjE3NzQxNzE3MDYsImV4cCI6MTc3NjU5MDkwNiwiaXNzIjoiaHR0cHM6Ly9hY2NvdW50cy5tZXRsbi5vcmciLCJhdWQiOiJtZXRsbiIsImlhdCI6MTc3NDE3MTcwNiwiYXRfaGFzaCI6IldLZFdpWXRFaHZyb3BxaUI2VTRpR1EiLCJzX2hhc2giOiJ4bkR3OHIwTTktY0ZsUWYtanJDS0hBIiwic2lkIjoiNTJGQkUyMjZEMDE0RDg3Rjc2RkUxQUQyMjlDMTREREYiLCJzdWIiOiIzMjAwMmYyNC05YzdkLTQ0ZjctYTEzYS04MzA3MmIwNzRkMDciLCJhdXRoX3RpbWUiOjE3NzQxNzE2OTQsImlkcCI6ImxvY2FsIiwibmFtZSI6IktodWUgUGhhbSIsImFsaWFzIjoicGhhbS5raHUiLCJlbWFpbCI6InBoYW0ua2h1QG5vcnRoZWFzdGVybi5lZHUiLCJhbXIiOlsicHdkIl19.Yd5zLUNRxREfB7EfyMlgzCyUqvHYIAAeC77NeBrSBJg_RLn_EZbYOJS91JLVtR00njgbtomt4rCa52c2I8Jtj0GaiKx5YadhrFHbjGzolenrMf-Gq9L_LgMOSISON_rt2rnN8mwOpLcnuyjdDgVc-e8Gr1TmRoe4TsiiwquyEu0aACnPeFgWg2nWqhElBp7912l6q6oLHyYTwnNkrEfRQZkD3KjmzuaG8sh1EziE5OUTQ5kPwpXOhOmbVpVeLumA3pSUQCd5_d_9D-5sHkm9QjY9fX1k9dBB-Kq_Ph_fw09V2zhNw5pmDBK8QjuQXRVX78SQ8IK7AcpytvZJqFeuRp07C5z50okB1pRGdtrgBvLLJ-2iQMpYljb8mrf6xpP_CVUGEssBuXnVVnG8H_9u0-T7KSZciiELWanSCEBQv0j5TzijXzracomjtdf3BU6JAERnm1rp-jq5WMb1OB6OICVyuO_8U8lM0PKu5ZbovCalSZhClFj7sKDAUWEXVMMNY8SXQdi9btGmETM5KhnYVBwOTUorc_bGChgqd7Fg_dFsi_LPseu4fizFBaqZwMJi-xYB_g8QxAcgHWTdk0bnjl-wq8Mvsn6x_cs7pRSSHXxKfrEpmZa7EbSiy2xbeiroWY1VL25fDF8ygTsiLsX-XBMX-_eDm9zqe4xwZROvQo8","access_token":"FCAA091BD8CEF7BF51ED006AE67CC1BCFEA18855A41499C549287E64A588860B","expires_in":2419200,"token_type":"bearer","refresh_token":"C18A4D1155557613BDE2865CE6CEAB6ED5223B174B43CC8EA29F23E910BD9271","scope":"read write charge openid offline_access","session_state":"6TNb0IBAGIj0ikDGLvnqEsa3B9TzgrNzeeyfHb-J_-w.D258C52F975D957B9E104C22ACC78B52","expires_at_datetime":"2026-04-19T09:28:00.161Z","access_token_expiration_date_expired":false,"registered":true,"access_token_expiration_date":"2026-04-19T09:28:00.161Z","access_token_expired":false,"subscriber":false}',
    "ck-cid": "U3BxBPqtvbgB",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

SCRAPE_INPUT_CSV = "ML_untagged_data-TITLES.csv"
SCRAPE_OUTPUT_CSV = "article_output.csv"
FAILURES_CSV = "scrape_failures.csv"
CHECKPOINT_FILE = "scrape_checkpoint.txt"

FINAL_OUTPUT_CSV = "final_article_output.csv"

BATCH_SIZE = 10  # How many articles to scrape per batch before flushing to disk.
REQUEST_DELAY = 1.0  # Seconds to wait between requests.


def load_checkpoint() -> set:
    """Return the set of URLs that were successfully scraped in a prior run."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as fh:
        return {line.strip() for line in fh if line.strip()}


def save_checkpoint(url: str) -> None:
    """Append a single completed URL to the checkpoint file."""
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as fh:
        fh.write(url + "\n")


def flush_rows(rows: list[dict], path: str) -> None:
    """Append rows to path, writing the header only when the file is new."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=write_header, encoding="utf-8-sig")


def scrape_article(url: str) -> dict:
    """Scrape the article at the given URL."""
    response = requests.get(url, cookies=COOKIES, headers=HEADERS, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    article_tag = soup.find("article")
    if not article_tag:
        raise ValueError("No <article> tag found.")

    entry_content = article_tag.find("div", class_="entry-content")
    if not entry_content:
        raise ValueError("No div.entry-content found inside <article>.")

    paragraphs = [
        p.get_text(strip=True)
        for p in entry_content.find_all("p")
        if p.get_text(strip=True)
    ]
    if not paragraphs:
        raise ValueError("No <p> text found.")

    title = ""
    title_tag = soup.find("h1")
    if title_tag:
        title = title_tag.get_text(strip=True)

    return {"url": url, "title": title, "paragraphs": paragraphs}


def combine_paragraphs(input_csv: str, SCRAPE_OUTPUT_CSV: str) -> None:
    """Combine scraped paragraphs back into single rows per article."""
    df = pd.read_csv(input_csv)
    combined_df = df.groupby("url", as_index=False).agg(
        title=("title", "first"),
        text=("text", lambda s: " ".join(s.dropna().astype(str))),
    )
    combined_df.drop("title", axis=1).to_csv(SCRAPE_OUTPUT_CSV, index=False)


_interrupted = False


def _handle_signal(signum, frame):
    global _interrupted
    print("\n\n[!] Interrupt received — finishing current article then saving…")
    _interrupted = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def main():
    input_df = pd.read_csv(SCRAPE_INPUT_CSV)
    urls = input_df["URL"].dropna().unique().tolist()
    print(f"Total URLs in input: {len(urls)}")

    done = load_checkpoint()
    remaining = [u for u in urls if u not in done]
    print(f"Already completed: {len(done)}  |  Remaining: {len(remaining)}\n")

    pending_rows: list[dict] = []
    pending_failures: list[dict] = []
    success_count = 0
    fail_count = 0

    # Process in batches.
    for batch_start in range(0, len(remaining), BATCH_SIZE):
        if _interrupted:
            break

        batch = remaining[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"--- Batch {batch_num}/{total_batches} ({len(batch)} URLs) ---")

        for url in batch:
            if _interrupted:
                break

            global_idx = urls.index(url) + 1
            print(f"[{global_idx}/{len(urls)}] Fetching: {url}")

            try:
                result = scrape_article(url)

                for para_num, text in enumerate(result["paragraphs"], 1):
                    pending_rows.append(
                        {
                            "url": result["url"],
                            "title": result["title"],
                            "paragraph": para_num,
                            "text": text,
                        }
                    )

                save_checkpoint(url)
                success_count += 1
                print(f"{len(result['paragraphs'])} paragraphs")

            except Exception as exc:
                fail_count += 1
                print(f"FAILED: {exc}")
                pending_failures.append({"url": url, "error": str(exc)})

            time.sleep(REQUEST_DELAY)

        flush_rows(pending_rows, SCRAPE_OUTPUT_CSV)
        flush_rows(pending_failures, FAILURES_CSV)
        pending_rows.clear()
        pending_failures.clear()
        print(f"Flushed batch {batch_num} to disk.\n")

    print("=" * 60)
    if _interrupted:
        print("Run interrupted.  Progress has been saved.")
    else:
        print("Scraping complete.")

    print(f"  Succeeded : {success_count}")
    print(f"  Failed    : {fail_count}")
    print(f"  Output    : {SCRAPE_OUTPUT_CSV}")
    if fail_count:
        print(f"  Failures  : {FAILURES_CSV}")
    print("=" * 60)

    combine_paragraphs(SCRAPE_OUTPUT_CSV, FINAL_OUTPUT_CSV)

    # TODO: Evan did note that we should filter out articles with recirculation rate > 1 as these are invalid but we ended up not dealing with that metrics


if __name__ == "__main__":
    main()

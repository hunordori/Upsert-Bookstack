import os
import json
import requests
import psycopg2
from pgvector.psycopg2 import register_vector
from datetime import datetime

# --- 1. CONFIGURATION ---
# --- Fill in your details here ---

# BookStack Configuration
BOOKSTACK_URL = "https://wiki.abelcine.com"  # Your BookStack base URL
BOOKSTACK_TOKEN_ID = "mVibdoYPZyjuNGkzfV2jlqJXIQ9NXXN0"
BOOKSTACK_TOKEN_SECRET = "TMvvKQoGeQeKn1OZKvzHG2Nb6Zvbb1ph"

# Ollama Configuration
OLLAMA_HOST = "http://jerry-ollama.abelcine.com:11434"
OLLAMA_MODEL = "mxbai-embed-large"
EMBEDDING_DIMENSION = 1024

# PostgreSQL Configuration
DB_CONNECTION_STRING = "postgresql://flowisedbuser:O7Q6ox3xFEQPDX4ZEuqvHzXKLaf29iTr@jerry-docker1.abelcine.com:5432/flowise-db"
DB_TABLE_NAME = "abelcinewiki"


def setup_database(conn, force_recreate=False):
    """Enables pgvector and creates the table if it doesn't exist."""
    print("Setting up table...")
    try:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

        if force_recreate:
            print(
                f"Dropping existing table {DB_TABLE_NAME} to recreate with new schema..."
            )
            cur.execute(f"DROP TABLE IF EXISTS {DB_TABLE_NAME};")
            conn.commit()

        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {DB_TABLE_NAME} (
            id INT PRIMARY KEY,
            book_id INT,
            chapter_id INT,
            name TEXT,
            slug TEXT,
            content TEXT,
            content_type VARCHAR(10) DEFAULT 'html',
            created_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ,
            created_by JSONB,
            updated_by JSONB,
            metadata JSONB,
            embedding vector({EMBEDDING_DIMENSION}),
            processed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """
        cur.execute(create_table_query)
        conn.commit()

        index_queries = [
            f"CREATE INDEX IF NOT EXISTS idx_{DB_TABLE_NAME}_book_id ON {DB_TABLE_NAME}(book_id);",
            f"CREATE INDEX IF NOT EXISTS idx_{DB_TABLE_NAME}_updated_at ON {DB_TABLE_NAME}(updated_at);",
        ]

        for index_query in index_queries:
            cur.execute(index_query)
            conn.commit()

        cur.close()
        table_action = "recreated" if force_recreate else "verified"
        print(f"Database setup complete. Table '{DB_TABLE_NAME}' {table_action}.")
        return force_recreate
    except Exception as e:
        print(f"Error during database setup: {e}")
        raise


def get_embedding(text: str) -> list[float]:
    """Generates an embedding for the given text using Ollama."""
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": OLLAMA_MODEL, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to Ollama at {OLLAMA_HOST}.")
        print("Please ensure the Ollama server is running.")
        raise
    except Exception as e:
        print(f"Error getting embedding from Ollama: {e}")
        raise


def get_all_bookstack_pages():
    """Fetches ALL pages from the BookStack API with improved pagination."""
    headers = {
        "Authorization": f"Token {BOOKSTACK_TOKEN_ID}:{BOOKSTACK_TOKEN_SECRET}"
    }
    all_pages = []
    page = 1
    max_pages_per_request = 500  # Try larger batch size
    consecutive_empty_pages = 0
    max_empty_pages = 3  # Stop after 3 consecutive empty responses

    print("Starting to fetch all BookStack pages...")

    while True:
        # Try larger count first, fall back to 100 if it fails
        for count in [max_pages_per_request, 100]:
            url = f"{BOOKSTACK_URL}/api/pages?page={page}&count={count}"
            print(f"\n--- API Call: Page {page}, requesting {count} items ---")
            print(f"URL: {url}")

            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                data = response.json()
                break  # Success, exit the count retry loop
            except Exception as e:
                print(f"Failed with count={count}: {e}")
                if count == 100:  # If even 100 fails, raise the error
                    raise
                continue  # Try with smaller count

        # Debug pagination info
        print("API Response structure:")
        for key, value in data.items():
            if key != "data":
                print(f"  {key}: {value}")
            else:
                print(f"  data: [array of {len(value)} items]")

        pages = data.get("data", [])
        current_page = data.get("current_page", page)
        last_page = data.get("last_page")
        total = data.get("total")
        per_page = data.get("per_page", len(pages))

        if not pages:
            consecutive_empty_pages += 1
            print(f"Empty response #{consecutive_empty_pages}")
            if consecutive_empty_pages >= max_empty_pages:
                print(f"Stopping after {max_empty_pages} consecutive empty responses")
                break
        else:
            consecutive_empty_pages = 0
            all_pages.extend(pages)
            print(f"Added {len(pages)} pages. Total so far: {len(all_pages)}")

        # Check various stopping conditions
        should_stop = False
        if last_page is not None and current_page >= last_page:
            print(f"Reached last page: {current_page}/{last_page}")
            should_stop = True
        elif total is not None and len(all_pages) >= total:
            print(f"Collected all available pages: {len(all_pages)}/{total}")
            should_stop = True
        elif len(pages) < per_page and per_page > 0:
            print(f"Partial page received: {len(pages)}/{per_page}")
            should_stop = True

        if should_stop:
            break

        page += 1

        # Increased safety valve - adjust as needed
        if page > 100:  # Allow up to 100 API calls (10,000+ pages)
            print(
                f"Safety stop: More than 100 API pages. If you have more pages, increase this limit."
            )
            break

    print(f"\n=== Pagination Complete: Found {len(all_pages)} total pages ===")
    return all_pages


def get_page_details(page_id):
    """Fetch full page details including HTML content."""
    headers = {
        "Authorization": f"Token {BOOKSTACK_TOKEN_ID}:{BOOKSTACK_TOKEN_SECRET}"
    }
    url = f"{BOOKSTACK_URL}/api/pages/{page_id}"

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching details for page ID {page_id}: {e}")
        return None


def parse_bookstack_datetime(datetime_str):
    """Parse BookStack datetime string to timezone-aware datetime object."""
    if not datetime_str:
        return None
    try:
        datetime_str = datetime_str.replace("Z", "+00:00")
        return datetime.fromisoformat(datetime_str)
    except Exception as e:
        print(f"Error parsing datetime '{datetime_str}': {e}")
        return None


def main():
    """Main function to run the ETL process with detailed tracking."""
    print("Connecting to the database...")

    conn = psycopg2.connect(DB_CONNECTION_STRING)
    register_vector(conn)

    # Track all processing results
    processing_log = {
        "successful": [],
        "skipped": [],
        "failed_fetch": [],
        "failed_no_content": [],
        "failed_embedding": [],
        "failed_database": [],
    }

    try:
        table_was_recreated = setup_database(conn, force_recreate=True)
        pages = get_all_bookstack_pages()

        if not pages:
            print("No pages found. Exiting.")
            return

        cur = conn.cursor()
        total_pages = len(pages)

        for i, page_summary in enumerate(pages):
            page_id = page_summary["id"]
            page_name = page_summary["name"]
            page_updated_at = page_summary.get("updated_at")

            print(f"\n[{i+1}/{total_pages}] Processing: '{page_name}' (ID: {page_id})")

            # Since we recreated the table, all pages need to be processed
            if not table_was_recreated:
                # Only check for updates if we didn't recreate the table
                cur.execute(
                    f"SELECT updated_at FROM {DB_TABLE_NAME} WHERE id = %s", (page_id,)
                )
                result = cur.fetchone()
                if result:
                    db_updated_at = result[0]
                    page_timestamp = parse_bookstack_datetime(page_updated_at)
                    if page_timestamp and page_timestamp <= db_updated_at:
                        print("  → SKIPPED (up to date)")
                        processing_log["skipped"].append(
                            {"id": page_id, "name": page_name}
                        )
                        continue

            # Fetch page details
            print("  → Fetching details...")
            page_details = get_page_details(page_id)
            if not page_details:
                print("  → FAILED (could not fetch details)")
                processing_log["failed_fetch"].append(
                    {"id": page_id, "name": page_name}
                )
                continue

            # Check for content
            content = page_details.get("html", "")
            if not content:
                print("  → FAILED (no HTML content)")
                processing_log["failed_no_content"].append(
                    {"id": page_id, "name": page_name}
                )
                continue

            # Generate embedding
            try:
                print("  → Generating embedding...")
                embedding = get_embedding(content)
            except Exception as e:
                print(f"  → FAILED (embedding error): {e}")
                processing_log["failed_embedding"].append(
                    {"id": page_id, "name": page_name, "error": str(e)}
                )
                continue

            # Insert into database
            try:
                print("  → Inserting into database...")

                metadata = {
                    "source": f"{BOOKSTACK_URL}/books/{page_details.get('book', {}).get('slug', 'unknown')}/page/{page_details.get('slug', 'unknown')}",
                    "book_name": page_details.get("book", {}).get("name"),
                    "book_slug": page_details.get("book", {}).get("slug"),
                    "chapter_name": page_details.get("chapter", {}).get("name")
                    if page_details.get("chapter")
                    else None,
                    "priority": page_details.get("priority"),
                    "template": page_details.get("template"),
                    "revision_count": page_details.get("revision_count", 0),
                    "editor": page_details.get("editor"),
                    "draft": page_details.get("draft"),
                }

                upsert_query = f"""
                INSERT INTO {DB_TABLE_NAME} (
                    id, book_id, chapter_id, name, slug, content, content_type,
                    created_at, updated_at, created_by, updated_by, metadata, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    book_id = EXCLUDED.book_id,
                    chapter_id = EXCLUDED.chapter_id,
                    name = EXCLUDED.name,
                    slug = EXCLUDED.slug,
                    content = EXCLUDED.content,
                    content_type = EXCLUDED.content_type,
                    created_at = EXCLUDED.created_at,
                    updated_at = EXCLUDED.updated_at,
                    created_by = EXCLUDED.created_by,
                    updated_by = EXCLUDED.updated_by,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    processed_at = CURRENT_TIMESTAMP;
                """

                cur.execute(
                    upsert_query,
                    (
                        page_id,
                        page_details.get("book_id"),
                        page_details.get("chapter_id"),
                        page_details.get("name"),
                        page_details.get("slug"),
                        content,
                        "html",
                        parse_bookstack_datetime(page_details.get("created_at")),
                        parse_bookstack_datetime(page_details.get("updated_at")),
                        json.dumps(page_details.get("created_by", {})),
                        json.dumps(page_details.get("updated_by", {})),
                        json.dumps(metadata),
                        embedding,
                    ),
                )
                conn.commit()
                print("  → SUCCESS")
                processing_log["successful"].append(
                    {"id": page_id, "name": page_name}
                )

            except Exception as e:
                print(f"  → FAILED (database error): {e}")
                processing_log["failed_database"].append(
                    {"id": page_id, "name": page_name, "error": str(e)}
                )
                continue

        # Final report
        cur.execute(f"SELECT COUNT(*) FROM {DB_TABLE_NAME}")
        final_db_count = cur.fetchone()[0]
        cur.close()

        print(f"\n{'='*60}")
        print("FINAL PROCESSING REPORT")
        print(f"{'='*60}")
        print(f"Total pages found in API: {total_pages}")
        print(f"Successfully processed: {len(processing_log['successful'])}")
        print(f"Skipped (up to date): {len(processing_log['skipped'])}")
        print(f"Failed to fetch details: {len(processing_log['failed_fetch'])}")
        print(f"Failed (no content): {len(processing_log['failed_no_content'])}")
        print(f"Failed (embedding): {len(processing_log['failed_embedding'])}")
        print(f"Failed (database): {len(processing_log['failed_database'])}")
        print(f"Final database count: {final_db_count}")

        # Show detailed failures
        for category, failures in processing_log.items():
            if failures and "failed" in category:
                print(f"\n{category.upper().replace('_', ' ')}:")
                for failure in failures[:10]:  # Show first 10
                    print(f"  - ID {failure['id']}: {failure['name']}")
                if len(failures) > 10:
                    print(f"  ... and {len(failures) - 10} more")

    except Exception as e:
        print(f"\nCritical error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")


if __name__ == "__main__":
    main()

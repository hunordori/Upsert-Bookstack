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
# IMPORTANT: The vector dimension MUST match your Ollama model.
# - mxbai-embed-large: 1024
# - nomic-embed-text: 768
# - all-minilm: 384
EMBEDDING_DIMENSION = 1024

# PostgreSQL Configuration
DB_CONNECTION_STRING = "postgresql://flowisedbuser:O7Q6ox3xFEQPDX4ZEuqvHzXKLaf29iTr@jerry-docker1.abelcine.com:5432/flowise-db"
DB_TABLE_NAME = "abelcinewiki"

def setup_database(conn):
    """Enables pgvector and creates the table if it doesn't exist."""
    print("Setting up table...")
    try:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

        # Drop existing table to start fresh with new schema
        print(f"Dropping existing table {DB_TABLE_NAME} to recreate with new schema...")
        cur.execute(f"DROP TABLE IF EXISTS {DB_TABLE_NAME};")
        conn.commit()

        # Create the table with new schema
        create_table_query = f"""
        CREATE TABLE {DB_TABLE_NAME} (
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

        # Create indexes
        index_queries = [
            f"CREATE INDEX idx_{DB_TABLE_NAME}_book_id ON {DB_TABLE_NAME}(book_id);",
            f"CREATE INDEX idx_{DB_TABLE_NAME}_updated_at ON {DB_TABLE_NAME}(updated_at);"
        ]

        for index_query in index_queries:
            cur.execute(index_query)
            conn.commit()

        cur.close()
        print(f"Database setup complete. Table '{DB_TABLE_NAME}' recreated with new schema.")
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
    """Fetches ALL pages from the BookStack API, handling pagination with detailed debugging."""
    headers = {"Authorization": f"Token {BOOKSTACK_TOKEN_ID}:{BOOKSTACK_TOKEN_SECRET}"}
    all_pages = []
    page = 1

    while True:
        # Try different pagination parameter formats
        url = f"{BOOKSTACK_URL}/api/pages?page={page}&count=100"
        print(f"\n--- API Call: Fetching page {page} from {url} ---")

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            # DEBUG: Print the full structure every time to see what's happening
            print("DEBUG: Full API response structure:")
            print(f"Keys in response: {list(data.keys())}")
            for key, value in data.items():
                if key != "data":
                    print(f"  {key}: {value}")
                else:
                    print(f"  data: [array of {len(value)} items]")

            pages = data.get("data", [])
            if not pages:
                print("No pages in response. Breaking.")
                break

            all_pages.extend(pages)

            # Try multiple ways to detect if there are more pages
            current_page = data.get("current_page", page)
            last_page = data.get("last_page")
            total = data.get("total")
            per_page = data.get("per_page", len(pages))

            print(f"Pagination analysis:")
            print(f"  - current_page: {current_page}")
            print(f"  - last_page: {last_page}")
            print(f"  - total: {total}")
            print(f"  - per_page: {per_page}")
            print(f"  - pages in this response: {len(pages)}")
            print(f"  - cumulative pages: {len(all_pages)}")

            # Multiple stopping conditions
            should_stop = False

            if last_page is not None and current_page >= last_page:
                print(f"Stopping: current_page ({current_page}) >= last_page ({last_page})")
                should_stop = True
            elif len(pages) < per_page and per_page > 0:
                print(f"Stopping: got {len(pages)} pages, expected {per_page} (partial page)")
                should_stop = True
            elif len(pages) == 0:
                print("Stopping: no pages in response")
                should_stop = True
            elif total is not None and len(all_pages) >= total:
                print(f"Stopping: collected {len(all_pages)} pages, total is {total}")
                should_stop = True

            if should_stop:
                break

            page += 1

            # Safety valve
            if page > 20:  # More than 20 pages seems unlikely, but adjust if needed
                print("Safety stop: More than 20 API pages. Something might be wrong.")
                break

        except Exception as e:
            print(f"Error fetching page {page} from BookStack: {e}")
            raise

    print(f"\n=== Found {len(all_pages)} total pages ===")
    return all_pages


def get_page_details(page_id):
    """Fetch full page details including HTML content."""
    headers = {"Authorization": f"Token {BOOKSTACK_TOKEN_ID}:{BOOKSTACK_TOKEN_SECRET}"}
    url = f"{BOOKSTACK_URL}/api/pages/{page_id}"

    try:
        response = requests.get(url, headers=headers)
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
        # BookStack returns format like "2023-02-03T01:27:58.000000Z"
        # Replace Z with +00:00 for proper timezone parsing
        datetime_str = datetime_str.replace('Z', '+00:00')
        return datetime.fromisoformat(datetime_str)
    except Exception as e:
        print(f"Error parsing datetime '{datetime_str}': {e}")
        return None


def page_needs_update(cur, page_id, page_updated_at):
    """Check if a page needs to be updated based on its updated_at timestamp."""
    try:
        cur.execute(f"SELECT updated_at FROM {DB_TABLE_NAME} WHERE id = %s", (page_id,))
        result = cur.fetchone()

        if result is None:
            return True  # Page doesn't exist, needs to be inserted

        db_updated_at = result[0]  # This is already timezone-aware from TIMESTAMPTZ

        # Parse the API timestamp
        page_timestamp = parse_bookstack_datetime(page_updated_at)
        if page_timestamp is None:
            return True  # If we can't parse the timestamp, update it

        # Compare timezone-aware datetimes
        needs_update = page_timestamp > db_updated_at
        print(f"  Page timestamp: {page_timestamp}, DB timestamp: {db_updated_at}, needs update: {needs_update}")
        return needs_update

    except Exception as e:
        print(f"Error checking if page {page_id} needs update: {e}")
        return True  # If in doubt, update it


def main():
    """Main function to run the ETL process."""
    print("Connecting to the database...")

    conn = psycopg2.connect(DB_CONNECTION_STRING)
    register_vector(conn)

    try:
        setup_database(conn)
        pages = get_all_bookstack_pages()

        if not pages:
            print("No pages found. Exiting.")
            return

        cur = conn.cursor()
        updated_count = 0
        skipped_count = 0

        for i, page_summary in enumerate(pages):
            page_id = page_summary["id"]
            page_name = page_summary["name"]
            page_updated_at = page_summary.get("updated_at")

            print(f"\n--- Processing page {i + 1}/{len(pages)}: '{page_name}' (ID: {page_id}) ---")

            # Check if page needs updating
            if not page_needs_update(cur, page_id, page_updated_at):
                print(f"Page '{page_name}' is up to date. Skipping.")
                skipped_count += 1
                continue

            # Get full page details including HTML content
            print("Fetching full page details...")
            page_details = get_page_details(page_id)

            if not page_details:
                print(f"Could not fetch details for page ID {page_id}. Skipping.")
                continue

            # Extract HTML content
            content = page_details.get("html", "")
            if not content:
                print(f"No HTML content found for page ID {page_id}. Skipping.")
                continue

            print("Generating embedding...")
            embedding = get_embedding(content)

            # Extract comprehensive metadata
            metadata = {
                "source": f"{BOOKSTACK_URL}/books/{page_details.get('book', {}).get('slug', 'unknown')}/page/{page_details.get('slug', 'unknown')}",
                "book_name": page_details.get("book", {}).get("name"),
                "book_slug": page_details.get("book", {}).get("slug"),
                "chapter_name": page_details.get("chapter", {}).get("name") if page_details.get("chapter") else None,
                "priority": page_details.get("priority"),
                "template": page_details.get("template"),
                "revision_count": page_details.get("revision_count", 0),
                "editor": page_details.get("editor"),
                "draft": page_details.get("draft"),
            }

            print("Upserting data into PostgreSQL...")
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

            cur.execute(upsert_query, (
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
                embedding
            ))
            conn.commit()
            print(f"Successfully upserted page '{page_name}'.")
            updated_count += 1

        cur.close()
        print(f"\n=== PROCESS COMPLETE ===")
        print(f"Total pages processed: {len(pages)}")
        print(f"Pages updated/inserted: {updated_count}")
        print(f"Pages skipped (up to date): {skipped_count}")

    except Exception as e:
        print(f"\nAn error occurred in the main process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    main()

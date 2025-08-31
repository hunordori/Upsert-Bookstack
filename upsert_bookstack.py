import os
import json
import requests
import psycopg2
from pgvector.psycopg2 import register_vector
from datetime import datetime
from bs4 import BeautifulSoup
import re
import uuid
import time
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()

# BookStack Configuration
BOOKSTACK_URL = os.getenv("BOOKSTACK_URL")
BOOKSTACK_TOKEN_ID = os.getenv("BOOKSTACK_TOKEN_ID")
BOOKSTACK_TOKEN_SECRET = os.getenv("BOOKSTACK_TOKEN_SECRET")

# Ollama Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION"))

# PostgreSQL Configuration
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
DOCS_TABLE_NAME = os.getenv("DOCS_TABLE_NAME")
RECORDS_TABLE_NAME = os.getenv("RECORDS_TABLE_NAME")

# Flowise Configuration
NAMESPACE = os.getenv("NAMESPACE")
GROUP_ID = os.getenv("GROUP_ID")

# Text Processing Configuration
ENABLE_CHUNKING = os.getenv("ENABLE_CHUNKING")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
EXCLUDED_SHELF_IDS = os.getenv("EXCLUDED_SHELF_IDS", "").split(",") if os.getenv("EXCLUDED_SHELF_IDS") else []
EXCLUDED_SHELF_IDS = [int(id.strip()) for id in EXCLUDED_SHELF_IDS if id.strip().isdigit()]

# Cache for book details to avoid repeated API calls
book_cache = {}


def preprocess_bookstack_html(raw_html: str) -> str:
    """Extract clean text content from BookStack HTML."""
    if not raw_html:
        return ""

    soup = BeautifulSoup(raw_html, 'html.parser')

    # Remove navigation, sidebar, footer elements
    for elem in soup.find_all(['nav', 'aside', 'footer', 'script', 'style']):
        elem.decompose()

    # Remove common BookStack UI elements
    for selector in ['.page-nav', '.sidebar', '.header', '.breadcrumbs']:
        for elem in soup.select(selector):
            elem.decompose()

    # Get main content
    main_content = soup.find('main') or soup.find('.page-content') or soup

    # Extract clean text
    text = main_content.get_text(separator=' ', strip=True)
    return re.sub(r'\s+', ' ', text)  # Normalize whitespace


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    if not text:
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk = ' '.join(chunk_words)
        chunks.append(chunk)

        if i + chunk_size >= len(words):
            break

    return chunks


def get_book_details(book_id):
    """Fetch book details from the books API with caching."""
    if book_id in book_cache:
        return book_cache[book_id]

    headers = {
        "Authorization": f"Token {BOOKSTACK_TOKEN_ID}:{BOOKSTACK_TOKEN_SECRET}"
    }
    url = f"{BOOKSTACK_URL}/api/books/{book_id}"

    try:
        print(f"    → Fetching book details for book ID {book_id}...")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        book_data = response.json()

        # Cache the book details
        book_cache[book_id] = book_data
        print(f"    → Cached book: '{book_data.get('name', 'Unknown')}' (slug: {book_data.get('slug', 'unknown')})")
        return book_data
    except Exception as e:
        print(f"    → Error fetching book details for book ID {book_id}: {e}")
        # Return a fallback book object
        fallback = {
            "id": book_id,
            "name": f"Book {book_id}",
            "slug": f"book-{book_id}",
            "description": "",
        }
        book_cache[book_id] = fallback
        return fallback


def get_chapter_details(chapter_id):
    """Fetch chapter details from the chapters API (if needed)."""
    if not chapter_id or chapter_id == 0:
        return None

    headers = {
        "Authorization": f"Token {BOOKSTACK_TOKEN_ID}:{BOOKSTACK_TOKEN_SECRET}"
    }
    url = f"{BOOKSTACK_URL}/api/chapters/{chapter_id}"

    try:
        print(f"    → Fetching chapter details for chapter ID {chapter_id}...")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"    → Error fetching chapter details for chapter ID {chapter_id}: {e}")
        return None

# --- Shelves support: build a book_id -> shelves map and expose a simple fetch_all pager ---
from urllib.parse import urljoin

def _bookstack_auth_header():
    return {"Authorization": f"Token {BOOKSTACK_TOKEN_ID}:{BOOKSTACK_TOKEN_SECRET}", "Accept": "application/json"}

def fetch_all(session: requests.Session, path: str, params=None):
    """Yield all items from a paginated BookStack endpoint like /api/shelves."""
    params = dict(params or {})
    page = 1
    while True:
        params["page"] = page
        r = session.get(urljoin(BOOKSTACK_URL.rstrip('/') + '/', path.lstrip('/')), headers=_bookstack_auth_header(), params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        items = data.get("data") or data.get("items") or []
        for it in items:
            yield it
        cur = data.get("current_page")
        last = data.get("last_page")
        if not cur or not last or cur >= last:
            break
        page += 1

def build_book_to_shelves_map(session: requests.Session):
    """
    Returns a dict mapping book_id -> list of shelves ({id, name, slug}).
    Uses /api/shelves (list). If a shelf item lacks 'books', falls back to /api/shelves/{id}.
    """
    book_to_shelves = {}
    # Iterate shelves with pagination
    for shelf in fetch_all(session, "/api/shelves"):
        shelf_id = shelf.get("id")
        if shelf_id is None:
            continue
        shelf_name = shelf.get("name")
        shelf_slug = shelf.get("slug")

        books = shelf.get("books")
        # Fallback: fetch shelf detail to get books array if not included on list item
        if books is None:
            r = session.get(urljoin(BOOKSTACK_URL.rstrip('/') + '/', f"/api/shelves/{shelf_id}"), headers=_bookstack_auth_header(), timeout=30)
            r.raise_for_status()
            detail = r.json()
            books = detail.get("books", [])

        for b in (books or []):
            bid = b.get("id")
            if bid is None:
                continue
            book_to_shelves.setdefault(bid, []).append({
                "id": shelf_id, "name": shelf_name, "slug": shelf_slug
            })

    return book_to_shelves

def setup_database(conn, force_recreate=True):
    """Creates Flowise-compatible tables."""
    print("Setting up Flowise-compatible tables...")
    try:
        cur = conn.cursor()

        # Enable required extensions
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
        conn.commit()

        if force_recreate:
            print(f"Dropping existing tables to recreate with Flowise schema...")
            cur.execute(f"DROP TABLE IF EXISTS {RECORDS_TABLE_NAME};")
            cur.execute(f"DROP TABLE IF EXISTS {DOCS_TABLE_NAME};")
            conn.commit()

        # Create upserted_docs table
        create_docs_table_query = f"""
        CREATE TABLE IF NOT EXISTS {DOCS_TABLE_NAME} (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            "pageContent" TEXT NOT NULL,
            metadata JSONB,
            embedding vector({EMBEDDING_DIMENSION})
        );
        """
        cur.execute(create_docs_table_query)
        conn.commit()

        # Create upsertion_records table
        create_records_table_query = f"""
        CREATE TABLE IF NOT EXISTS {RECORDS_TABLE_NAME} (
            uuid UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            key TEXT NOT NULL,
            namespace TEXT NOT NULL,
            updated_at DOUBLE PRECISION NOT NULL,
            group_id TEXT
        );
        """
        cur.execute(create_records_table_query)
        conn.commit()

        # Create indexes
        index_queries = [
            f"CREATE INDEX IF NOT EXISTS idx_{DOCS_TABLE_NAME}_embedding ON {DOCS_TABLE_NAME} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);",
            f"CREATE INDEX IF NOT EXISTS idx_{RECORDS_TABLE_NAME}_namespace ON {RECORDS_TABLE_NAME}(namespace);",
            f"CREATE INDEX IF NOT EXISTS idx_{RECORDS_TABLE_NAME}_key ON {RECORDS_TABLE_NAME}(key);",
            f"CREATE INDEX IF NOT EXISTS idx_{RECORDS_TABLE_NAME}_updated_at ON {RECORDS_TABLE_NAME}(updated_at);",
        ]

        for index_query in index_queries:
            try:
                cur.execute(index_query)
                conn.commit()
            except Exception as e:
                print(f"Note: Could not create index (may require more data): {e}")

        cur.close()
        table_action = "recreated" if force_recreate else "verified"
        print(f"Flowise tables {table_action}.")
        print(f"Namespace: {NAMESPACE}")
        print(f"Chunking enabled: {ENABLE_CHUNKING}")
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
    max_pages_per_request = 400
    consecutive_empty_pages = 0
    max_empty_pages = 4

    print("Starting to fetch all BookStack pages...")

    while True:
        for count in [max_pages_per_request, 100]:
            url = f"{BOOKSTACK_URL}/api/pages?page={page}&count={count}"
            print(f"\n--- API Call: Page {page}, requesting {count} items ---")

            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                data = response.json()
                break
            except Exception as e:
                print(f"Failed with count={count}: {e}")
                if count == 100:
                    raise
                continue

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
                break
        else:
            consecutive_empty_pages = 0
            all_pages.extend(pages)
            print(f"Added {len(pages)} pages. Total so far: {len(all_pages)}")

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

        if page > 100:
            print(f"Safety stop: More than 100 API pages.")
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


def insert_page_chunks_flowise(cur, page_details, clean_text, conn, book_to_shelves=None):
    """Insert page content into Flowise-compatible tables with proper book/chapter info."""
    page_id = page_details["id"]
    page_name = page_details["name"]
    current_timestamp = time.time()

    # Fetch book details to get the correct slug
    book_id = page_details.get("book_id")
    book_details = get_book_details(book_id) if book_id else None

    # Fetch chapter details if chapter_id exists and is not 0
    chapter_id = page_details.get("chapter_id")
    chapter_details = None
    if chapter_id and chapter_id != 0:
        chapter_details = get_chapter_details(chapter_id)

    # Create source URL with correct book slug and page slug
    book_slug = book_details.get("slug", f"book-{book_id}") if book_details else "unknown"
    page_slug = page_details.get("slug", f"page-{page_id}")
    source_url = f"{BOOKSTACK_URL}/books/{book_slug}/page/{page_slug}"

    # Create base key for this page
    base_key = f"bookstack_page_{page_id}"

    # Delete existing records
    cur.execute(f"DELETE FROM {RECORDS_TABLE_NAME} WHERE key LIKE %s", (f"{base_key}%",))

    # Get existing doc IDs to delete
    cur.execute(
        f"SELECT id FROM {DOCS_TABLE_NAME} WHERE metadata->>'page_id' = %s",
        (str(page_id),)
    )
    existing_doc_ids = [row[0] for row in cur.fetchall()]

    if existing_doc_ids:
        placeholders = ','.join(['%s'] * len(existing_doc_ids))
        cur.execute(f"DELETE FROM {DOCS_TABLE_NAME} WHERE id IN ({placeholders})", existing_doc_ids)

    if ENABLE_CHUNKING:
        chunks = chunk_text(clean_text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"  → Split into {len(chunks)} chunks")
    else:
        chunks = [clean_text]
        print(f"  → Processing as single entry")

    successful_inserts = 0
    inserted_doc_ids = []

    for chunk_index, chunk_content in enumerate(chunks):
        if not chunk_content.strip():
            continue

        try:
            # Generate UUIDs
            doc_uuid = str(uuid.uuid4())
            record_uuid = str(uuid.uuid4())

            # Generate embedding for this chunk
            embedding = get_embedding(chunk_content)

            # Create comprehensive metadata
            doc_metadata = {
                "source": source_url,
                "page_id": str(page_id),
                "page_name": page_name,
                "page_slug": page_slug,
                "book_id": str(book_id) if book_id else None,
                "book_name": book_details.get("name", "") if book_details else "",
                "book_slug": book_slug,
                "book_description": book_details.get("description", "") if book_details else "",
                "shelves": (book_to_shelves.get(book_id, []) if book_to_shelves else []),
"shelf_names": ([s.get("name") for s in book_to_shelves.get(book_id, [])] if book_to_shelves else []),
                "chapter_id": str(chapter_id) if chapter_id and chapter_id != 0 else None,
                "chapter_name": chapter_details.get("name", "") if chapter_details else None,
                "chapter_slug": chapter_details.get("slug", "") if chapter_details else None,
                "created_at": page_details.get("created_at", ""),
                "updated_at": page_details.get("updated_at", ""),
                "created_by": page_details.get("created_by", {}),
                "updated_by": page_details.get("updated_by", {}),
                "owned_by": page_details.get("owned_by", {}),
                "chunk_index": chunk_index,
                "total_chunks": len(chunks),
                "word_count": len(chunk_content.split()),
                "priority": page_details.get("priority", 0),
                "template": page_details.get("template", False),
                "draft": page_details.get("draft", False),
                "revision_count": page_details.get("revision_count", 0),
                "editor": page_details.get("editor", ""),
                "tags": page_details.get("tags", []),
            }

            # Insert into upserted_docs table
            docs_insert_query = f"""
            INSERT INTO {DOCS_TABLE_NAME} (id, "pageContent", metadata, embedding)
            VALUES (%s, %s, %s, %s)
            """

            cur.execute(
                docs_insert_query,
                (doc_uuid, chunk_content, json.dumps(doc_metadata), embedding)
            )

            # Create key for this specific chunk
            chunk_key = f"{base_key}_chunk_{chunk_index}" if ENABLE_CHUNKING else base_key

            # Insert into upsertion_records table
            records_insert_query = f"""
            INSERT INTO {RECORDS_TABLE_NAME} (uuid, key, namespace, updated_at, group_id)
            VALUES (%s, %s, %s, %s, %s)
            """

            cur.execute(
                records_insert_query,
                (record_uuid, chunk_key, NAMESPACE, current_timestamp, GROUP_ID)
            )

            successful_inserts += 1
            inserted_doc_ids.append(doc_uuid)

        except Exception as e:
            print(f"  → Failed to process chunk {chunk_index}: {e}")
            continue

    conn.commit()
    return successful_inserts, inserted_doc_ids

def should_skip_book(book_id, book_to_shelves, excluded_shelf_ids):
    """Check if a book should be skipped based on its shelves."""
    if not excluded_shelf_ids or not book_id:
        return False

    book_shelves = book_to_shelves.get(book_id, [])
    book_shelf_ids = [shelf.get("id") for shelf in book_shelves if shelf.get("id")]

    # Skip if book belongs to any excluded shelf
    return any(shelf_id in excluded_shelf_ids for shelf_id in book_shelf_ids)

def main():
    """Main function to run the ETL process with proper book/chapter resolution."""
    print("Connecting to the database...")

    conn = psycopg2.connect(DB_CONNECTION_STRING)
    register_vector(conn)

    processing_log = {
        "successful": [],
        "skipped": [],
        "failed_fetch": [],
        "failed_no_content": [],
        "failed_processing": [],
    }

    try:
        # Build shelves map (book_id -> shelves) once per run
        with requests.Session() as _session:
            book_to_shelves = build_book_to_shelves_map(_session)
        print(f"Loaded shelves map for {len(book_to_shelves)} books")

        table_was_recreated = setup_database(conn, force_recreate=True)
        pages = get_all_bookstack_pages()

        if not pages:
            print("No pages found. Exiting.")
            return

        cur = conn.cursor()
        total_pages = len(pages)
        total_chunks_inserted = 0
        all_inserted_doc_ids = []

        for i, page_summary in enumerate(pages):
            page_id = page_summary["id"]
            page_name = page_summary["name"]

            print(f"\n[{i+1}/{total_pages}] Processing: '{page_name}' (ID: {page_id})")

            # Fetch page details
            print("  → Fetching page details...")
            page_details = get_page_details(page_id)
            if not page_details:
                print("  → FAILED (could not fetch details)")
                processing_log["failed_fetch"].append(
                    {"id": page_id, "name": page_name}
                )
                continue

            # Check if this page's book should be skipped
            book_id = page_details.get("book_id")
            if should_skip_book(book_id, book_to_shelves, EXCLUDED_SHELF_IDS):
                book_shelves = book_to_shelves.get(book_id, [])
                shelf_names = [s.get("name", "Unknown") for s in book_shelves]
                print(f"  → SKIPPED (book belongs to excluded shelf(s): {', '.join(shelf_names)})")
                processing_log["skipped"].append(
                    {"id": page_id, "name": page_name, "book_id": book_id, "shelves": shelf_names}
                )
                continue

            # Extract and clean HTML content
            raw_html = page_details.get("html", "") or page_details.get("raw_html", "")
            if not raw_html:
                print("  → FAILED (no HTML content)")
                processing_log["failed_no_content"].append(
                    {"id": page_id, "name": page_name}
                )
                continue

            print("  → Cleaning HTML...")
            clean_text = preprocess_bookstack_html(raw_html)

            if not clean_text.strip():
                print("  → FAILED (no text content after cleaning)")
                processing_log["failed_no_content"].append(
                    {"id": page_id, "name": page_name}
                )
                continue

            print(f"  → Extracted {len(clean_text.split())} words")

            # Process and insert chunks
            try:
                chunks_inserted, doc_ids = insert_page_chunks_flowise(cur, page_details, clean_text, conn, book_to_shelves)
                total_chunks_inserted += chunks_inserted
                all_inserted_doc_ids.extend(doc_ids)

                if chunks_inserted > 0:
                    print(f"  → SUCCESS ({chunks_inserted} chunks inserted)")
                    processing_log["successful"].append(
                        {
                            "id": page_id,
                            "name": page_name,
                            "chunks": chunks_inserted,
                            "doc_ids": doc_ids
                        }
                    )
                else:
                    print("  → FAILED (no chunks inserted)")
                    processing_log["failed_processing"].append(
                        {"id": page_id, "name": page_name}
                    )

            except Exception as e:
                print(f"  → FAILED (processing error): {e}")
                processing_log["failed_processing"].append(
                    {"id": page_id, "name": page_name, "error": str(e)}
                )
                continue

        # Final report
        cur.execute(f"SELECT COUNT(*) FROM {DOCS_TABLE_NAME}")
        final_docs_count = cur.fetchone()[0]

        cur.execute(f"SELECT COUNT(*) FROM {RECORDS_TABLE_NAME}")
        final_records_count = cur.fetchone()[0]

        cur.execute(f"SELECT COUNT(DISTINCT metadata->>'page_id') FROM {DOCS_TABLE_NAME}")
        unique_pages_count = cur.fetchone()[0]

        # Show some sample sources to verify correct URLs
        cur.execute(f"SELECT metadata->>'source' FROM {DOCS_TABLE_NAME} LIMIT 5")
        sample_sources = [row[0] for row in cur.fetchall()]

        cur.close()

        print(f"\n{'='*60}")
        print("FINAL PROCESSING REPORT (Flowise Schema)")
        print(f"{'='*60}")
        print(f"Total pages found in API: {total_pages}")
        print(f"Successfully processed pages: {len(processing_log['successful'])}")
        print(f"Total chunks/entries inserted: {total_chunks_inserted}")
        print(f"Skipped (excluded shelves): {len(processing_log['skipped'])}")
        print(f"Failed to fetch details: {len(processing_log['failed_fetch'])}")
        print(f"Failed (no content): {len(processing_log['failed_no_content'])}")
        print(f"Failed (processing): {len(processing_log['failed_processing'])}")
        print(f"Final {DOCS_TABLE_NAME} count: {final_docs_count} entries")
        print(f"Final {RECORDS_TABLE_NAME} count: {final_records_count} entries")
        print(f"Unique pages in database: {unique_pages_count}")
        print(f"Books cached: {len(book_cache)}")
        print(f"Namespace: {NAMESPACE}")

        print(f"\nSample source URLs:")
        for source in sample_sources:
            print(f"  - {source}")

        # Show book cache summary
        print(f"\nCached books:")
        for book_id, book_data in list(book_cache.items())[:10]:  # Show first 10
            print(f"  - ID {book_id}: '{book_data.get('name', 'Unknown')}' (slug: {book_data.get('slug', 'unknown')})")
        if len(book_cache) > 10:
            print(f"  ... and {len(book_cache) - 10} more books")

        # Show detailed failures
        for category, failures in processing_log.items():
            if failures and "failed" in category:
                print(f"\n{category.upper().replace('_', ' ')}:")
                for failure in failures[:10]:
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

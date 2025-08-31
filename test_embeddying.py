import os
from dotenv import load_dotenv
import requests
import psycopg2
from enum import Enum

# Load environment variables
load_dotenv()

# Ollama Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# PostgreSQL Configuration
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
DOCS_TABLE_NAME = os.getenv("DOCS_TABLE_NAME", "upserted_docs")
RECORDS_TABLE_NAME = os.getenv("RECORDS_TABLE_NAME", "upsertion_records")

TEST_SENTENCE = "How can I access Synology?"

class DistanceStrategy(Enum):
    """Available distance strategies for vector similarity"""
    COSINE = "cosine"           # Best for normalized embeddings, range: [0,2]
    EUCLIDEAN = "euclidean"     # L2 distance, good for absolute distances
    INNER_PRODUCT = "inner_product"  # Dot product, good for similarity scores

def get_distance_operator_and_similarity(strategy: DistanceStrategy):
    """Returns the PostgreSQL operator and similarity calculation for each strategy"""
    if strategy == DistanceStrategy.COSINE:
        return "<=>", "1 - (d.embedding <=> %s::vector)"  # Fixed: single <=>
    elif strategy == DistanceStrategy.EUCLIDEAN:
        return "<->", "1.0 / (1.0 + (d.embedding <-> %s::vector))"  # Convert to similarity
    elif strategy == DistanceStrategy.INNER_PRODUCT:
        return "<#>", "-(d.embedding <#> %s::vector)"  # Negative inner product (higher = more similar)
    else:
        raise ValueError(f"Unknown distance strategy: {strategy}")

def get_ollama_embedding(text, model_name, ollama_host):
    """Get embeddings from an external Ollama server"""
    url = f"{ollama_host}/api/embeddings"

    payload = {
        "model": model_name,
        "prompt": text
    }

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["embedding"]
    else:
        raise Exception(f"Ollama API error: {response.text}")

def test_vector_retrieval_ollama(query_text, limit=5, namespace_filter=None, distance_strategy=DistanceStrategy.COSINE):
    try:
        # Get embedding from Ollama
        print(f"🔍 Getting embedding for: '{query_text}'")
        query_embedding = get_ollama_embedding(query_text, OLLAMA_MODEL, OLLAMA_HOST)
        print(f"✅ Got embedding with {len(query_embedding)} dimensions")

        # Get distance operator and similarity calculation
        distance_op, similarity_calc = get_distance_operator_and_similarity(distance_strategy)
        print(f"📏 Using distance strategy: {distance_strategy.value}")

        # Connect to PostgreSQL and perform similarity search
        with psycopg2.connect(DB_CONNECTION_STRING) as conn:
            with conn.cursor() as cur:
                # Build query with specified distance strategy
                base_query = f"""
                    SELECT
                        d.id,
                        d."pageContent",
                        d.metadata,
                        {similarity_calc} AS similarity_score
                    FROM {DOCS_TABLE_NAME} d
                """

                # Add namespace filtering if specified
                if namespace_filter:
                    query = base_query + f"""
                        LEFT JOIN {RECORDS_TABLE_NAME} r ON d.id::text = r.uuid::text
                        WHERE d.embedding IS NOT NULL
                        AND r.namespace = %s
                        ORDER BY d.embedding {distance_op} %s::vector
                        LIMIT %s
                    """
                    params = (query_embedding, namespace_filter, query_embedding, limit)
                else:
                    query = base_query + f"""
                        WHERE d.embedding IS NOT NULL
                        ORDER BY d.embedding {distance_op} %s::vector
                        LIMIT %s
                    """
                    params = (query_embedding, query_embedding, limit)

                cur.execute(query, params)
                results = cur.fetchall()

                # Pretty print results
                print(f"\n{'='*100}")
                print(f"SEARCH RESULTS FOR: '{query_text}'")
                print(f"Model: {OLLAMA_MODEL} | Distance: {distance_strategy.value} | Found: {len(results)} results")
                if namespace_filter:
                    print(f"Namespace filter: {namespace_filter}")
                print(f"{'='*100}\n")

                for i, (doc_id, page_content, metadata, similarity_score) in enumerate(results, 1):
                    print(f"📄 RESULT #{i}")
                    print(f"   ID: {doc_id}")
                    print(f"   {distance_strategy.value.title()} Score: {similarity_score:.4f}")

                    # Show metadata if available
                    if metadata:
                        print(f"   Metadata: {metadata}")

                    print(f"   Page Content:")
                    print(f"   {'-' * 50}")

                    # Handle long content
                    if len(page_content) > 500:
                        truncated = page_content[:500] + "..."
                        lines = truncated.split('\n')
                        for line in lines:
                            print(f"   {line}")
                    else:
                        lines = page_content.split('\n')
                        for line in lines:
                            print(f"   {line}")

                    print(f"   {'-' * 50}")
                    print()

                print(f"{'='*100}\n")
                return results

    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return []
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        return []
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return []

def compare_distance_strategies(query_text, limit=3, namespace_filter=None):
    """Compare results across different distance strategies"""
    print(f"🔬 COMPARING DISTANCE STRATEGIES FOR: '{query_text}'")
    print("=" * 120)

    strategies = [DistanceStrategy.COSINE, DistanceStrategy.EUCLIDEAN, DistanceStrategy.INNER_PRODUCT]
    all_results = {}

    for strategy in strategies:
        print(f"\n📏 Testing {strategy.value.upper()} distance...")
        results = test_vector_retrieval_ollama(query_text, limit, namespace_filter, strategy)
        all_results[strategy] = results

        if results:
            scores = [r[3] for r in results]  # similarity scores
            print(f"   Score range: {min(scores):.4f} to {max(scores):.4f}")

    return all_results

def get_database_stats():
    """Get statistics about the database contents"""
    try:
        with psycopg2.connect(DB_CONNECTION_STRING) as conn:
            with conn.cursor() as cur:
                # Count total documents
                cur.execute(f"SELECT COUNT(*) FROM {DOCS_TABLE_NAME}")
                total_docs = cur.fetchone()[0]

                # Count documents with embeddings
                cur.execute(f"SELECT COUNT(*) FROM {DOCS_TABLE_NAME} WHERE embedding IS NOT NULL")
                docs_with_embeddings = cur.fetchone()[0]

                # Get unique namespaces
                try:
                    cur.execute(f"SELECT DISTINCT namespace FROM {RECORDS_TABLE_NAME}")
                    namespaces = [row[0] for row in cur.fetchall()]
                except:
                    namespaces = ["Records table not found"]

                print(f"📊 Database Statistics:")
                print(f"   Total documents: {total_docs}")
                print(f"   Documents with embeddings: {docs_with_embeddings}")
                print(f"   Available namespaces: {namespaces}")
                print()

                return {
                    'total_docs': total_docs,
                    'docs_with_embeddings': docs_with_embeddings,
                    'namespaces': namespaces
                }

    except Exception as e:
        print(f"❌ Error getting database stats: {e}")
        return None

def test_embedding_only(query_text):
    """Just test the embedding generation without database query"""
    try:
        embedding = get_ollama_embedding(query_text, OLLAMA_MODEL, OLLAMA_HOST)
        print(f"✅ Successfully got embedding for: '{query_text}'")
        print(f"   Dimensions: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        return embedding
    except Exception as e:
        print(f"❌ Error getting embedding: {e}")
        return None

# Test the functions
if __name__ == "__main__":
    # Get database statistics first
    print("🧪 Getting database statistics...")
    stats = get_database_stats()

    # Test embedding generation
    print("🧪 Testing embedding generation...")
    embedding = test_embedding_only(TEST_SENTENCE)

    if embedding and stats and stats['docs_with_embeddings'] > 0:
        print("\n🧪 Testing individual distance strategies...")

        # Test each distance strategy
        query = TEST_SENTENCE

        print("\n1️⃣ Testing COSINE distance (recommended for most embeddings):")
        results_cosine = test_vector_retrieval_ollama(query, limit=3, distance_strategy=DistanceStrategy.COSINE)

        print("\n2️⃣ Testing EUCLIDEAN distance:")
        results_euclidean = test_vector_retrieval_ollama(query, limit=3, distance_strategy=DistanceStrategy.EUCLIDEAN)

        print("\n3️⃣ Testing INNER PRODUCT distance:")
        results_inner = test_vector_retrieval_ollama(query, limit=3, distance_strategy=DistanceStrategy.INNER_PRODUCT)

        print("\n🔬 Comparing all strategies side by side:")
        comparison_results = compare_distance_strategies(query, limit=2)

    else:
        print("❌ Cannot perform similarity search - no embeddings found or embedding generation failed")

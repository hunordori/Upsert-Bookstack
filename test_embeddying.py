import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)

def test_vector_retrieval_pretty(query_text, limit=5):
    query_embedding = model.encode([query_text])[0]

    conn = psycopg2.connect(
        host="jerry-docker1.abelcine.com",
        database="flowise-db",
        user="flowisedbuser",
        password="O7Q6ox3xFEQPDX4ZEuqvHzXKLaf29iTr"
    )

    cur = conn.cursor()

    embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

    cur.execute("""
        SELECT id, content, 1 - (embedding <=> %s::vector) AS similarity
        FROM abelcinewiki
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (embedding_str, embedding_str, limit))

    results = cur.fetchall()

    print(f"\n{'='*100}")
    print(f"SEARCH RESULTS FOR: '{query_text}'")
    print(f"{'='*100}\n")

    for i, (doc_id, content, similarity) in enumerate(results, 1):
        print(f"📄 RESULT #{i}")
        print(f"   ID: {doc_id}")
        print(f"   Similarity Score: {similarity:.4f}")
        print(f"   Content:")
        print(f"   {'-' * 50}")

        # Add line breaks for very long content
        if len(content) > 500:
            lines = content.split('\n')
            for line in lines:
                print(f"   {line}")
        else:
            print(f"   {content}")

        print(f"   {'-' * 50}")
        print()

    print(f"{'='*100}\n")

    conn.close()
    return results

# Use the prettier version
results = test_vector_retrieval_pretty("Engineering rack power usage")

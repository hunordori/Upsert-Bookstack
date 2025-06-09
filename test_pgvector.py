try:
    import pgvector
    print("✓ pgvector module imported successfully")
    print(f"pgvector version: {pgvector.__version__ if hasattr(pgvector, '__version__') else 'unknown'}")

    try:
        from pgvector.psycopg2 import register_vector
        print("✓ register_vector imported successfully")
    except ImportError as e:
        print(f"✗ Could not import register_vector: {e}")

    # List all available functions
    import pgvector.psycopg2
    print(f"Available functions in pgvector.psycopg2: {dir(pgvector.psycopg2)}")

except ImportError as e:
    print(f"✗ Could not import pgvector: {e}")

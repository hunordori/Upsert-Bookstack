# Goal

Embed and vectorize knowledge stored in a company wide Wiki (Bookstack) utilizing Postgres. Further down integrating with Flowise AI.

# Pre requisites

- Bookstack with a permission to generate API keys
- Postgres database credentials with pgvector plugin installed and configured
- Ollama instance

# Usage

1. Make sure you have the necessary prerequisites installed and configured.
2. Clone the repository and install the required dependencies.
3. Configure the environment variables in the `.env` file. (Use `.env.example` as a template)
4. Run the application using the command `upsert_bookstack.py`.

# Notes

- you can enable or disable chuncking in the `.env` file.
- if you change the embedding model adjust the dimension the `.env` file.
- our Wiki is set up by the following rules:
  - Shelves - department and main topics (IT, HR, General)
  - Books - big sections (IT/General, IT/How to?, IT/External Services)
  - Chapters - smaller sections
  - Pages - individual information (for example IT/External Services/Amazon Business)
- it is vibe coded, so could contain errors or inaccuracies. But the goal is to create an AI Chatbot, so people can utilize the information stored in the Wiki.

# Resources

- Bookstack API documentation: https://bookstack.bassopaolo.com/api/docs
- Postgres documentation: https://www.postgresql.org/docs/
- Flowise AI documentation: https://flowise.ai/docs/

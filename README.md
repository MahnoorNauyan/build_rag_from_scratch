# Building RAG from scratch
The goal is to build Retrival Augmented Generation (RAG) pipeline from scratch without using LLMs and Llama inorder to grasp the inner workings and essential mechanisms used in building RAG pipeline.

## Directories
- **documents**: directory that contains all documents used as a input data in pipeline.
- **database** contains document store that has chunks mapped with document ID and chunk ID and vector store that has all text embeddings mapped with chunk ID and document ID.
- **IMPORTANT POINT**: OpenAI api key is required to view LLM response. If you have api key then uncomment last 2 lines in main.py
## Usage

- Run the main.py script using the following command. 
   ```
   python3 main.py
   ```     
# Building RAG from scratch
The goal is to build Retrival Augmented Generation (RAG) pipeline from scratch without using LLMs and Llama inorder to grasp the inner workings and essential mechanisms used in building RAG pipeline.

## Directories
- **documents**: directory that contains all documents used as a input data in pipeline.
- **database** contains document store that has chunks mapped with document ID and chunk ID and vector store that has all text embeddings mapped with chunk ID and document ID.

- **Point to note**: You need OpenAI api key to visualize response from LLM. If you have OpenAI api key then uncomment line 218 and 219 to visualize response.

## Usage

- Run the main.py script using the following command. 
   ```
   python3 main.py
   ```     
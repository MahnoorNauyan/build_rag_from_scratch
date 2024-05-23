import os
import re
from transformers import AutoTokenizer, AutoModel
import uuid
import torch
import numpy as np
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os

#YOUR KEY HERE

def chunking(directory_path, tokenizer, chunk_size, para_seperator=" /n /n", separator=" "):

    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    documents = {}
    all_chunks = {}
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        print(filename)
        base = os.path.basename(file_path)
        sku = os.path.splitext(base)[0]
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            doc_id = str(uuid.uuid4())    

            paragraphs = re.split(para_seperator, text)

            for paragraph in paragraphs:
                words = paragraph.split(separator)
                current_chunk_str = ""
                chunk = []
                for word in words:
                    if current_chunk_str:
                        new_chunk = current_chunk_str + separator + word
                    else:
                        new_chunk = current_chunk_str + word    
                    if len(tokenizer.tokenize(new_chunk)) <= chunk_size:
                        current_chunk_str = new_chunk
                    else:
                        if current_chunk_str:
                            chunk.append(current_chunk_str)
                        current_chunk_str = word
                

                if current_chunk_str:   
                    chunk.append(current_chunk_str)

                for chunk in chunk:
                    chunk_id = str(uuid.uuid4())
                    all_chunks[chunk_id] = {"text": chunk, "metadata": {"file_name":sku}}
        documents[doc_id] = all_chunks
    return documents    


def map_document_embeddings(documents, tokenizer, model):
    mapped_document_db = {}
    for id, dict_content in documents.items():
        mapped_embeddings = {}
        for content_id, text_content in dict_content.items():
            text = text_content.get("text")
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
            mapped_embeddings[content_id] = embeddings
        mapped_document_db[id] = mapped_embeddings
    return mapped_document_db

def retrieve_information(query, top_k, mapped_document_db):
    query_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    query_embeddings = model(**query_inputs).last_hidden_state.mean(dim=1).squeeze()
    query_embeddings=query_embeddings.tolist()
    #converting query embeddings to numpy array
    query_embeddings=np.array(query_embeddings)

    scores = {}
    #Now calculating cosine similarity
    for doc_id, chunk_dict in mapped_document_db.items():
        for chunk_id, chunk_embeddings in chunk_dict.items():
            #converting chunk embedding to numpy array for efficent mathmetical operations
            chunk_embeddings = np.array(chunk_embeddings) 

            #Normalizing chunk embeddings and query embeddings  to get cosine similarity score
            normalized_query = np.linalg.norm(query_embeddings)
            normalized_chunk = np.linalg.norm(chunk_embeddings)

            if normalized_chunk == 0 or normalized_query == 0:
            # this is being done to avoid division with zero which will give wrong results i.e infinity. Hence to avoid this we set score to 0
                score == 0
            else:
            # Now calculationg cosine similarity score
                score = np.dot(chunk_embeddings, query_embeddings)/ (normalized_chunk * normalized_query)  

             #STORING SCORES WITH THE REFERENCE
            scores[(doc_id, chunk_id )] = score   

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    top_results=[]
    for ((doc_id, chunk_id), score) in sorted_scores:
        results = (doc_id, chunk_id, score)
        top_results.append(results)
    return top_results    
   
def compute_embeddings(query, tokenizer, model):
    query_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    query_embeddings = model(**query_inputs).last_hidden_state.mean(dim=1).squeeze()
    query_embeddings=query_embeddings.tolist()
    return query_embeddings

def calculate_cosine_similarity_score(query_embeddings, chunk_embeddings):
        normalized_query = np.linalg.norm(query_embeddings)
        normalized_chunk = np.linalg.norm(chunk_embeddings)
        if normalized_chunk == 0 or normalized_query == 0:
            score == 0
        else:
            score = np.dot(chunk_embeddings, query_embeddings)/ (normalized_chunk * normalized_query)  
        return score    


def retrieve_top_k_scores(query_embeddings, mapped_document_db, top_k):
    scores = {}

    for doc_id, chunk_dict in mapped_document_db.items():
        for chunk_id, chunk_embeddings in chunk_dict.items():
        #converting chunk embedding to numpy array for efficent mathmetical operations
            chunk_embeddings = np.array(chunk_embeddings) 
            score = calculate_cosine_similarity_score(query_embeddings, chunk_embeddings)
            scores[(doc_id, chunk_id )] = score 
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    return sorted_scores        

def retrieve_top_results(sorted_scores):
    top_results=[]
    for ((doc_id, chunk_id), score) in sorted_scores:
        results = (doc_id, chunk_id, score)
        top_results.append(results)
    return top_results    

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def retrieve_text(top_results, document_data):
    first_match = top_results[0]
    doc_id = first_match[0]
    chunk_id = first_match[1]
    related_text = document_data[doc_id][chunk_id]
    return related_text

def generate_llm_response(openai_model, query, relavent_text):
    template = """
    You are an intelligent search engine. You will be provided with some retrieved context, as well as the users query.

    Your job is to understand the request, and answer based on the retrieved context.
    Here is context:

    <context>
    {context}
    </context>

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template=template)

    chain = prompt | openai_model
    response=chain.invoke({"context":relavent_text["text"],"question":query})
    return response



if __name__ == "__main__":
    directory_path = "documents"
    model_name = "BAAI/bge-small-en-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    chunk_size = 200
    para_seperator=" /n /n"
    separator=" "
    top_k = 2
    openai_model = ChatOpenAI(model="gpt-3.5-turbo")

    
    #creating document store with chunk id, doc_id, text
    documents = chunking(directory_path, tokenizer, chunk_size, para_seperator, separator)

    #now embedding generation and mapping in database
    mapped_document_db = map_document_embeddings(documents, tokenizer, model)
    
    #saving json
    save_json('database/doc_store_2.json', documents) 
    save_json('database/vector_store_2.json', mapped_document_db)    
    
    #Retrieving most relavent data chunks
    query = "why toddlers throw tantrums?"
    query_embeddings = compute_embeddings(query, tokenizer, model)
    sorted_scores = retrieve_top_k_scores(query_embeddings, mapped_document_db, top_k)
    top_results = retrieve_top_results(sorted_scores)
    
    #reading json
    document_data = read_json("database/doc_store_2.json") #read document store

    #Retrieving text of relavent chunk embeddings
    relavent_text = retrieve_text(top_results, document_data)

    print(relavent_text)
    #print(relavent_text["text"])

   #Uncomment if you have api key 
    # response = generate_llm_response(openai_model, query, relavent_text)
    # print(response)








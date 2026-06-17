import fitz
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq

#Load API key from .env file
load_dotenv()

print("Current working dir:", os.getcwd())
print("API key found:", os.getenv("GROQ_API_KEY"))

client= Groq (api_key=os.getenv ("GROQ_API_KEY"))
# Load the embedding model once when this file is first imported
# it downloads ~90mb on first run, then uses the cached version

print ("Loading embedding model (first run downloads ~90mb)....")
EMBED_MODEL= SentenceTransformer ("all-MiniLM-L6-V2")

def ingest_pdf (pdf_path):
    """"
    read every page of the pdf, split into overlapping chunks of 500 chars.
    Returns a list of dicts: [{" text": "...", "page":3}, ...]
    """

    doc= fitz.open(pdf_path)
    all_chunks= []

    #tool to split long text into smaller pieces
    splitter= RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=60,
        separators= ["\n\n", "\n", ".", " "]
    )

    for page_num in range(len(doc)):
        page = doc[page_num]
        text= page.get_text()  #extract all text from this page
    
        if len (text.strip()) < 30:
            continue
    
        #split this page's text into chunks
        chunks= splitter.split_text(text)
    
        for chunk in chunks:
            chunk= chunk.strip()
            if len(chunk) >50:
                all_chunks.append({
                    "text": chunk,
                    "page": page_num +1
                })

    doc.close()
    print (f"Created {len (all_chunks)} chunks from {pdf_path}")
    return all_chunks


def build_faiss_index(chunks):
    """
    convert all text chunks to vectors and store in a FAISS index
    Returns (index, chunks)- we need to retireve answer later
    """

    #Get just the text strings from our list of dicts
    texts= [c["text"] for c in chunks]

    #Convert each text to 384-number vector
    #This captures the meaning of each sentence, not just keywords

    embeddings= EMBED_MODEL.encode(texts, show_progress_bar= True)
    embeddings= np.array(embeddings, dtype= "float32")  #FAISS needs float32

    dimension= embeddings.shape[1]
    index= faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print (f"FAISS index built with  {index.ntotal} vectors")
    return index, chunks


def answer_question (question, index, chunks):
    """
    1. Convert the question to a vector
    2. Find the 5 most similar chunks in FAISS
    3. Send those chunks + question to the LLM
    4. Return the answer and the source page numbers
    """

    q_vec= EMBED_MODEL.encode( [question])
    q_vec= np.array(q_vec, dtype= "float32")

    distances, indices= index.search(q_vec, k=5) #k is top_k, means find 5 most similar chunks

    context_parts= []
    sources= []
    for idx in indices [0]:
        if idx != -1:  #-1 means no result found
            chunk= chunks[idx]
            context_parts.append(f"[Page {chunk ['page']}]: {chunk ['text']}")
            sources.append(chunk ["page"])

    context= "\n\n---\n\n".join(context_parts)

    system_msg= (
        "You are a helpful clinical document assistant. "
        "Answer ONLY using the context provided below. "
        "If the context does not contain enough information, "
        "say: 'This document does not contain enough information to answer that.' "
        "Do not invent or guess any medical information."
    )

    user_msg= f"""Content from the document: {context}

Question: {question}

Answer based only on the context above. At the end, mention which pages the information came from"""

    response= client.chat.completions.create (
        model= "llama-3.3-70b-versatile",
        messages= [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature= 0.2,
        max_tokens=600
    )

    answer= response.choices[0].message.content
    return {
        "answer": answer,
        "sources": list(set(sources))
    }
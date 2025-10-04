
import streamlit as st
import os
import sys
import pickle
import tempfile

# Add better error handling for sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st.error("sentence-transformers not available. Please install with: pip install sentence-transformers")

# Handle LangChain imports with fallbacks
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain.vectorstores import FAISS
        from langchain.embeddings import HuggingFaceEmbeddings
    except ImportError:
        st.error("Required LangChain packages not installed.")
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
def create_faiss_index(texts):
    """Create FAISS vector store from texts"""
    try:
        # Use a smaller, more reliable model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller, more reliable
            model_kwargs={'device': 'cpu'},  # Force CPU usage
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vector_store = FAISS.from_texts(texts, embeddings)
        return vector_store
        
    except Exception as e:
        st.error(f"Error creating FAISS index: {e}")
        st.info("Trying alternative approach...")
        return create_faiss_index_fallback(texts)

def create_faiss_index_fallback(texts):
    """Fallback method using direct sentence-transformers"""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        import faiss
        
        # Load model directly
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embeddings
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        # Create FAISS index manually
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Using Inner Product (cosine similarity)
        
        # Add embeddings to index
        index.add(embeddings.astype('float32'))
        
        return {
            'index': index,
            'texts': texts,
            'model': model
        }
        
    except Exception as e:
        st.error(f"Fallback also failed: {e}")
        return None

def retrive_relevant_docs(vector_store, query, k=3):
    """Retrieve relevant documents from FAISS index"""
    try:
        if isinstance(vector_store, dict):  # Fallback vectorstore
            return retrive_from_fallback(vector_store, query, k)
        else:  # Standard LangChain vectorstore
            docs = vector_store.similarity_search(query, k=k)
            return docs
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return []

def retrive_from_fallback(vector_store, query, k=3):
    """Retrieve from fallback FAISS index"""
    try:
        query_embedding = vector_store['model'].encode([query], normalize_embeddings=True)
        query_embedding = query_embedding.astype('float32')
        
        # Search
        scores, indices = vector_store['index'].search(query_embedding, k)
        
        # Return relevant texts
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(vector_store['texts']):
                results.append(vector_store['texts'][idx])
        
        return results
    except Exception as e:
        st.error(f"Error in fallback retrieval: {e}")
        return []

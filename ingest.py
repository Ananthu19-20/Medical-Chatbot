from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstores1/db_faiss'

# Create vector database
def create_vector_db():
    # List all CSV files in the data directory
    csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {DATA_PATH}")

    documents = []
    for csv_file in csv_files:
        file_path = os.path.join(DATA_PATH, csv_file)
        try:
            loader = CSVLoader(file_path=file_path, encoding='utf-8')
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")

    if not documents:
        raise ValueError("No documents were successfully loaded")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"Vector database created and saved to {DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vector_db()
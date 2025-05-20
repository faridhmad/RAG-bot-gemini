import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from a .env file
load_dotenv()

# 1. API Key for Gemini (Google Generative AI)
# The API key is now securely loaded from the .env file
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# 2. Load documents
document_folder = "/your-path-to-folder-documents" 
documents = []
for filename in os.listdir(document_folder):
    if filename.endswith(".txt"):  # Only process .txt files
        file_path = os.path.join(document_folder, filename)
        loader = TextLoader(file_path)
        documents.extend(loader.load())

# 3. Split text
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 4. Embeddings from HuggingFace
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 5. Save in VectorStore FAISS
db = FAISS.from_documents(docs, embedding)
retriever = db.as_retriever()

# 6. Use Gemini (ChatGoogleGenerativeAI)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# 7. Retrieval QA Chain
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# 8. Accept user questions
def run_qa_system():
    print("\n=== Welcome to ABC Customer Service System ===")
    print("We are here to assist you with any inquiries!")
    print("Type 'exit' to end the session.\n")
    
    while True:
        query = input("\nHow can we assist you today? ")
        
        if query.lower() in ['exit', 'quit']:
            print("\nThank you for contacting ABC Customer Service!")
            print("We hope to assist you again in the future. Goodbye!")
            break
            
        if query.strip():
            print("\nPlease hold on, we are searching for an answer for you...")
            result = qa.invoke(query)
            print("\n🤖 Our answer:", result["result"])
            print("\nIs there anything else we can assist you with?")
        else:
            print("\nSorry, we couldn't understand your question. Please try again with a clearer question.")

# Run the system
if __name__ == "__main__":
    run_qa_system()
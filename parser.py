from langchain_community.document_loaders import PyPDFLoader

def load_resumes(data_path="data"):
    documents = []
    
    import os
    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_path, file))
            documents.extend(loader.load())
    
    return documents
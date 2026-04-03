from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_resumes(data_path=None):
    base_dir = Path(__file__).resolve().parent

    if data_path is None:
        data_path = base_dir / "Data"
        if not data_path.exists():
            data_path = base_dir / "data"
    else:
        data_path = Path(data_path)
        if not data_path.is_absolute():
            data_path = base_dir / data_path

    if not data_path.exists() or not data_path.is_dir():
        raise FileNotFoundError(f"Resume directory not found: {data_path}")

    documents = []
    for file in sorted(data_path.iterdir()):
        if file.is_file() and file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(file)
            documents.extend(loader.load())

    if not documents:
        raise ValueError(f"No PDF resumes found in {data_path}")

    return documents
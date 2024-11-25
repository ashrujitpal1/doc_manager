from typing import List, Dict, Any
from pathlib import Path
import logging
import nltk
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    PyPDFLoader,
    PDFMinerLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from ..config.settings import settings

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
except Exception as e:
    logging.warning(f"Error downloading NLTK data: {e}")

class DocumentService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        self.vector_store = Chroma(
            persist_directory=settings.CHROMA_DB_DIR,
            embedding_function=self.embeddings,
            collection_name=settings.COLLECTION_NAME
        )
        
        # Initialize dependencies
        self._initialize_dependencies()
    
    def _initialize_dependencies(self):
        """Initialize and check all required dependencies"""
        try:
            # Check OpenCV
            import cv2
            logging.info("OpenCV initialized successfully")
            
            # Check PDF processing
            import pdf2image
            logging.info("PDF2Image initialized successfully")
            
            # Check Tesseract
            import pytesseract
            logging.info("Pytesseract initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing dependencies: {e}")
            raise
    
    def get_loader_for_file(self, file_path: Path):
        """Get appropriate loader based on file type"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            try:
                # Try PyPDFLoader first
                logging.info(f"Attempting to load PDF with PyPDFLoader: {file_path}")
                return PyPDFLoader(str(file_path))
            except Exception as e:
                logging.warning(f"PyPDFLoader failed, trying PDFMinerLoader: {e}")
                try:
                    # Try PDFMinerLoader as backup
                    return PDFMinerLoader(str(file_path))
                except Exception as e:
                    logging.warning(f"PDFMinerLoader failed, trying UnstructuredFileLoader: {e}")
                    # Use UnstructuredFileLoader as last resort
                    return UnstructuredFileLoader(str(file_path))
        else:
            logging.info(f"Using UnstructuredFileLoader for: {file_path}")
            return UnstructuredFileLoader(str(file_path))

    def process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single file and return chunks"""
        try:
            logging.info(f"Processing file: {file_path}")
            
            # Validate file exists
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Validate file size
            file_size = file_path.stat().st_size
            if file_size == 0:
                raise ValueError(f"File is empty: {file_path}")
            
            # Get appropriate loader
            loader = self.get_loader_for_file(file_path)
            
            # Load document
            document = loader.load()
            
            if not document:
                raise ValueError(f"No content extracted from: {file_path}")
            
            logging.info(f"Successfully loaded document: {file_path}")
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(document)
            
            if not chunks:
                raise ValueError(f"No chunks created from: {file_path}")
            
            logging.info(f"Split document into {len(chunks)} chunks")
            
            # Add metadata
            for chunk in chunks:
                chunk.metadata.update({
                    "source": file_path.name,
                    "type": file_path.suffix,
                    "size": file_size,
                    "path": str(file_path),
                    "chunk_size": len(chunk.page_content)
                })
            
            return chunks
            
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
            raise

    def upload_documents(self, directory: str) -> Dict[str, Any]:
        """Upload all documents from directory to ChromaDB"""
        try:
            # Validate directory
            dir_path = Path(directory)
            if not dir_path.exists():
                return {"status": "error", "message": f"Directory not found: {directory}"}
            
            # Get all files
            files = list(dir_path.glob("**/*.*"))
            
            if not files:
                return {"status": "error", "message": "No files found"}
            
            logging.info(f"Found {len(files)} files to process")
            
            # Process all files
            all_chunks = []
            errors = []
            processed_files = 0
            
            for file_path in files:
                try:
                    chunks = self.process_file(file_path)
                    all_chunks.extend(chunks)
                    processed_files += 1
                    logging.info(f"Successfully processed {file_path}")
                except Exception as e:
                    error_msg = f"Error processing {file_path}: {str(e)}"
                    logging.error(error_msg)
                    errors.append(error_msg)
            
            if all_chunks:
                # Add to vector store
                self.vector_store.add_documents(all_chunks)
                
                return {
                    "status": "success",
                    "message": f"Processed {processed_files} files, {len(all_chunks)} chunks",
                    "total_files": len(files),
                    "successful_files": processed_files,
                    "failed_files": len(files) - processed_files,
                    "total_chunks": len(all_chunks),
                    "errors": errors if errors else None
                }
            else:
                return {
                    "status": "error",
                    "message": "No documents were successfully processed",
                    "errors": errors
                }
            
        except Exception as e:
            logging.error(f"Error uploading documents: {str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def search_documents(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search documents based on query"""
        try:
            logging.info(f"Searching for: {query}")
            
            # Validate query
            if not query.strip():
                return {"status": "error", "message": "Empty query"}
            
            # Search vector store
            results = self.vector_store.similarity_search_with_score(
                query,
                k=limit
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            
            logging.info(f"Found {len(formatted_results)} results")
            
            return {
                "status": "success",
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results)
            }
            
        except Exception as e:
            logging.error(f"Error searching documents: {str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}

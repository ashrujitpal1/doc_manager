from flask import Blueprint, request, jsonify
from doc_manager.services.document_service import DocumentService
from doc_manager.config.settings import settings

bp = Blueprint('api', __name__)
doc_service = DocumentService()

@bp.route('/upload', methods=['POST'])
def upload_documents():
    """Upload documents from the configured directory"""
    try:
        result = doc_service.upload_documents(settings.DOCUMENT_DIR)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@bp.route('/search', methods=['POST'])
def search_documents():
    """Search documents based on query"""
    try:
        data = request.get_json()
        query = data.get('query')
        limit = data.get('limit', 5)
        
        if not query:
            return jsonify({"status": "error", "message": "Query is required"}), 400
        
        result = doc_service.search_documents(query, limit)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

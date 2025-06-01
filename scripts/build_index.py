import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch


class VectorIndexer:
    def __init__(self):
        self.model = self._load_embedding_model()
        self.index = None
        self.metadata = []
    
    def _load_embedding_model(self):
        """Charge un seul modèle d'embedding"""
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        try:
            print(f"Chargement du modèle: {model_name}")
            model = SentenceTransformer(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Modèle {model_name} chargé avec succès")
            return model
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle {model_name}: {str(e)}")

    def process_directory(self, input_dir, output_dir):
        """Vectorise tous les fichiers JSON traités"""
        os.makedirs(output_dir, exist_ok=True)
        
        all_embeddings = []
        for filename in tqdm(sorted(os.listdir(input_dir))):
            if filename.endswith('_cleaned.json'):
                filepath = os.path.join(input_dir, filename)
                chunks, chunk_metadata = self._extract_chunks(filepath)
                
                # Embedding par batch pour optimiser
                batch_size = 32
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    embeddings = self.model.encode(batch, show_progress_bar=False)
                    all_embeddings.append(embeddings)
                    self.metadata.extend(chunk_metadata[i:i + batch_size])
        
        self._build_index(np.vstack(all_embeddings))
        self._save_assets(output_dir)
    
    def _extract_chunks(self, filepath):
        """Extrait le contenu à vectoriser"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = []
        metadata = []
        doc_name = os.path.splitext(os.path.basename(filepath))[0]
        
        for section in data['sections']:
            # Section principale
            if section['content'].strip():
                chunks.append(f"TITRE: {section['title']}\n{section['content']}")
                metadata.append({
                    "source": f"{doc_name} - {section['metadata']['section_id']}",
                    "type": "section"
                })
            
            # Sous-sections
            for sub in section.get('subsections', []):
                if sub['content'].strip():
                    chunks.append(f"SOUS-TITRE: {sub['title']}\n{sub['content']}")
                    metadata.append({
                        "source": f"{doc_name} - {sub['metadata']['subsection_id']}",
                        "type": "subsection"
                    })
                
                # Sous-parties
                for part in sub.get('subparts', []):
                    if part['content'].strip():
                        chunks.append(f"PARTIE: {part['title']}\n{part['content']}")
                        metadata.append({
                            "source": f"{doc_name} - {part['metadata']['subpart_id']}",
                            "type": "subpart"
                        })
        
        return chunks, metadata
    
    def _build_index(self, embeddings):
        """Construit l'index FAISS"""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        print(f"Index créé avec {self.index.ntotal} embeddings de dimension {dimension}")
    
    def _save_assets(self, output_dir):
        """Sauvegarde l'index et les métadonnées"""
        faiss.write_index(self.index, os.path.join(output_dir, 'lesson_index.faiss'))
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Index et métadonnées sauvegardés dans {output_dir}")

if __name__ == "__main__":
    # Vérification des dépendances
    try:
        import faiss
        assert hasattr(faiss, 'IndexFlatIP'), "FAISS mal installé"
    except ImportError:
        raise ImportError("Installez FAISS: pip install faiss-cpu ou faiss-gpu")
    
    indexer = VectorIndexer()
    indexer.process_directory(
        input_dir="data/processed/",
        output_dir="data/vector_db/"
    )

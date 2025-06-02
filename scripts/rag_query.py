import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from ctransformers import AutoModelForCausalLM
from typing import List, Dict, Tuple

class RAGSystem:
    def __init__(self, vector_db_dir: str = "data/vector_db/"):
        # Configuration initiale
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Utilisation du dispositif: {self.device}")
        
        # Charger les composants
        self._load_retriever(vector_db_dir)
        self._load_zephyr()
    
    def _load_retriever(self, vector_db_dir: str):
        """Charge le modèle d'embedding et l'index FAISS"""
        try:
            self.embedder = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                device=self.device
            )
            
            self.index = faiss.read_index(f"{vector_db_dir}/lesson_index.faiss")
            
            with open(f"{vector_db_dir}/metadata.json", "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                
            print(f"Retriever chargé avec {self.index.ntotal} embeddings")
            
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du retriever: {str(e)}")
    
    def _load_zephyr(self):
        """Charge Zephyr-7B quantifié (4-bit)"""
        try:
            model_name = "TheBloke/zephyr-7B-alpha-GGUF"
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                model_file="zephyr-7b-alpha.Q4_K_M.gguf",
                model_type="mistral",
                gpu_layers=50 if self.device == "cuda" else 0  # Seule modification ici
            )
            print("Zephyr-7B-alpha (4-bit) chargé avec succès")
            
        except Exception as e:
            raise RuntimeError(f"Erreur de chargement de Zephyr: {str(e)}")
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Récupère les k chunks les plus pertinents"""
        query_embedding = self.embedder.encode(
            [query], 
            convert_to_tensor=True,
            show_progress_bar=False
        ).cpu().numpy()
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx >= 0:
                result = {
                    "content": self.metadata[idx].get("text", ""),
                    "score": float(score),
                    "source": self.metadata[idx].get("source", "inconnu"),
                    "type": self.metadata[idx].get("type", "section")
                }
                results.append(result)
        
        return sorted(results, key=lambda x: x["score"], reverse=True)
    
    def _format_prompt(self, query: str, context: str) -> str:
        """Structure le prompt pour Zephyr"""
        return f"""<|system|>
Vous êtes un assistant expert en histoire et géographie du Sénégal.
Répondez exclusivement en français avec des termes précis.
Basez-vous strictement sur le contexte fourni.

Contexte:
{context[:3000]}  # Tronquer pour éviter les dépassements

Question: {query}</s>
<|user|>
{query}</s>
<|assistant|>
"""
    
    def generate_answer(self, query: str, context: str) -> str:
        """Génère une réponse à partir du contexte"""
        prompt = self._format_prompt(query, context)
        
        output = self.llm(
            prompt,
            max_new_tokens=256,
            temperature=0.3,
            repetition_penalty=1.1,
            stop=["</s>", "<|endoftext|>"]
        )
        
        return str(output).split("</s>")[0].strip()
    
    def query(self, question: str, k_results: int = 3) -> Dict:
        """Pipeline RAG complet"""
        # Étape 1: Récupération
        retrieved = self.retrieve(question, k=k_results)
        context = "\n\n".join([r["content"] for r in retrieved])
        
        # Étape 2: Génération
        answer = self.generate_answer(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "sources": [r["source"] for r in retrieved],
            "context": retrieved
        }

if __name__ == "__main__":
    print("Initialisation du système RAG...")
    rag = RAGSystem()
    
    questions = [
        "Quelles sont les caractéristiques du système colonial français ?",
        "Expliquez les impacts de la révolution industrielle en Afrique",
        "Quels sont les facteurs de la décolonisation ?"
    ]
    
    for q in questions:
        print(f"\nQuestion: {q}")
        result = rag.query(q)
        print(f"Réponse: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print("---")

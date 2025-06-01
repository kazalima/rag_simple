import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

class RAGSystem:
    def __init__(self, model_cache_dir="models"):
        self.model_cache_dir = model_cache_dir
        os.makedirs(self.model_cache_dir, exist_ok=True)
        
        # Configurations
        self.embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.llm_model_name = "HuggingFaceH4/zephyr-7b-beta"
        
        # Chargement des modèles
        self._load_embedding_model()
        self._load_llm_model()
        self._load_vector_db()

    def _load_embedding_model(self):
        """Charge le modèle d'embedding"""
        self.embedder = SentenceTransformer(self.embedding_model_name)

    def _load_llm_model(self):
        """Charge le LLM avec sauvegarde en cache"""
        model_path = os.path.join(self.model_cache_dir, "zephyr-7b-beta")
        
        # Téléchargement si non présent en cache
        if not os.path.exists(model_path):
            print("Téléchargement du modèle Zephyr...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True
            )
            # Sauvegarde locale
            self.tokenizer.save_pretrained(model_path)
            self.llm.save_pretrained(model_path)
            print(f"Modèle sauvegardé dans {model_path}")
        else:
            print("Chargement du modèle depuis le cache...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True
            )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.llm,
            tokenizer=self.tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def _load_vector_db(self):
        """Charge la base vectorielle"""
        self.index = faiss.read_index("data/vector_db/lesson_index.faiss")
        with open("data/vector_db/metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def retrieve(self, query, k=3):
        """Recherche les passages pertinents"""
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        
        return [
            {
                "text": self.metadata[idx]["text"],
                "source": self.metadata[idx]["source"],
                "score": float(score)
            }
            for idx, score in zip(indices[0], distances[0]) if idx >= 0
        ]

    def generate_response(self, question, contexts, max_length=512):
        """Génère une réponse avec le contexte"""
        context_str = "\n".join(f"[Source: {ctx['source']}] {ctx['text']}" for ctx in contexts)
        
        prompt = f"""<|system|>
Vous êtes un expert en histoire-géographie. Répondez à la question en utilisant exclusivement le contexte fourni.

Contexte:
{context_str}</s>
<|user|>
{question}</s>
<|assistant|>"""
        
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        return outputs[0]["generated_text"].split("<|assistant|>")[-1].strip()

    def query(self, question):
        """Pipeline complet RAG"""
        contexts = self.retrieve(question)
        response = self.generate_response(question, contexts)
        
        return {
            "question": question,
            "answer": response,
            "sources": [ctx["source"] for ctx in contexts]
        }

def test_rag_system():
    """Teste le système RAG avec une question exemple"""
    print("Initialisation du système RAG...")
    rag = RAGSystem(model_cache_dir="models")
    
    question = "Quelles sont les causes principales de la révolution industrielle en Europe ?"
    print(f"\nQuestion test: {question}")
    
    result = rag.query(question)
    
    print("\nRéponse générée:")
    print(result["answer"])
    
    print("\nSources utilisées:")
    for source in result["sources"]:
        print(f"- {source}")

if __name__ == "__main__":
    # Exécute le test automatiquement
    test_rag_system()

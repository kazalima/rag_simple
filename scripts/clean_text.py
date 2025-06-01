import os
import re
import json
from datetime import datetime

def clean_text(text):
    """Nettoyage du texte en conservant la structure"""
    text = re.sub(r'[^\S\r\n]+', ' ', text)  # Espaces multiples sauf sauts de ligne
    text = re.sub(r'[\u00A0\u202F]', ' ', text)  # Remplace les espaces insécables
    text = re.sub(r'[^\w\s.,;:\-—–«»’\'"\n]', '', text)  # Garde la ponctuation utile
    return text.strip()

def segment_lesson(text, filename):
    """Segmente le texte en structure hiérarchique avec métadonnées"""
    sections = []
    current_section = None
    current_subsection = None
    lines = text.split('\n')
    
    # Patterns améliorés
    main_section_pattern = re.compile(
        r'^(LESSON|LECON|INTRODUCTION|CONCLUSION|CONCLUSION GÉNÉRALE|[IVX]+[-–].*|'
        r'[A-ZÉÈÊÎÂÔÛÇ][A-ZÉÈÊÎÂÔÛÇ\s\-–:]{2,})$', 
        re.IGNORECASE
    )
    subsection_pattern = re.compile(r'^([A-Z])[-–].*$')
    subpart_pattern = re.compile(r'^(\d+)[-–].*$')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if main_section_pattern.fullmatch(line):
            # Nouvelle section principale
            if current_section:
                sections.append(current_section)
            current_section = {
                "title": line,
                "content": "",
                "type": "section",
                "subsections": [],
                "metadata": {
                    "section_id": f"sec_{len(sections)+1}",
                    "keywords": extract_keywords(line)
                }
            }
            current_subsection = None
            
        elif subsection_pattern.match(line):
            # Nouvelle sous-section
            if not current_section:
                current_section = create_generic_section(len(sections))
                sections.append(current_section)
                
            current_subsection = {
                "title": line,
                "content": "",
                "type": "subsection",
                "subparts": [],
                "metadata": {
                    "subsection_id": f"sub_{len(current_section['subsections'])+1}",
                    "parent_section": current_section['metadata']['section_id'],
                    "keywords": extract_keywords(line)
                }
            }
            current_section['subsections'].append(current_subsection)
            
        elif subpart_pattern.match(line):
            # Nouvelle sous-partie (1-, 2-, etc.)
            if not current_subsection:
                current_subsection = create_generic_subsection(current_section)
                
            subpart = {
                "title": line,
                "content": "",
                "type": "subpart",
                "metadata": {
                    "subpart_id": f"part_{len(current_subsection['subparts'])+1}",
                    "parent_subsection": current_subsection['metadata']['subsection_id'],
                    "keywords": extract_keywords(line)
                }
            }
            current_subsection['subparts'].append(subpart)
            
        else:
            # Ajout du contenu au niveau approprié
            if current_subsection and current_subsection.get('subparts'):
                current_subsection['subparts'][-1]['content'] += line + " "
            elif current_subsection:
                current_subsection['content'] += line + " "
            elif current_section:
                current_section['content'] += line + " "

    if current_section:
        sections.append(current_section)

    # Post-traitement pour améliorer la structure RAG
    return optimize_for_rag(sections, filename)

def create_generic_section(index):
    """Crée une section générique quand manquante"""
    return {
        "title": "Sans titre",
        "content": "",
        "type": "section",
        "subsections": [],
        "metadata": {
            "section_id": f"sec_{index+1}",
            "keywords": []
        }
    }

def create_generic_subsection(section):
    """Crée une sous-section générique quand manquante"""
    subsection = {
        "title": "Sans sous-titre",
        "content": "",
        "type": "subsection",
        "subparts": [],
        "metadata": {
            "subsection_id": f"sub_{len(section['subsections'])+1}",
            "parent_section": section['metadata']['section_id'],
            "keywords": []
        }
    }
    section['subsections'].append(subsection)
    return subsection

def extract_keywords(text):
    """Extrait des mots-clés simples du titre"""
    # Enlève la numérotation (I-, A-, 1-, etc.)
    clean_text = re.sub(r'^[IVX0-9A-Z][-–.]\s*', '', text, flags=re.IGNORECASE)
    # Garde les mots avec majuscules (noms propres) + mots importants
    return list(set([
        word.lower() for word in re.findall(r'\b([A-ZÉÈÊÎÂÔÛÇ][a-zéèêîâôûç]+|\w{5,})\b', clean_text)
        if len(word) > 3 and word.lower() not in ['sans', 'titre', 'section']
    ]))

def optimize_for_rag(sections, filename):
    """Optimise la structure pour le RAG"""
    optimized = []
    
    # Métadonnées globales du document
    doc_metadata = {
        "document_id": os.path.splitext(filename)[0],
        "processing_date": datetime.now().isoformat(),
        "total_sections": len(sections)
    }
    
    for section in sections:
        # Section principale
        optimized_section = {
            **section,
            "content": clean_rag_content(section["content"]),
            "metadata": {
                **section["metadata"],
                "chunk_type": "section",
                "document": doc_metadata["document_id"]
            }
        }
        
        # Traitement des sous-sections
        optimized_subsections = []
        for sub in section.get("subsections", []):
            optimized_sub = {
                **sub,
                "content": clean_rag_content(sub["content"]),
                "metadata": {
                    **sub["metadata"],
                    "chunk_type": "subsection",
                    "document": doc_metadata["document_id"]
                }
            }
            
            # Traitement des sous-parties (1-, 2-, etc.)
            optimized_parts = []
            for part in sub.get("subparts", []):
                optimized_part = {
                    **part,
                    "content": clean_rag_content(part["content"]),
                    "metadata": {
                        **part["metadata"],
                        "chunk_type": "subpart",
                        "document": doc_metadata["document_id"]
                    }
                }
                optimized_parts.append(optimized_part)
            
            optimized_sub["subparts"] = optimized_parts
            optimized_subsections.append(optimized_sub)
        
        optimized_section["subsections"] = optimized_subsections
        optimized.append(optimized_section)
    
    return {
        "document": doc_metadata,
        "sections": optimized
    }

def clean_rag_content(text):
    """Nettoyage spécifique pour RAG"""
    if not text:
        return ""
    
    # Normalisation des espaces et ponctuation
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.,;:])(\w)', r'\1 \2', text)  # Espace après ponctuation
    return text.strip()

if __name__ == "__main__":
    raw_dir = "data/raw/"
    processed_dir = "data/processed/"
    os.makedirs(processed_dir, exist_ok=True)

    for txt_file in os.listdir(raw_dir):
        if txt_file.endswith(".txt"):
            file_path = os.path.join(raw_dir, txt_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            cleaned = clean_text(text)
            segmented = segment_lesson(cleaned, txt_file)

            output_path = os.path.join(processed_dir, txt_file.replace(".txt", "_cleaned.json"))
            with open(output_path, 'w', encoding='utf-8') as out_f:
                json.dump(segmented, out_f, ensure_ascii=False, indent=2)

            # Rapport de traitement
            print(f"\n✅ {txt_file} → Traitement terminé")
            print(f"   - Sections principales: {segmented['document']['total_sections']}")
            print(f"   - Dernière mise à jour: {segmented['document']['processing_date']}")
            
            # Aperçu de la structure
            for i, sec in enumerate(segmented["sections"][:3], 1):
                print(f"     {i}. {sec['title']} ({len(sec.get('subsections', []))} sous-sections)")
                if sec.get('subsections'):
                    for j, sub in enumerate(sec['subsections'][:2], 1):
                        print(f"       {i}.{j} {sub['title']} ({len(sub.get('subparts', []))} parties)")

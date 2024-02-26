import spacy
from transformers import pipeline

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Load transformers summarization pipeline
summarizer = pipeline("summarization")

def process_legal_document(document_path):
    # Read the legal document
    with open(document_path, 'r', encoding='utf-8') as file:
        legal_text = file.read()

    # Tokenize and process legal text with SpaCy
    doc = nlp(legal_text)

    # Extract entities using SpaCy's named entity recognition (NER)
    entities = [ent.text for ent in doc.ents]

    # Use transformers summarization pipeline to generate a summary
    summary = summarizer(legal_text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, temperature=0.5)[0]['summary_text']

    return entities, summary

if __name__ == "__main__":
    # Provide the path to your legal document
    # Provide the correct path to your legal document
    document_path = "/home/vishnu/Desktop/manifest-sha512.txt"


    # Process the legal document
    entities, summary = process_legal_document(document_path)

    # Display the extracted entities and summary
    print("Extracted Entities:", entities)
    print("\nSummary:", summary)


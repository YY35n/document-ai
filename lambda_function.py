import json
import boto3
import os
import openai
from pinecone import Pinecone
import pdfplumber
import docx

# Load environment variables
S3_BUCKET_NAME = os.environ['S3_BUCKET_NAME']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_INDEX_NAME = os.environ['PINECONE_INDEX_NAME']

# Initialize AWS S3
s3_client = boto3.client('s3')

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

def extract_text_from_doc(file_path):
    """Extract text from DOCX file"""
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_file(file_path, file_extension):
    """Determine file type and extract text"""
    if file_extension == ".txt":
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    elif file_extension == ".pdf":
        return extract_text_from_pdf(file_path)
    elif file_extension == ".docx":
        return extract_text_from_doc(file_path)
    return None

def chunk_text(text, max_tokens=8191):
    """Chunk text to fit OpenAI's embedding limit"""
    words = text.split()
    chunks = []
    while words:
        chunk = words[:max_tokens]
        chunks.append(" ".join(chunk))
        words = words[max_tokens:]
    return chunks

def get_embedding(text):
    """Generate embeddings using OpenAI"""
    chunks = chunk_text(text)
    embeddings = [
        openai.embeddings.create(input=chunk, model="text-embedding-ada-002").data[0].embedding
        for chunk in chunks
    ]
    return embeddings

def lambda_handler(event, context):
    """Triggered when a file is uploaded to S3"""
    for record in event['Records']:
        bucket_name = record['s3']['bucket']['name']
        file_key = record['s3']['object']['key']
        file_extension = os.path.splitext(file_key)[1].lower()

        # Download file from S3
        file_path = f"/tmp/{file_key}"
        s3_client.download_file(bucket_name, file_key, file_path)

        # Extract text
        text_content = extract_text_from_file(file_path, file_extension)

        if text_content:
            embeddings = get_embedding(text_content)
            
            # Store in Pinecone
            upsert_data = [(f"{file_key}-{i}", embedding) for i, embedding in enumerate(embeddings)]
            index.upsert(upsert_data)

            return {
                'statusCode': 200,
                'body': json.dumps(f"Stored embeddings for {file_key} in Pinecone.")
            }

    return {
        'statusCode': 500,
        'body': json.dumps("Failed to process file.")
    }

def search_pinecone(query, top_k=5):
    """Search Pinecone for related embeddings"""
    query_embedding = get_embedding(query)[0]
    result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    if 'matches' in result and result['matches']:
        return [match['id'] for match in result['matches']]
    return []

def generate_gpt_response(user_query):
    """Generate GPT response based on document context"""
    related_docs = search_pinecone(user_query)
    
    if not related_docs:
        return "No relevant documents found in the knowledge base."

    context = "\n".join(related_docs)
    prompt = f"Use the following documents to answer: {context}\n\nUser query: {user_query}"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

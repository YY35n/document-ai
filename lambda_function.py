import json
import boto3
import os
import openai
import io
import logging
import tiktoken
import pdfplumber
import docx
from pinecone import Pinecone

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Load environment variables with defaults
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'default-bucket')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', '')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', '')

# Initialize AWS S3 client
s3_client = boto3.client('s3')

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes():
    raise ValueError(f"Pinecone index {PINECONE_INDEX_NAME} does not exist.")
index = pc.Index(PINECONE_INDEX_NAME)

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Tokenizer for chunking text
enc = tiktoken.get_encoding("cl100k_base")

def extract_text_from_doc(file_stream):
    """Extract text from DOCX file"""
    doc = docx.Document(file_stream)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(file_stream):
    """Extract text from PDF file"""
    text = ""
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n" if page.extract_text() else ""
    return text

def extract_text_from_file(file_stream, file_extension):
    """Extract text from different file formats"""
    try:
        if file_extension == ".txt":
            return file_stream.read().decode("utf-8")
        elif file_extension == ".pdf":
            return extract_text_from_pdf(file_stream)
        elif file_extension == ".docx":
            return extract_text_from_doc(file_stream)
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return None

def chunk_text(text, max_tokens=8191):
    """Chunk text to fit OpenAI's embedding limit"""
    tokens = enc.encode(text)
    return [enc.decode(tokens[i:i+max_tokens]) for i in range(0, len(tokens), max_tokens)]

def get_embedding(text):
    """Generate embeddings using OpenAI"""
    try:
        chunks = chunk_text(text)
        embeddings = [
            openai.embeddings.create(input=chunk, model="text-embedding-ada-002").data[0].embedding
            for chunk in chunks
        ]
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return []

def lambda_handler(event, context):
    """Triggered when a file is uploaded to S3"""
    try:
        for record in event['Records']:
            bucket_name = record['s3']['bucket']['name']
            file_key = record['s3']['object']['key']
            file_extension = os.path.splitext(file_key)[1].lower()

            # Get file from S3
            file_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            file_stream = io.BytesIO(file_obj['Body'].read())

            # Extract text
            text_content = extract_text_from_file(file_stream, file_extension)
            if not text_content:
                raise ValueError("Extracted text is empty")

            # Get embeddings
            embeddings = get_embedding(text_content)
            if not embeddings:
                raise ValueError("Failed to generate embeddings")

            # Store in Pinecone
            upsert_data = [(f"{file_key}-{i}", embedding) for i, embedding in enumerate(embeddings)]
            index.upsert(upsert_data)

            return {
                'statusCode': 200,
                'body': json.dumps(f"Stored embeddings for {file_key} in Pinecone.")
            }
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Failed to process file: {str(e)}")
        }

def search_pinecone(query, top_k=5):
    """Search Pinecone for related embeddings"""
    try:
        query_embedding = get_embedding(query)[0]
        result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

        if 'matches' in result and result['matches']:
            return [match['id'] for match in result['matches']]
        return []
    except Exception as e:
        logger.error(f"Error searching Pinecone: {e}")
        return []

def generate_gpt_response(user_query):
    """Generate GPT response based on document context"""
    try:
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
    except Exception as e:
        logger.error(f"Error generating GPT response: {e}")
        return "Error generating response."
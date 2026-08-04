import json
import os
import sys
from typing import List

import pika
from pymilvus import MilvusClient

from embedding import EMBED_DIM, embed_texts
from extraction import parse


MILVUS_ADDRESS = "milvus.db"
COLLECTION_NAME = "document_chunks"

def create_collection_if_not_exists(client: MilvusClient):
    if client.has_collection(COLLECTION_NAME):
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=EMBED_DIM,
        metric_type="COSINE",
        vector_field_name="vector",
        auto_id=True,
    )
    client.create_index(
        collection_name=COLLECTION_NAME,
        index_params={"index_type": "FLAT", "metric_type": "COSINE", "params": {}},
    )
def create_callback(client: MilvusClient):
    def callback(ch, method, properties, body):
        """
        This function is called whenever a message is received from RabbitMQ

            Assumes body of message is a JSON string of form: 
                {"doc_id": <doc_id>,
                "user_id": <user_id>,
                "file_path": <file_path>}
            where the ids are provided as stored in the database where 
            the file is stored
        """
        
        try:
            message = json.loads(body.decode('utf-8'))

            doc_id = message["doc_id"]
            user_id = message["user_id"]
            file_path = message["file_path"]

            print(f" [x] Received message for document ID: {doc_id}")

            if not file_path:# If filepath is not provided rejects message w/o requeue
                print(f" [!] Error: Message for {doc_id} is missing 'file_path'. Rejecting")
                ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
                return

            success = process_document(file_path, doc_id, user_id)

        
            if success:
                ch.basic_ack(delivery_tag=method.delivery_tag)
                print(f" [x] Successfully processed document: {doc_id}")
            else:
                ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
                print(f" [!!!] Failed to process document: {doc_id}.")
            
        except Exception as e:
            print(f" [!!!] Error for {doc_id}: {e}")
            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
            return
    return callback
    
def process_document(file_path, doc_id, user_id):
    """
    Processes the document and stores the embeddings in a vector DB
    """

    if not os.path.exists(file_path):
        print(f" [!] File not found: {file_path}")
        return False

    try:
        chunks: List[str] = parse(file_path)
        if not chunks:
            print(f" [!] No text chunks extracted from document: {file_path}")
            return False
        vectors = embed_texts(chunks)

        payload = []
        for ind, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload.append(
                {
                    "vector": vector,
                    "doc_id": doc_id,
                    "user_id": user_id,
                    "chunk_id": ind,
                    "text": chunk,
                }
            )

        client.insert(collection_name=COLLECTION_NAME, data=payload)
        return True
    except Exception as e:
        print(f" [!] Error processing document: {file_path}: {e}")
        return False

def start_worker(queue_name= 'document_queue'):
    """
    Sets up the connection to RabbitMQ running locally on docker and starts listening for tasks
    """
    global connection
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    channel.queue_declare(queue=queue_name, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=queue_name, on_message_callback=create_callback(client))

    print(f" [*] Connected to queue '{queue_name}'. Waiting for requests. To exit press CTRL+C")
    channel.start_consuming()

if __name__ == '__main__':
    try:
        client = MilvusClient(uri=MILVUS_ADDRESS)
        create_collection_if_not_exists(client)

        start_worker()
    except KeyboardInterrupt:
        client.close()
        connection.close()
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
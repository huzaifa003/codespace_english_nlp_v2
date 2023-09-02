
from haystack.utils import fetch_archive_from_http, convert_files_to_docs, clean_wiki_text
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, EmbeddingRetriever, FARMReader
from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers
from haystack.pipelines import Pipeline
from haystack.nodes import JoinDocuments
import pickle


document_store = InMemoryDocumentStore(use_bm25=True)
bm25_retriever = BM25Retriever()
p_retrieval = DocumentSearchPipeline(bm25_retriever)
# Initialize Sparse Retriever

doc_dir = "content/data/tutorial11/Mufti Taqi Usmani and Sahih International data with haystack format"

got_docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
document_store.delete_documents()

document_store.write_documents(got_docs)
bm25_retriever = BM25Retriever(document_store=document_store)
# Initialize embedding Retriever
embedding_retriever = EmbeddingRetriever(
    document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)
# document_store.update_embeddings(embedding_retriever, update_existing_embeddings=False)

document_store = pickle.load(open("document_store.pkl", "rb"))
print(document_store)

# Initialize Reader
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# def start():
#     global document_store
#     global bm25_retriever
#     global embedding_retriever
#     global reader
#     global p_retrieval
#     document_store = InMemoryDocumentStore(use_bm25=True)
#     bm25_retriever = BM25Retriever()
#     p_retrieval = DocumentSearchPipeline(bm25_retriever)
#     # Initialize Sparse Retriever
#     print(document_store)

#     doc_dir = "/content/data/tutorial11"
#     got_docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
#     # document_store.delete_documents()
#     document_store.write_documents(got_docs)
#     bm25_retriever = BM25Retriever(document_store=document_store)
#     # Initialize embedding Retriever
#     embedding_retriever = EmbeddingRetriever(
#         document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
#     )
#     document_store.update_embeddings(embedding_retriever, update_existing_embeddings=False)

#     # Initialize Reader
#     reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

def documentSearch(query):
#   global embedding_retriever
  
  global p_retrieval
  p_retrieval = DocumentSearchPipeline(embedding_retriever)
  res = p_retrieval.run(query=query, params={"Retriever": {"top_k": 5}})
  print_documents(res, max_text_len=500)
  return res

def questionAnswer(question):
  
  # Create ensembled pipeline
#   global document_store
#   global bm25_retriever
#   global embedding_retriever
#   global reader
  p_ensemble = Pipeline()
  p_ensemble.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["Query"])
  p_ensemble.add_node(component=embedding_retriever, name="EmbeddingRetriever", inputs=["Query"])
  p_ensemble.add_node(
      component=JoinDocuments(join_mode="concatenate"), name="JoinResults", inputs=["BM25Retriever", "EmbeddingRetriever"]
  )
  p_ensemble.add_node(component=reader, name="Reader", inputs=["JoinResults"])

  # Uncomment the following to generate the pipeline image
  # p_ensemble.draw("pipeline_ensemble.png")

  # Run pipeline
  res = p_ensemble.run(
      query= question, params={"EmbeddingRetriever": {"top_k": 5}, "BM25Retriever": {"top_k": 5}}
  )
  print_answers(res, details="all")
  return res

   

def extractiveQA(query):
#   global reader
#   global bm25_retriever
  p_extractive_premade = ExtractiveQAPipeline(reader=reader, retriever=bm25_retriever)
  res = p_extractive_premade.run(
      query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
  )
  print_answers(res, details="all")


    
def predict(query):
    # question = "times of prayer"
    answers = questionAnswer(query)
    documents = documentSearch(query)
    return {"answers" : answers, "documents" : documents}





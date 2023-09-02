import pandas as pd
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline

data = []
document_store = None
reader = None
retriever = None

def start():
    df = pd.read_excel("Mufti Taqi Usmani and Dr Muhsin.xlsx")
    data = []
    for i in range(len(df)):
        # print(df.loc[i,'Dr Mohsin Khan'])
        mohsin = {
            "content": df.loc[i, 'Dr Mohsin Khan'],
            "meta" : {"translator": "Dr Mohsin Khan","SuraID":str(df.loc[i, 'SuraID']), "AyaNo" : str(df.loc[i, 'AyaNo']), "SurahNameE" : df.loc[i,'SurahNameE'], "SurahNameU" : df.loc[i,'SurahNameU']}
                    
        }

        Taqi = {
            "content": df.loc[i, 'Mufti Taqi Usmani'],
            "meta" : {"translator": "Mufti Taqi Usmani","SuraID":str(df.loc[i, 'SuraID']), "AyaNo" : str(df.loc[i, 'AyaNo']), "SurahNameE" : df.loc[i,'SurahNameE'], "SurahNameU" : df.loc[i,'SurahNameU']}
                    
        }

        data.append(mohsin)
        data.append(Taqi)

    print(data[0])
    global document_store
    document_store = InMemoryDocumentStore(use_bm25=True)
    document_store.write_documents(data)

    global reader
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

    global retreiver
    retriever = BM25Retriever(document_store=document_store)
    
def predict(query):
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
    retriever = BM25Retriever(document_store=document_store)
    print(document_store)
    pipe = ExtractiveQAPipeline(reader, retriever)
    prediction = pipe.run(
        query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
    )
    return prediction



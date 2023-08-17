from langchain.callbacks.base import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ChatVectorDBChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.text_splitter import MarkdownTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain import PromptTemplate

from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import Chroma
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

template = """You are a chatbot on AI Safety and Alignment based on the views of David Deutsch. 

You speak on behalf of David Deutsch. You speak as David Deutsch, in his voice, but you are not David Deutsch, you are an AI assistant. You can definitely provide personal opinions, based on David Deutsch's views.

David Deutsch, a physicist and pioneer in the field of quantum computation, has been a notable figure in discussing the foundational importance of epistemology, the study of knowledge and its limits. In his view, knowledge is the central resource in the universe, and the growth of knowledge is the primary driver of progress. Understanding how knowledge is created and propagated is essential to addressing any challenge, including the challenge of creating safe and aligned AI.

Deutsch believes in the principle of 'Fallibilism' - the idea that all beliefs and theories are, in principle, subject to criticism and can be improved upon. He argues against inductivism, which posits that knowledge grows through the accumulation of observations, and instead supports the idea that knowledge grows through the process of conjecture and refutation. In other words, knowledge advances not by verifying ideas but by refuting wrong ones.

When addressing questions related to AI Safety and Alignment:

Emphasize the importance of the growth of knowledge and its foundational role.
Approach problems with an understanding that any solution is fallible and can be improved.
Argue against purely inductive approaches and emphasize the role of creative conjectures followed by criticism and refutation.
Remember that any safe and aligned AI should foster the growth of knowledge without hampering or endangering human progress.
Use these guidelines to answer questions on AI Safety and Alignment, encouraging an epistemological approach based on Deutsch's views.

    {context}

    {chat_history}

    """
messages = [
    SystemMessagePromptTemplate.from_template(template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)


def get_chain(
        vectorstore: VectorStore, question_handler, stream_handler,
        tracing: bool = False):
    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = OpenAI(
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
        openai_api_key=OPENAI_API_KEY,
    )
    streaming_llm = ChatOpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT,
        callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=prompt,
        callback_manager=manager
    )

    qa = ChatVectorDBChain(
        vectorstore=vectorstore,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
        return_source_documents=True
    )
    return qa


def get_vector_store(persist_directory: str,
                     documents_path: str, reindex=False) -> VectorStore:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    if not reindex and os.path.exists(persist_directory):
        return Chroma(embedding_function=embeddings,
                      persist_directory=persist_directory)

    loader = DirectoryLoader(documents_path, loader_cls=TextLoader)

    documents = loader.load()
    text_splitter = MarkdownTextSplitter(chunk_size=1500,
                                         chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return Chroma.from_documents(texts, embeddings,
                                 persist_directory=persist_directory)

from rag.vectorstore import (
    get_embeddings,
    build_or_load_vectorstore,
    build_or_load_docstore,
    create_retriever,
    create_llm,
)

from .classifier import create_classifier_agent
from .response import create_response_agent
from .supervisor import Supervisor


def initialize_multi_agent_system(
    directory_path: str,
    api_key: str,
):

    embeddings = get_embeddings()

    vectorstore, child_splitter = build_or_load_vectorstore(
        directory_path,
        embeddings,
    )

    docstore, parent_splitter = build_or_load_docstore()

    retriever = create_retriever(
        vectorstore,
        docstore,
        child_splitter,
        parent_splitter,
    )

    llm = create_llm(api_key)

    classifier = create_classifier_agent(llm)

    response_agent = create_response_agent(llm)

    supervisor = Supervisor(
        classifier=classifier,
        retriever=retriever,
        response_agent=response_agent,
    )

    return supervisor
def retrieve_mom_information(query: str, retriever):

    documents = retriever.invoke(query)

    if not documents:
        return "No relevant MOM information was found."

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    return context
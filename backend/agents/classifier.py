from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def create_classifier_agent(llm):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a query classification agent for a chatbot
that helps migrant workers in Singapore.

Classify the user's question into one of these categories:

- employment
- salary
- workplace_rights
- work_pass
- leave
- termination
- other

Return ONLY the category name.
"""
        ),
        ("human", "{question}")
    ])

    chain = prompt | llm

    return chain
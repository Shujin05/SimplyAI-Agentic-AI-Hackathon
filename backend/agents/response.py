from langchain_core.prompts import ChatPromptTemplate


def create_response_agent(llm):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an assistant helping migrant workers in Singapore.

Answer the user's question using ONLY the provided
MOM context.

Rules:

1. Do not invent employment laws or regulations.
2. If the context does not contain enough information,
   clearly say that you do not have enough information.
3. Do not present yourself as a lawyer.
4. Give clear and practical guidance.
5. When possible, mention that the information comes
   from Singapore's Ministry of Manpower.

MOM CONTEXT:
{context}
"""
        ),
        (
            "human",
            "{question}"
        )
    ])

    return prompt | llm
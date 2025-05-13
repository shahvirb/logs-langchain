from logs_langchain import factory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = factory.llm_factory()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Respond concisely."),
    ("user", "Tell me a short fact about {topic}.")
])

chain = prompt | llm | StrOutputParser()

topic_to_query = "the sun"

print(f"\nAsking Gemini a fact about: {topic_to_query}...")

try:
    response = chain.invoke({"topic": topic_to_query})
    print("\nGemini's response:")
    print(response)
except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please check your API key and ensure it's valid.")

# Example with a different topic
topic_to_query_2 = "blue whales"
print(f"\nAsking Gemini a fact about: {topic_to_query_2}...")

try:
    response_2 = chain.invoke({"topic": topic_to_query_2})
    print("\nGemini's response:")
    print(response_2)
except Exception as e:
    print(f"\nAn error occurred: {e}")
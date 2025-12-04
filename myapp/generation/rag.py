import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


class RAGGenerator:

    PROMPT_TEMPLATE = """
        You are an expert product advisor helping users choose the best option from retrieved e-commerce products.

        ## Instructions:
        1. Only recommend products that strictly match the user query criteria.
        2. 2. Include metadata: price, rating, stock, category.
        3. Present the recommendation clearly in this format:
        - Best Product: [Product PID] [Product Name]
        - Why: [Explain in plain language why this product is the best fit, referring to specific attributes like price, features, quality, or fit to user’s needs.]
        4. If there is another product that could also work, mention it briefly as an alternative.
        5. If no product is a good fit, return EXACTLY:
        "There are no good products that fit the request based on the retrieved results."
        ## Retrieved Products:
        {retrieved_results}

        ## User Request:
        {user_query}

        ## Output Format:
        - Best Product: ...
        - Why: ...
        - Alternative (optional): ...
    """

    def generate_response(self, user_query: str, retrieved_results: list, top_N: int = 20) -> dict:
        """
        Generate a response using the retrieved search results. 
        Returns:
            dict: Contains the generated suggestion and the quality evaluation.
        """
        DEFAULT_ANSWER = "RAG is not available. Check your credentials (.env file) or account limits."
        try:
            filtered_results = [res for res in retrieved_results if not res.out_of_stock and (res.average_rating >= 3.0)]
            if not filtered_results:
                return "There are no good products that fit the request based on the retrieved results."
            
            client = Groq(
                api_key=os.environ.get("GROQ_API_KEY"),
            )
            model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

            # Format the retrieved results for the prompt
            formatted_results = "\n".join(
                [f"- PID: {res.pid}, Title: {res.title}, Price: {res.selling_price}, Rating: {res.average_rating}, Stock: {'Yes' if not res.out_of_stock else 'No'}" for res in filtered_results[:top_N]]
            )

            prompt = self.PROMPT_TEMPLATE.format(
                retrieved_results=formatted_results,
                user_query=user_query
            )

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
            )

            generation = chat_completion.choices[0].message.content
            return generation
        except Exception as e:
            print(f"Error during RAG generation: {e}")
            return DEFAULT_ANSWER

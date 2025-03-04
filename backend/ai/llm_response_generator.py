import os
#import openai
import asyncio
import logging
from typing import List

class LLMResponseGenerator:
    def __init__(self, model="gpt-4"):
        """Initialize LLM Client"""
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY", "your_api_key_here")  # Replace with actual API Key
        self.local_llm = os.getenv("USE_LOCAL_LLM", "False").lower() == "true"

    async def generate_response(self, query: str, context: List[str], stream: bool = False):
        """Generate LLM response based on retrieved context."""

        full_prompt = f"""
        You are an AI assistant that answers user queries based on the provided context.
        Context: {context}
        User Query: {query}
        Response:
        """

        if self.local_llm:
            return await self._generate_response_local(full_prompt, stream)
        return "Sorry, I'm currently unable to generate responses."

        #return await self._generate_response_openai(full_prompt, stream)

    async def generate_response_without_context(self, query: str, stream: bool = False):
        """Generate LLM response based on retrieved context."""

        full_prompt = f"""
        You are an AI assistant that answers user queries based on the provided context.
        User Query: {query}
        Response:
        """

        if self.local_llm:
            return await self._generate_response_local(full_prompt, stream)
        return "Sorry, I'm currently unable to generate responses."


    """ async def _generate_response_openai(self, prompt: str, stream: bool):
       
        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=self.model,
                messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                          {"role": "user", "content": prompt}],
                temperature=0.7,
                stream=stream
            )

            if stream:
                return self._stream_response(response)
            return response["choices"][0]["message"]["content"]

        except Exception as e:
            logging.error(f"Error generating LLM response: {str(e)}")
            return "Sorry, I encountered an error while generating a response." """

    async def _generate_response_local(self, prompt: str, stream: bool):
        """Generate response using a local LLM model (DeepSeek, Mistral, etc.)."""
        try:
            # Simulate local model response (Replace with actual local model call)
            await asyncio.sleep(1)
            return f"[Local Model] Simulated response for: {prompt[:100]}..."

        except Exception as e:
            logging.error(f"Local LLM error: {str(e)}")
            return "Sorry, I encountered an issue with the local model."

    async def _stream_response(self, response):
        """Stream LLM responses for real-time updates."""
        async for chunk in response:
            yield chunk["choices"][0]["delta"].get("content", "")
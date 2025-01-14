import pprint
import os
import time
import logging

from .Base import BaseModel
from llama_index.llms.ollama import Ollama

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Codellama(BaseModel):
    def __init__(self, temperature=0):
        self.model = Ollama(
            model="codellama",
            request_timeout=600,
            temperature=temperature,
        )
        self.temperature = temperature

    def prompt(self, processed_input):
        response = None
        for i in range(1):
            try:
                response = self.model.complete(
                    processed_input[0]["content"]
                    # Pass temperature if supported
                )
                return response.text, 0, 0
            except Exception as e:
                logger.error(f"Attempt {i+1}: Error generating content - {e}")
                time.sleep(2)

        if response:
            return response.text, 0, 0
        else:
            logger.error("Failed to generate content after 10 attempts.")
            return "", 0, 0

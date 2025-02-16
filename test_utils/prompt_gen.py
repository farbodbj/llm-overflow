import random
import re
from typing import List

class RandomPromptGen:
    def __init__(self, corpus: List[str]):
        self._corpus = corpus
        self.templates: List[str] = [
            "Explain the following text in great detail, avoid short answers: {text}",
            "Summarize the following text in one sentence: {text}",
            "What are the key points in the following text? {text}",
            "Rewrite the following text in a more formal tone: {text}",
            "Translate the following text into a casual, conversational style: {text}",
            "Analyze the following text and provide your interpretation: {text}",
            "What is the main idea of the following text? {text}",
            "Break down the following text into bullet points: {text}",
            "What emotions or tone does the following text convey? {text}",
            "Create a question based on the following text: {text}",
            "What assumptions are made in the following text? {text}",
            "How would you simplify the following text for a child? {text}",
            "What are the potential implications of the following text? {text}",
            "What is the author trying to achieve in the following text? {text}",
            "What are the strengths and weaknesses of the following text? {text}",
            "How does the following text relate to current events? {text}",
            "What would you add to the following text to make it more comprehensive? {text}",
            "What is the most surprising part of the following text? {text}",
            "How would you argue against the following text? {text}",
            "What is the historical context of the following text? {text}",
        ]
    
    def get_prompt(self) -> str:
        """Generates a random prompt using a template and a random text from the corpus."""
        template = random.choice(self.templates)
        text = random.choice(self._corpus)
        return template.format(text=text)
    
class CorpusReader:    
    def read(self, path: str)->List[str]:
        with open(path, 'r') as f:
            text = ''.join(f.readlines())
            sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)(\s|[A-Z].*)',text)
            return [sentence for sentence in sentences if len(sentence) > 1]
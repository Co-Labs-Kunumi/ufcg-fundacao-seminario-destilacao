import json
import os
import time

from dotenv import load_dotenv  # pip3 install python-dotenv
from openai import OpenAI  # pip3 install openai

from tokens.cost_calculator import CostCalculator


class Distillation:

    def __init__(self):
        self.openai = self._initialize_openai()
        self.calculator = CostCalculator()

    def _initialize_openai(self):
        """
        Initializes and returns an instance of the OpenAI API configured for DeepInfra.
        Automatically loads the key from the DEEPINFRA_API_KEY environment variable.
        """
        load_dotenv()
        return OpenAI(
            api_key=os.getenv("DEEPINFRA_API_KEY"),  # export DEEPINFRA_API_KEY="..."
            base_url="https://api.deepinfra.com/v1/openai",
        )

    def load_questions(self, file_path):
        """
        Loads and returns the content of a JSON file containing the questions.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_questions_with_reasoning(self, questions, output_path):
        """
        Saves a list of questions, now with explanations, to a JSON file.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)

    def _generate_explanation(self, base_prompt, question, model):
        """
        Generates the explanation for a question using the DeepInfra API via streaming.
        """
        instruction = "ATTENTION: REASON IN PORTUGUESE! DO NOT USE ENGLISH AT ANY POINT. THINK OUT LOUD: 'Estou pensando que...'\n"
        user_message = (
            f"POSCOMP Question {question['id']}\n\n{question['enunciado']}\n\n"
            + "\n".join(question["alternativas"])
            + f"\n\n{instruction}"
        )

        print(f"\n🔹 Processing question {question['id']}...\n")

        try:
            stream = self.openai.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": base_prompt + " USE PORTUGUESE LANGUAGE",
                    },
                    {
                        "role": "user",
                        "content": f"{user_message}\nPlease explain the reasoning in Portuguese.",
                    },
                ],
                temperature=0.4,
                stream=True,
            )

            explanation = ""
            usage = None

            for chunk in stream:
                if chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    explanation += content
                if chunk.usage:
                    usage = chunk.usage
            return explanation, usage

        except Exception as e:
            error = f"[ERROR WHILE QUERYING THE MODEL]: {str(e)}"
            print(error)
            return error, None

    def process_questions(self, output_path, questions, base_prompt, model):
        questions_with_reasoning = []
        start_time = time.time()

        for question in questions:
            explanation, usage = self._generate_explanation(
                base_prompt, question, model
            )

            question_with_reasoning = question.copy()
            question_with_reasoning["reasoning"] = explanation  # changed key to English
            questions_with_reasoning.append(question_with_reasoning)

            if usage:
                self.calculator.add_tokens(usage.prompt_tokens, usage.completion_tokens)

            # Save the questions processed so far
            self.save_questions_with_reasoning(questions_with_reasoning, output_path)

        total_time = time.time() - start_time
        self.calculator.print_summary(total_time)

        return questions_with_reasoning

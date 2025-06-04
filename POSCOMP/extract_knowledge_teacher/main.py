from distillation.distillation_teacher import Distillation
from prompts.prompt import PromptPOSCOMP


def main():

    # Lembra de mudar os custos na classe tokens/cost_calculator.py
    # Lembra do export DEEPINFRA_API_KEY="...""

    # É interessante rodar o teste.json antes do amostra_poscomp.json
    # Verificar se tá tudo certo no ambiente
    input_path = "data/teste.json"
    output_path = "data/raciocinio.json"

    distiller = Distillation()
    questions = distiller.load_questions(input_path)

    base_prompt_teacher = PromptPOSCOMP.get()
    teacher_model = "Qwen/Qwen3-32B"
    distiller.process_questions(output_path, questions, base_prompt_teacher, teacher_model)


if __name__ == "__main__":
    main()

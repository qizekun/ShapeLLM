import time
import openai

NUM_SECONDS_TO_SLEEP = 1.0

openai.api_base = ""  # if needed
openai.api_key = ""


def gpt_get_score(question, question_type, answer_model, answer_label, model="gpt-3.5-turbo"):
    while True:
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": """
                            Now I will give you a question, the type of the question, an answer from model, and an answer from label. 
                            All you need to do is focus on these two answers and figure out whether they are saying the same thing about the specific type of question.
                            Your response should only be a confidence score ranging from 0 to 100. 
                            Remember the confidence score is to evaluate how much two answers are describing the same thing.
                            Your response confidence score should follow the scoring standard of the prompt I gave.
                            Firstly I will give you several question & answer pairs as long as their confidence score:
                            
                            question1: How many oranges will there be if 1/3 of them are removed?
                            question type: Knowledge
                            answer from model: There will be 6 left.
                            answer from label: As there are 9 oranges in total, there will be 6 oranges left if 1/3 of them are removed.
                            confidence score: 100
                            
                            question2: What is this object?
                            question type: General Visual Recognition
                            answer from model: This is a bathtub
                            answer from label: This is a dirty bathtub.
                            confidence score: 80
                            
                            question3: What is this object?
                            question type: General Visual Recognition
                            answer from model: This is a bottle of water
                            answer from label: This is a bottle of oil
                            confidence score: 50
                            
                            question4: What is holding in this boy's right hand?
                            question type: Spatial Recognition
                            answer from model: He is holding a white cup in his right hand.
                            answer from label: He is holding a sword in his right hand.
                            confidence score: 0
                            
                            Next, I will give you the elements:
                            question: {},
                            question type: {},
                            answer from model: {},
                            answer from label: {}.
                            Please remember, while outputting the confidence score, do not include any words, just the number.
                    """.format(question, question_type, answer_model, answer_label)},
                ]
            )
            response = completion.choices[0].message["content"]
            break
        except:
            pass
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response


def is_valid(value):
    try:
        value = float(value)
        if 0 <= value <= 100:
            return True
        else:
            return False
    except ValueError:
        return False


def gpt_get_average_score(question, question_type, answer_model, answer_label, model="gpt-3.5-turbo", times=5):
    scores = []
    while len(scores) < times:
        score = gpt_get_score(question, question_type, answer_model, answer_label, model)
        if is_valid(score):
            scores.append(float(score))
    return sum(scores) / len(scores)


if __name__ == "__main__":
    score = gpt_get_score(
        question="How many oranges will there be if 1/3 of them are removed?",
        question_type="Knowledge",
        answer_model="There will be 6 left.",
        answer_label="As there are 9 oranges in total, there will be 6 oranges left if 1/3 of them are removed.",
        model="gpt-3.5-turbo"
    )
    print(score)

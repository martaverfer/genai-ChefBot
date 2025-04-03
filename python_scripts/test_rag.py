import openai
from query_data import query_rag

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def test_definitions():
    assert query_and_validate(
        question="What does it mean 'al dente'?",
        expected_response="Pasta cooked until just firm",
    )


def test_winter():
    assert not query_and_validate(
        question="Tell me one fall season fruit.",
        expected_response="Blueberries",
    )

def test_out_of_context():
    assert not query_and_validate(
        question="In which year was the French Revolution?",
        expected_response="1789",
    )

def test_cook_time():
    assert query_and_validate(
        question="What is the cook time for making Ham and Cheese Oven Sandwiches?",
        expected_response="20-25 minutes",
    )

def test_meal():
    assert query_and_validate(
        question="What makes a meal?",
        expected_response="Protein, Grain and Produce (vegetables/fruit)",
    )

def test_waffles():
    assert query_and_validate(
        question="How can I make waffles?",
        expected_response=""" Waffle Recipe
### Ingredients
- 1 ¾ cup flour (white, wheat, or a combination of both flours)
- 1 tablespoon baking powder
- ½ teaspoon salt
- 2 egg yolks (save the whites!)
- 1 ¾ cups milk
- ½ cup cooking oil (or substitute with plain low-fat yogurt)
- 2 egg whites

### Instructions
1. In a large mixing bowl, stir together the flour, baking powder, and salt.
2. In a small mixing bowl, beat the egg yolks with a fork. Beat in milk and cooking oil (or yogurt).
3. Add the wet mixture to the flour mixture all at once. Stir until blended but still slightly lumpy.
4. In another smaller bowl, beat the egg whites with an electric beater until stiff peaks form.
5. Gently fold the beaten egg whites into the flour-milk mixture, leaving a few fluffs of egg white visible. Do not overmix.
6. Pour batter onto a preheated waffle iron.
        """,
    )

def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )
    
    client = openai.OpenAI()
    model_params = {
        'model': 'gpt-4o',
        'temperature': 0.7,  # Increase creativity
        'max_tokens': 4000,  # Allow for longer responses
        'top_p': 0.9,        # Use nucleus samplingtest
        'frequency_penalty': 0.5,  # Reduce repetition
        'presence_penalty': 0.6    # Encourage new topics
    }

    messages = [{'role': 'user', 'content': prompt}]
    completion = client.chat.completions.create(messages=messages, **model_params, timeout=120)
    answer = completion.choices[0].message.content
    evaluation_results_str_cleaned = answer.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )


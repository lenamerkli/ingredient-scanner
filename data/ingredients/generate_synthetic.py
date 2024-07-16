from tqdm import tqdm

import os
import random
import json


def relative_path(string: str) -> str:
    """
    Returns the absolute path of a given string by joining it with the directory of the current file.
    :param string: relative path
    :return: absolute path
    """
    return os.path.join(os.path.dirname(__file__), string)


SYSTEM_PROMPT: str = 'Du bist ein hilfreicher assistant.'
with open(relative_path('prompt.md'), 'r', encoding='utf-8') as _f:
    PROMPT: str = (f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
                   f"<|im_start|>user\n{_f.read()}\n"
                   f"<|im_end|>\n<|im_start|>assistant\n")
EOS: str = '\n<|im_end|>'
ITERATIONS: int = 10000  # number of examples to generate
SEED: int = 2024 - 7 - 28  # set seed for prediction


def main() -> None:
    """
    Generates synthetic_ingredients data from the 'lists' directory and saves it in the 'synthetic_ingredients'
    directory.
    :return: None
    """
    random.seed(SEED)
    # load lists
    with open(relative_path('lists/ingredients_label.json'), 'r', encoding='utf-8') as f:
        ingredients_label: list[str] = json.load(f)
    with open(relative_path('lists/no_contaminants.json'), 'r', encoding='utf-8') as f:
        no_contaminants: list[str] = json.load(f)
    with open(relative_path('lists/animal.json'), 'r', encoding='utf-8') as f:
        animal: list[str] = json.load(f)
    with open(relative_path('lists/sometimes_animal.json'), 'r', encoding='utf-8') as f:
        sometimes_animal: list[str] = json.load(f)
    with open(relative_path('lists/non_animal.json'), 'r', encoding='utf-8') as f:
        non_animal: list[str] = json.load(f)
    combined = animal + sometimes_animal + non_animal
    # remove vitamins and e-numbers, because they are added later
    to_remove = []
    for item in combined:
        if any(i in item for i in '0123456789'):
            to_remove.append(item)
    for item in to_remove:
        combined.remove(item)
    data = []
    for _ in tqdm(range(ITERATIONS)):
        num_ingredients = random.randint(1, 12)
        ingredients = random.sample(combined, num_ingredients)
        num_contaminants = random.randint(0, 6)
        contaminants = random.sample(combined, num_contaminants)
        if random.random() < 0.4:
            # add e-numbers
            num_e_numbers = random.randint(1, 8)
            e_numbers = [f"E{random.randint(100, 1200)}" for _ in range(num_e_numbers)]
            new_e_numbers = []
            for item in e_numbers:
                if item not in new_e_numbers:
                    new_e_numbers.append(item)
            e_numbers = new_e_numbers
            ingredients.extend(e_numbers)
        if random.random() < 0.4:
            # add vitamins
            num_vitamins = random.randint(1, 4)
            vitamins = [f"{random.choice('ABCD')}{random.randint(1, 12)}" for _ in range(num_vitamins)]
            new_vitamins = []
            for item in vitamins:
                if item not in new_vitamins:
                    new_vitamins.append(item)
            vitamins = new_vitamins
            if random.random() < 0.5:
                # add vitamins individually
                new_ingredients = ingredients + [f"Vitamin {item}" for item in vitamins]
            else:
                # add vitamins together
                new_ingredients = ingredients + [f"Vitamine ({', '.join(vitamins)})"]
        else:
            vitamins = []
            new_ingredients = ingredients
        solution = {
            'Zutaten': ingredients + [f"Vitamin {item}" for item in vitamins],
            'Verunreinigungen': contaminants,
        }
        instruction = random.choice(ingredients_label)
        ingredients = new_ingredients
        for j in range(len(ingredients)):
            if random.random() < 0.1:
                # convert to lowercase
                ingredients[j] = ingredients[j].lower()
            elif random.random() < 0.1:
                # convert to uppercase
                ingredients[j] = ingredients[j].upper()
            if random.random() < 0.3:
                # add asterisks
                num_asterisks = random.randint(1, 3)
                ingredients[j] = ingredients[j] + '*' * num_asterisks
            if random.random() < 0.3:
                # add non-essential information about the ingredient
                rand_value = str(random.randint(1, 90))
                if random.random() < 0.5:
                    # add decimals
                    rand_value = rand_value + '.' + str(random.randint(1, 9))
                if random.random() < 0.2:
                    ingredients[j] = ingredients[j] + f" ({rand_value}%)"
                elif random.random() < 0.2:
                    ingredients[j] = ingredients[j] + f" ({rand_value} g)"
                else:
                    ingredients[j] = ingredients[j] + f" ({rand_value} ml)"
                if random.random() < 0.5:
                    ingredients[j] = ingredients[j].replace(' (', '').replace(')', '')
            if random.random() < 0.2:
                # add a random prefix
                choice = random.choice(['Emulgator', 'Farbstoff'])
                ingredients[j] = choice + ': ' + ingredients[j]
        if random.random() < 0.7:
            instruction += '\n' + '\n'.join(ingredients)
        elif random.random() < 0.5:
            instruction += '\n' + ', '.join(ingredients)
        else:
            instruction += ' ' + ', '.join(ingredients)
        instruction += '\n'
        if len(contaminants) > 0:
            instruction += 'Verunreinigungen:'
            if random.random() < 0.5:
                instruction += '\n' + '\n'.join(contaminants)
            elif random.random() < 0.3:
                instruction += '\n' + ', '.join(contaminants)
            elif random.random() < 0.3:
                instruction += ' ' + ', '.join(contaminants)
            else:
                instruction += '\nKann ' + ', '.join(contaminants) + ' enthalten.'
        else:
            instruction += random.choice(no_contaminants)
        example = PROMPT.replace('{{old_data}}', instruction) + '```json\n' + json.dumps(solution) + '\n```' + EOS
        data.append(example)
    with open(relative_path('synthetic/train.jsonl'), 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps({'text': d}) + '\n')


if __name__ == '__main__':
    main()

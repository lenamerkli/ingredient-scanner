import shutil
import os
from general import relative_path


def main() -> None:
    """
    Build the project, including copying necessary files. Does not remove any files, only replaces them.
    :return: None
    """
    # copy files
    for file in os.listdir(relative_path('build_artifacts')):
        if not file.startswith('.'):
            shutil.copy(relative_path(f"build_artifacts/{file}"), relative_path(f"build/{file}"))
    # select latest version of the vision model
    vision_model = None
    for file in sorted(os.listdir(relative_path('models')), reverse=True):
        if file.endswith('.pt'):
            vision_model = file
            break
    if vision_model is None:
        raise Exception('vision model not found')
    print(f'{vision_model=}')
    # copy all other necessary files
    shutil.copy(relative_path(f"models/{vision_model}"), relative_path('build/vision_model.pt'))
    shutil.copy(relative_path('.env'), relative_path('build/.env'))
    shutil.copy(relative_path('llm_models/unsloth.Q4_K_M.gguf'), relative_path('build/llm.Q4_K_M.gguf'))
    shutil.copy(relative_path('data/ingredients/lists/animal.json'), relative_path('build/animal.json'))
    shutil.copy(relative_path('data/ingredients/lists/sometimes_animal.json'),
                relative_path('build/sometimes_animal.json'))
    shutil.copy(relative_path('data/ingredients/lists/milk.json'), relative_path('build/milk.json'))
    shutil.copy(relative_path('data/ingredients/lists/gluten.json'), relative_path('build/gluten.json'))
    shutil.copy(relative_path('latex/ingredient-scanner.pdf'), relative_path('build/ingredient-scanner.pdf'))


if __name__ == '__main__':
    main()
